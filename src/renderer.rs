use std::{mem, sync::Arc};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendState, Buffer, BufferAddress, BufferBindingType,
    BufferUsages, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
    DepthStencilState, Device, Face, FragmentState, FrontFace, MultisampleState,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, Queue, RenderPass,
    RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    StencilState, SurfaceConfiguration, TextureFormat, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexState, VertexStepMode,
};

use crate::{
    renderer::ui_renderer::Reticle,
    texture,
    world::{camera::CameraController, world_loader::WorldLoader, World},
};

mod ui_renderer;

const CHUNK_RENDER_DISTANCE: u32 = 8;

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct Vertex {
    pub position: [f32; 3],
    pub tex_coordinates: [f32; 2],
}
impl Vertex {
    fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },
                VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as BufferAddress,
                    shader_location: 1,
                    format: VertexFormat::Float32x2,
                },
            ],
        }
    }
}

// Cube face pointing in negative Z direction
const CUBE_FACE_VERTICES: &[Vertex] = &[
    Vertex {
        position: [0.0, 0.0, 0.0],
        tex_coordinates: [0.0, 1.0],
    },
    Vertex {
        position: [0.0, 1.0, 0.0],
        tex_coordinates: [0.0, 0.0],
    },
    Vertex {
        position: [1.0, 0.0, 0.0],
        tex_coordinates: [1.0, 1.0],
    },
    Vertex {
        position: [1.0, 1.0, 0.0],
        tex_coordinates: [1.0, 0.0],
    },
];

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct CubeFaceInstance {
    pub attributes: u32,
}
impl CubeFaceInstance {
    fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: mem::size_of::<u32>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &[VertexAttribute {
                offset: 0 as BufferAddress,
                shader_location: 2,
                format: VertexFormat::Uint32,
            }],
        }
    }
}

pub struct WorldRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    vertex_buffer: Buffer,
    pub camera_controller: CameraController,
    camera_uniform: Buffer,
    camera_bind_group: BindGroup,
    chunk_bind_group_layout: BindGroupLayout,
    texture_bind_group: BindGroup,
    render_pipeline: RenderPipeline,
    reticle_renderer: ui_renderer::Reticle,
    world_loader: WorldLoader,
}

impl WorldRenderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface_config: &SurfaceConfiguration,
        world: World,
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("cube face vertex buffer"),
            contents: bytemuck::cast_slice(CUBE_FACE_VERTICES),
            usage: BufferUsages::VERTEX,
        });

        let camera_controller: CameraController = CameraController::new(
            glam::Vec3::NEG_X,
            -0.5,
            0.0,
            1.6,
            surface_config.width as f32 / surface_config.height as f32,
            0.1,
            1000.0,
            10.0,
            0.1,
        );

        let camera_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("camera uniform buffer"),
            contents: bytemuck::cast_slice(&[camera_controller.get_view_projection_matrix()]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("camera bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: camera_uniform.as_entire_binding(),
            }],
            label: Some("camera bind group"),
        });

        let chunk_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("chunk bind group layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("world shader"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let (texture_bind_group_layout, texture_bind_group) =
            texture::load_textures(&device, &queue).unwrap();

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("world render pipeline layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &camera_bind_group_layout,
                &chunk_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("world render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), CubeFaceInstance::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: surface_config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: FrontFace::Cw,
                cull_mode: Some(Face::Back),
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let reticle_renderer =
            Reticle::new(&device, camera_bind_group_layout, surface_config.format);

        WorldRenderer {
            device,
            queue,
            vertex_buffer,
            camera_controller,
            camera_uniform,
            camera_bind_group,
            chunk_bind_group_layout,
            texture_bind_group,
            render_pipeline,
            reticle_renderer,
            world_loader: WorldLoader::new(world, CHUNK_RENDER_DISTANCE),
        }
    }

    pub fn update(&mut self) {
        self.queue.write_buffer(
            &self.camera_uniform,
            0,
            bytemuck::cast_slice(&[self.camera_controller.get_view_projection_matrix()]),
        );

        self.world_loader
            .update(&self.camera_controller, Arc::clone(&self.device));
        self.world_loader.create_buffers(
            &self.camera_controller,
            Arc::clone(&self.device),
            &self.chunk_bind_group_layout,
        );
    }

    pub fn render<'a: 'b, 'b>(&'a self, render_pass: &mut RenderPass<'b>) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        let (range_u, range_v, range_w) = self
            .world_loader
            .visible_chunk_range(&self.camera_controller);

        for u in range_u {
            for w in range_w.clone() {
                for v in range_v.clone() {
                    if let Some(buf) = self.world_loader.get_buffer(u, v as i32, w) {
                        render_pass.set_bind_group(2, &buf.chunk_bind_group, &[]);
                        render_pass.set_vertex_buffer(1, buf.instance_buffer.slice(..));

                        render_pass.draw(0..CUBE_FACE_VERTICES.len() as u32, 0..buf.instance_count);
                    }
                }
            }
        }
        self.reticle_renderer
            .render(render_pass, &self.camera_bind_group);
    }
}
