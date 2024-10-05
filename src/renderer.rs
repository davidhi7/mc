use std::{
    mem,
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingType, BlendState, Buffer, BufferAddress,
    BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    CompareFunction, DepthBiasState, DepthStencilState, Device, Face, FragmentState, FrontFace,
    MultisampleState, PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology,
    Queue, RenderPass, RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StencilState, SurfaceConfiguration, TextureFormat, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
};

use crate::{
    renderer::ui_renderer::Reticle,
    texture,
    world::{camera::CameraController, World, CHUNK_DIMENSIONS, VERTICAL_CHUNK_COUNT},
};

mod ui_renderer;

const CHUNK_RENDER_DISTANCE: i32 = 4;

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
    pub chunk: [i32; 3],
    pub attributes: u32,
}
impl CubeFaceInstance {
    fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: (mem::size_of::<[i32; 3]>() + mem::size_of::<u32>()) as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: VertexFormat::Sint32x3,
                },
                VertexAttribute {
                    offset: mem::size_of::<[i32; 3]>() as BufferAddress,
                    shader_location: 3,
                    format: VertexFormat::Uint32,
                },
            ],
        }
    }
}

pub struct WorldRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    vertex_buffer: Buffer,
    instance_buffer: Buffer,
    pub camera_controller: CameraController,
    camera_uniform: Buffer,
    camera_bind_group: BindGroup,
    texture_bind_group: BindGroup,
    render_pipeline: RenderPipeline,
    buffer_capacity: usize,
    previous_camera_u: Option<i32>,
    previous_camera_w: Option<i32>,
    reticle_renderer: ui_renderer::Reticle,

    loading_thread_handle: Vec<JoinHandle<Vec<CubeFaceInstance>>>,
}

impl WorldRenderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface_config: &SurfaceConfiguration,
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("cube face vertex buffer"),
            contents: bytemuck::cast_slice(CUBE_FACE_VERTICES),
            usage: BufferUsages::VERTEX,
        });

        // TODO use sensible default size, research `mapped_at_creation`
        let instance_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("cube face instance buffer"),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            size: 0,
            mapped_at_creation: false,
        });

        let camera_controller = CameraController::new(
            glam::Vec3::NEG_X,
            -0.5,
            0.0,
            45.0,
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
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("camera bind group layout"),
            });

        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: camera_uniform.as_entire_binding(),
            }],
            label: Some("camera bind group"),
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("world shader"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let (texture_bind_group_layout, texture_bind_group) =
            texture::load_textures(&device, &queue).unwrap();

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("world render pipeline layout"),
            bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
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
        });

        let reticle_renderer =
            Reticle::new(&device, camera_bind_group_layout, surface_config.format);

        WorldRenderer {
            device,
            queue,
            vertex_buffer,
            instance_buffer,
            camera_controller,
            camera_uniform,
            camera_bind_group,
            texture_bind_group,
            render_pipeline,
            buffer_capacity: 0,
            previous_camera_u: None,
            previous_camera_w: None,
            reticle_renderer,

            loading_thread_handle: Vec::new(),
        }
    }

    pub fn update(&mut self, world: Arc<Mutex<World>>) {
        self.queue.write_buffer(
            &self.camera_uniform,
            0,
            bytemuck::cast_slice(&[self.camera_controller.get_view_projection_matrix()]),
        );

        if let Some(handle) = self.loading_thread_handle.pop() {
            if handle.is_finished() {
                let instances = handle.join().unwrap();
                if instances.len() > self.buffer_capacity {
                    self.instance_buffer.destroy();
                    self.instance_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
                        label: Some("cube face instance buffer"),
                        contents: bytemuck::cast_slice(instances.as_slice()),
                        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    });
                    self.buffer_capacity = instances.len();
                } else {
                    self.queue.write_buffer(
                        &self.instance_buffer,
                        0,
                        bytemuck::cast_slice(instances.as_slice()),
                    );
                }
            } else {
                self.loading_thread_handle.push(handle);
            }
        }

        let camera_u = self.camera_controller.get_position().x as i32 / CHUNK_DIMENSIONS;
        let camera_w = self.camera_controller.get_position().z as i32 / CHUNK_DIMENSIONS;

        if self.previous_camera_u.is_some_and(|u| u == camera_u)
            && self.previous_camera_w.is_some_and(|w| w == camera_w)
        {
            return;
        }

        self.previous_camera_u = Some(camera_u);
        self.previous_camera_w = Some(camera_w);

        let device_clone = Arc::clone(&self.device);
        // let instance_buffer_clone = self.instance_buffer;

        let handle = thread::spawn(move || {
            let chunk_range_u =
                camera_u - CHUNK_RENDER_DISTANCE..camera_u + CHUNK_RENDER_DISTANCE + 1;
            let chunk_range_w =
                camera_w - CHUNK_RENDER_DISTANCE..camera_w + CHUNK_RENDER_DISTANCE + 1;

            let mut world_handle = world.lock().unwrap();

            for u in chunk_range_u.clone() {
                for w in chunk_range_w.clone() {
                    // TODO error handling
                    if world_handle.chunk_columns.get(&(u, w)).is_none() {
                        world_handle.create_chunks(u, w);
                    }
                }
            }

            let mut instances: Vec<&CubeFaceInstance> = Vec::new();
            for u in chunk_range_u.clone() {
                for w in chunk_range_w.clone() {
                    let instances_ = (0..VERTICAL_CHUNK_COUNT)
                        .flat_map(|v| world_handle.meshed_chunks.get(&(u, v as i32, w)).unwrap())
                        .collect::<Vec<&CubeFaceInstance>>();

                    instances.extend(instances_);
                }
            }

            let instances: Vec<CubeFaceInstance> = instances
                .iter()
                .map(|instance| (*instance).to_owned())
                .collect();

            instances
        });

        self.loading_thread_handle.push(handle);

        // let chunk_range_u = camera_u - CHUNK_RENDER_DISTANCE..camera_u + CHUNK_RENDER_DISTANCE + 1;
        // let chunk_range_w = camera_w - CHUNK_RENDER_DISTANCE..camera_w + CHUNK_RENDER_DISTANCE + 1;

        // let handle = thread::spawn(|| {
        //     let mut instances: Vec<&RawCubeFaceInstance> = Vec::new();

        //     world.chunk_columns.get(&(0, 0));
        // });

        // for u in chunk_range_u.clone() {
        //     for w in chunk_range_w.clone() {
        //         if let None = world.chunk_columns.get(&(u, w)) {
        //             world.create_chunks(u, w);
        //         }
        //     }
        // }

        // let instances: Vec<&RawCubeFaceInstance> = Vec::new();
        // for u in chunk_range_u {
        //     for w in chunk_range_w.clone() {
        //         instances.extend(
        //             (0..VERTICAL_CHUNK_COUNT)
        //                 .flat_map(|v| world.meshed_chunks.get(&(u, v as u32, w)).unwrap())
        //                 .collect::<Vec<&RawCubeFaceInstance>>(),
        //         );
        //     }
        // }

        // let instances: Vec<RawCubeFaceInstance> = instances
        //     .iter()
        //     .map(|instance| *instance.to_owned())
        //     .collect();

        // if instances.len() > self.buffer_capacity {
        //     self.instance_buffer.destroy();
        //     self.instance_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
        //         label: Some("cube face instance buffer"),
        //         contents: bytemuck::cast_slice(instances.as_slice()),
        //         usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        //     });
        //     self.buffer_capacity = instances.len();
        // } else {
        //     self.queue.write_buffer(
        //         &self.instance_buffer,
        //         0,
        //         bytemuck::cast_slice(instances.as_slice()),
        //     );
        // }
    }

    pub fn render<'a: 'b, 'b>(&'a self, render_pass: &mut RenderPass<'b>) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

        render_pass.draw(
            0..CUBE_FACE_VERTICES.len() as u32,
            0..self.buffer_capacity as u32,
        );

        self.reticle_renderer
            .render(render_pass, &self.camera_bind_group);
    }
}
