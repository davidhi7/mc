use std::{collections::HashMap, sync::Arc};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendState, Buffer, BufferBindingType, BufferUsages,
    ColorTargetState, ColorWrites, CompareFunction, DepthBiasState, DepthStencilState, Device,
    Face, FragmentState, FrontFace, MultisampleState, PipelineLayoutDescriptor, PolygonMode,
    PrimitiveState, PrimitiveTopology, Queue, RenderPass, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState, SurfaceConfiguration,
    TextureFormat, VertexState,
};

use crate::{
    renderer::{
        indirect_buffer::MultiDrawIndirectBuffer,
        ui_renderer::Reticle,
        vertex_buffer::{QuadInstance, TransparentQuadInstance, QUAD_VERTEX_COUNT},
    },
    texture,
    world::{
        camera::CameraController,
        world_loader::{ChunkBuffers, WorldLoader},
        World,
    },
};

mod indirect_buffer;
mod ui_renderer;

pub mod vertex_buffer;
const CHUNK_RENDER_DISTANCE: u32 = 8;

pub struct WorldRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pub camera_controller: CameraController,
    vertex_bind_group: BindGroup,
    camera_uniform: Buffer,
    camera_bind_group: BindGroup,
    chunk_bind_group_layout: BindGroupLayout,
    texture_bind_group: BindGroup,
    render_pipeline: RenderPipeline,
    water_render_pipeline: RenderPipeline,
    reticle_renderer: ui_renderer::Reticle,
    world_loader: WorldLoader,

    indirect_draw_buffer: Option<MultiDrawIndirectBuffer<QuadInstance>>,
}

impl WorldRenderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface_config: &SurfaceConfiguration,
        world: World,
    ) -> Self {
        let camera_controller: CameraController = CameraController::new(
            glam::Vec3::NEG_X,
            -0.5,
            0.0,
            1.6,
            surface_config.width as f32 / surface_config.height as f32,
            0.1,
            1000.0,
            10.0,
            0.002,
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

        let shader_vert = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("world shader"),
            source: ShaderSource::Glsl {
                shader: include_str!("renderer/tv.glsl").into(),
                stage: wgpu::naga::ShaderStage::Vertex,
                defines: Default::default(),
            },
        });

        let shader_frag = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("world shader"),
            source: ShaderSource::Glsl {
                shader: include_str!("renderer/tf.glsl").into(),
                stage: wgpu::naga::ShaderStage::Fragment,
                defines: Default::default(),
            },
        });

        let water_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("world water shader"),
            source: ShaderSource::Wgsl(include_str!("renderer/water.wgsl").into()),
        });

        let (vertex_bind_group_layout, vertex_bind_group) = vertex_buffer::get_bind_group(&device);

        let (texture_bind_group_layout, texture_bind_group) =
            texture::load_textures(&device, &queue).unwrap();

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("world render pipeline layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &camera_bind_group_layout,
                &vertex_bind_group_layout,
                &chunk_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("world render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader_vert,
                entry_point: Some("main"),
                buffers: &[QuadInstance::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader_frag,
                entry_point: Some("main"),
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

        let water_render_pipeline: RenderPipeline =
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("world water render pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: VertexState {
                    module: &water_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[TransparentQuadInstance::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &water_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: surface_config.format,
                        blend: Some(BlendState::ALPHA_BLENDING),
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
            camera_controller,
            vertex_bind_group,
            camera_uniform,
            camera_bind_group,
            chunk_bind_group_layout,
            texture_bind_group,
            render_pipeline,
            water_render_pipeline,
            reticle_renderer,
            world_loader: WorldLoader::new(world, CHUNK_RENDER_DISTANCE),
            indirect_draw_buffer: None,
        }
    }

    pub fn update(&mut self) {
        self.queue.write_buffer(
            &self.camera_uniform,
            0,
            bytemuck::cast_slice(&[self.camera_controller.get_view_projection_matrix()]),
        );

        self.world_loader.update(&self.camera_controller);
        self.world_loader.create_buffers(
            &self.camera_controller,
            &self.device,
            &self.chunk_bind_group_layout,
        );
        if let Some(meshes) = self.world_loader.chunk_meshes.get(&(0, 0)) {
            if self.indirect_draw_buffer.is_some() {
                return;
            }
            println!("Create buffer");
            self.indirect_draw_buffer = Some(MultiDrawIndirectBuffer::new(
                &self.device,
                &self.queue,
                "vertex",
                vec![
                    meshes.get(0).unwrap().quads.as_slice(),
                    meshes.get(1).unwrap().quads.as_slice(),
                ],
                QuadInstance::desc().array_stride,
            ));
        }
    }

    pub fn render<'a: 'b, 'b>(&'a self, render_pass: &mut RenderPass<'b>) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(2, &self.vertex_bind_group, &[]);

        if self.indirect_draw_buffer.is_none() {
            return;
        }

        if let Some(ChunkBuffers {
            instance_buffer: Some(buffer),
            chunk_bind_group,
            quad_instance_count,
            ..
        }) = self.world_loader.get_buffer((0, 0, 0))
        {
            render_pass.set_bind_group(3, &*chunk_bind_group, &[]);
            // render_pass.set_vertex_buffer(0, buffer.slice(..));
            render_pass.set_vertex_buffer(
                0,
                self.indirect_draw_buffer
                    .as_ref()
                    .unwrap()
                    .vertex_buffer
                    .slice(..),
            );

            render_pass.multi_draw_indirect(
                &self.indirect_draw_buffer.as_ref().unwrap().indirect_buffer,
                0,
                2,
            );

            // render_pass.draw(0..QUAD_VERTEX_COUNT, 0..*quad_instance_count);
        }

        // render_pass.set_pipeline(&self.water_render_pipeline);

        // for uvw in self
        //     .world_loader
        //     .visible_chunk_range_uvw(&self.camera_controller)
        // {
        //     if let Some(ChunkBuffers {
        //         transparent_instance_buffer: Some(buffer),
        //         chunk_bind_group,
        //         transparent_quad_instance_count,
        //         ..
        //     }) = self.world_loader.get_buffer(uvw)
        //     {
        //         render_pass.set_bind_group(3, &chunk_bind_group, &[]);
        //         render_pass.set_vertex_buffer(0, buffer.slice(..));

        //         render_pass.draw(0..QUAD_VERTEX_COUNT, 0..*transparent_quad_instance_count);
        //     }
        // }

        self.reticle_renderer
            .render(render_pass, &self.camera_bind_group);
    }
}
