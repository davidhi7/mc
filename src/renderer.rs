use std::{collections::HashMap, sync::Arc};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindingType,
    BlendState, Buffer, BufferBindingType, BufferUsages, ColorTargetState, ColorWrites,
    CompareFunction, DepthBiasState, DepthStencilState, Device, Face, FragmentState, FrontFace,
    MultisampleState, PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode,
    PrimitiveState, PrimitiveTopology, Queue, RenderPass, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState, SurfaceConfiguration,
    TextureFormat, VertexState,
};

use crate::{
    renderer::{
        indirect_buffer_manager::MultiDrawIndirectBuffer,
        ui_renderer::Reticle,
        vertex_buffer::{QuadInstance, TransparentQuadInstance},
    },
    texture,
    world::{
        camera::CameraController,
        chunk::VERTICAL_CHUNK_COUNT,
        world_loader::{ChunkUniform, TerrainBuckets, WorldLoader},
        World,
    },
};

pub mod buffers;
pub mod indirect_buffer_manager;
mod ui_renderer;

pub mod vertex_buffer;

const CHUNK_RENDER_DISTANCE: u32 = 16;

pub struct WorldRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pub camera_controller: CameraController,
    vertex_bind_group: BindGroup,
    camera_uniform: Buffer,
    camera_bind_group: BindGroup,
    texture_bind_group: BindGroup,
    render_pipeline: RenderPipeline,
    water_render_pipeline: RenderPipeline,
    reticle_renderer: ui_renderer::Reticle,
    world_loader: WorldLoader,
    indirect_draw_buffer: MultiDrawIndirectBuffer<ChunkUniform, TerrainBuckets, 2>,
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
            100.0,
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

        let terrain_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("world terrain shader"),
            source: ShaderSource::Wgsl(include_str!("renderer/terrain.wgsl").into()),
        });

        let water_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("world water shader"),
            source: ShaderSource::Wgsl(include_str!("renderer/water.wgsl").into()),
        });

        let (vertex_bind_group_layout, vertex_bind_group) = vertex_buffer::get_bind_group(&device);

        let (texture_bind_group_layout, texture_bind_group) =
            texture::load_textures(&device, &queue).unwrap();

        let mut world_loader =
            WorldLoader::new(world, 8, Arc::clone(&device), CHUNK_RENDER_DISTANCE);

        // TODO find better values
        let mut batches_map = HashMap::new();
        batches_map.insert(TerrainBuckets::SOLID, 3000);
        batches_map.insert(TerrainBuckets::TRANSPARENT, 1000);
        let mut ib = MultiDrawIndirectBuffer::new(
            &device,
            "",
            [TerrainBuckets::SOLID, TerrainBuckets::TRANSPARENT],
            (2 * CHUNK_RENDER_DISTANCE as u64 + 1).pow(2)
                * u64::min(
                    CHUNK_RENDER_DISTANCE as u64 * 2 + 1,
                    VERTICAL_CHUNK_COUNT as u64,
                ),
            &batches_map,
        );

        world_loader.load_chunks(&device, &queue, &mut ib, &camera_controller);

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("world render pipeline layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &camera_bind_group_layout,
                &vertex_bind_group_layout,
                &ib.uniform_layout.layout,
            ],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("world render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &terrain_shader,
                entry_point: Some("vs_main"),
                buffers: &[QuadInstance::desc()],
                compilation_options: PipelineCompilationOptions {
                    constants: &HashMap::new(),
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(FragmentState {
                module: &terrain_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions {
                    constants: &HashMap::new(),
                    zero_initialize_workgroup_memory: false,
                },
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
                    compilation_options: PipelineCompilationOptions {
                        constants: &HashMap::new(),
                        zero_initialize_workgroup_memory: false,
                    },
                },
                fragment: Some(FragmentState {
                    module: &water_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: surface_config.format,
                        blend: Some(BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                    compilation_options: PipelineCompilationOptions {
                        constants: &HashMap::new(),
                        zero_initialize_workgroup_memory: false,
                    },
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
            texture_bind_group,
            render_pipeline,
            water_render_pipeline,
            reticle_renderer,
            world_loader,
            indirect_draw_buffer: ib,
        }
    }

    pub fn update(&mut self) {
        self.queue.write_buffer(
            &self.camera_uniform,
            0,
            bytemuck::cast_slice(&[self.camera_controller.get_view_projection_matrix()]),
        );

        self.world_loader.load_chunks(
            &self.device,
            &self.queue,
            &mut self.indirect_draw_buffer,
            &self.camera_controller,
        );
    }

    pub fn render<'a: 'b, 'b>(&'a self, render_pass: &mut RenderPass<'b>) {
        render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(2, &self.vertex_bind_group, &[]);
        render_pass.set_bind_group(3, &self.indirect_draw_buffer.uniform_layout.binding, &[]);
        render_pass.set_vertex_buffer(0, self.indirect_draw_buffer.vertex_buffer.slice(..));

        if self.indirect_draw_buffer.draw_count(TerrainBuckets::SOLID) > 0 {
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.multi_draw_indirect(
                &self.indirect_draw_buffer.indirect_buffer,
                self.indirect_draw_buffer
                    .indirect_buffer_offset_bytes(TerrainBuckets::SOLID, 0),
                self.indirect_draw_buffer.draw_count(TerrainBuckets::SOLID) as u32,
            );
        }

        if self
            .indirect_draw_buffer
            .draw_count(TerrainBuckets::TRANSPARENT)
            > 0
        {
            render_pass.set_pipeline(&self.water_render_pipeline);
            render_pass.multi_draw_indirect(
                &self.indirect_draw_buffer.indirect_buffer,
                self.indirect_draw_buffer
                    .indirect_buffer_offset_bytes(TerrainBuckets::TRANSPARENT, 0),
                self.indirect_draw_buffer
                    .draw_count(TerrainBuckets::TRANSPARENT) as u32,
            );
        }

        self.reticle_renderer
            .render(render_pass, &self.camera_bind_group);
    }
}
