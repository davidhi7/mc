use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendState, Buffer, BufferBindingType,
    ColorTargetState, ColorWrites, CompareFunction, DepthBiasState, DepthStencilState, Device,
    Face, FragmentState, FrontFace, MultisampleState, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, RenderPass,
    RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerBindingType, ShaderStages,
    StencilState, TextureFormat, TextureSampleType, TextureView, TextureViewDimension, VertexState,
};

use crate::{
    renderer::{
        pipelines::{
            GlobalsBinding,
            shadow_mapping::{NUM_CASCADES, ShadowMapBinding},
        },
        vertex_buffer::{QuadInstance, TransparentQuadInstance},
    },
    shaders,
};

pub struct TerrainTexturesBinding {
    pub layout: BindGroupLayout,
    pub binding: BindGroup,
}

pub struct TerrainBuffersBinding {
    pub layout: BindGroupLayout,
    pub binding: BindGroup,
}

pub struct TerrainBinding {
    pub buffers: TerrainBuffersBinding,
    pub textures: TerrainTexturesBinding,
}

impl TerrainBinding {
    pub fn new(
        device: &Device,
        vertex_buffer: &Buffer,
        chunk_buffer: &Buffer,
        texture_array: TextureView,
        sampler: &Sampler,
    ) -> Self {
        let layout_buffers = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("terrain pipeline textures layout"),
            entries: &[
                // Constant vertex buffer
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Chunk coordinate buffer
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout_textures = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("terrain pipeline buffers layout"),
            entries: &[
                // Textures
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2Array,
                        sample_type: TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Textures sampler
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let binding_buffers = device.create_bind_group(&BindGroupDescriptor {
            label: Some("terrain pipeline buffers binding"),
            layout: &layout_buffers,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: chunk_buffer.as_entire_binding(),
                },
            ],
        });

        let binding_textures = device.create_bind_group(&BindGroupDescriptor {
            label: Some("terrain pipeline textures binding"),
            layout: &layout_textures,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&texture_array),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(sampler),
                },
            ],
        });

        Self {
            buffers: TerrainBuffersBinding {
                layout: layout_buffers,
                binding: binding_buffers,
            },
            textures: TerrainTexturesBinding {
                layout: layout_textures,
                binding: binding_textures,
            },
        }
    }
}

pub struct TerrainPipeline {
    pub binding: TerrainBinding,
    terrain_pipeline: RenderPipeline,
    water_pipeline: RenderPipeline,
}

impl TerrainPipeline {
    pub fn new(
        device: &Device,
        globals_binding: &GlobalsBinding,
        binding: TerrainBinding,
        surface_format: TextureFormat,
        shadow_map_binding: &ShadowMapBinding,
    ) -> Self {
        let terrain_shader = device.create_shader_module(shaders::SHADER_TERRAIN);

        let water_shader = device.create_shader_module(shaders::SHADER_WATER);

        let terrain_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("terrain render pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("terrain render pipeline layout"),
                bind_group_layouts: &[
                    &globals_binding.layout,
                    &binding.buffers.layout,
                    &binding.textures.layout,
                    &shadow_map_binding.layout,
                ],
                push_constant_ranges: &[],
            })),
            vertex: VertexState {
                module: &terrain_shader,
                entry_point: Some("vs_main"),
                buffers: &[QuadInstance::desc()],
                compilation_options: Default::default(),
            },
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
            fragment: Some(FragmentState {
                module: &terrain_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        let water_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("terrain water render pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("terrain water render pipeline layout"),
                bind_group_layouts: &[
                    &globals_binding.layout,
                    &binding.buffers.layout,
                    &binding.textures.layout,
                ],
                push_constant_ranges: &[],
            })),
            vertex: VertexState {
                module: &water_shader,
                entry_point: Some("vs_main"),
                buffers: &[TransparentQuadInstance::desc()],
                compilation_options: Default::default(),
            },
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
            fragment: Some(FragmentState {
                module: &water_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        Self {
            binding,
            terrain_pipeline,
            water_pipeline,
        }
    }

    pub fn render_terrain(
        &self,
        render_pass: &mut RenderPass,
        globals: &GlobalsBinding,
        shadow_map_binding: &ShadowMapBinding,
        vertex_buffer: &Buffer,
        indirect_buffer: &Buffer,
        indirect_offset: u64,
        indirect_count: u32,
    ) {
        render_pass.set_pipeline(&self.terrain_pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_bind_group(0, &globals.binding, &[]);
        render_pass.set_bind_group(1, Some(&self.binding.buffers.binding), &[]);
        render_pass.set_bind_group(2, Some(&self.binding.textures.binding), &[]);
        render_pass.set_bind_group(3, Some(&shadow_map_binding.binding), &[]);
        render_pass.multi_draw_indirect(indirect_buffer, indirect_offset, indirect_count);
    }

    pub fn render_water(
        &self,
        render_pass: &mut RenderPass,
        globals: &GlobalsBinding,
        vertex_buffer: &Buffer,
        indirect_buffer: &Buffer,
        indirect_offset: u64,
        indirect_count: u32,
    ) {
        render_pass.set_pipeline(&self.water_pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_bind_group(0, &globals.binding, &[]);
        render_pass.set_bind_group(1, Some(&self.binding.buffers.binding), &[]);
        render_pass.set_bind_group(2, Some(&self.binding.textures.binding), &[]);
        render_pass.multi_draw_indirect(indirect_buffer, indirect_offset, indirect_count);
    }
}
