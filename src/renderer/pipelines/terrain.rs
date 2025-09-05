use std::{collections::HashMap, num::NonZeroU32};

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendState, Buffer, BufferBindingType,
    ColorTargetState, ColorWrites, CompareFunction, DepthBiasState, DepthStencilState, Device,
    Face, FragmentState, FrontFace, MultisampleState, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, RenderPass,
    RenderPipeline, RenderPipelineDescriptor, Sampler, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StencilState, TextureFormat, TextureView, VertexState,
};

use crate::renderer::{
    pipelines::GlobalsBinding,
    vertex_buffer::{QuadInstance, TransparentQuadInstance},
};

struct TerrainBinding {
    layout: BindGroupLayout,
    binding: BindGroup,
}

impl TerrainBinding {
    fn new(
        device: &Device,
        vertex_buffer: &Buffer,
        chunk_buffer: &Buffer,
        textures: Vec<TextureView>,
        sampler: &Sampler,
    ) -> Self {
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("terrain pipeline data layout"),
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
                // Textures
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: NonZeroU32::new(textures.len() as u32),
                },
                // Textures sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let binding = device.create_bind_group(&BindGroupDescriptor {
            label: Some("terrain pipeline data binding"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: chunk_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureViewArray(
                        &(textures.iter().collect::<Vec<_>>()),
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self { layout, binding }
    }
}

pub struct TerrainPipeline {
    binding: TerrainBinding,
    terrain_pipeline: RenderPipeline,
    water_pipeline: RenderPipeline,
}

impl TerrainPipeline {
    pub fn new(
        device: &Device,
        globals_binding: &GlobalsBinding,
        vertex_buffer: &Buffer,
        chunk_buffer: &Buffer,
        textures: Vec<TextureView>,
        sampler: &Sampler,
        surface_format: TextureFormat,
    ) -> Self {
        let binding = TerrainBinding::new(device, vertex_buffer, chunk_buffer, textures, sampler);

        let terrain_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("terrain solid shader"),
            source: ShaderSource::Wgsl(include_str!("../../../res/shaders/terrain.wgsl").into()),
        });

        let water_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("terrain water shader"),
            source: ShaderSource::Wgsl(include_str!("../../../res/shaders/water.wgsl").into()),
        });

        let terrain_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("terrain render pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("terrain render pipeline layout"),
                bind_group_layouts: &[&globals_binding.layout, &binding.layout],
                push_constant_ranges: &[],
            })),
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
                    format: surface_format,
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

        let water_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("terrain water render pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("terrain water render pipeline layout"),
                bind_group_layouts: &[&globals_binding.layout, &binding.layout],
                push_constant_ranges: &[],
            })),
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
                    format: surface_format,
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
        vertex_buffer: &Buffer,
        indirect_buffer: &Buffer,
        indirect_offset: u64,
        indirect_count: u32,
    ) {
        render_pass.set_pipeline(&self.terrain_pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_bind_group(0, &globals.binding, &[]);
        render_pass.set_bind_group(1, Some(&self.binding.binding), &[]);
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
        render_pass.set_bind_group(1, Some(&self.binding.binding), &[]);
        render_pass.multi_draw_indirect(indirect_buffer, indirect_offset, indirect_count);
    }
}
