use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    Buffer, Color, ColorTargetState, ColorWrites, CommandEncoder, CompareFunction, DepthBiasState,
    DepthStencilState, Device, Extent3d, Face, FilterMode, FragmentState, FrontFace, LoadOp,
    MultisampleState, Operations, PolygonMode, PrimitiveState, PrimitiveTopology,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, SamplerBindingType, SamplerBorderColor,
    SamplerDescriptor, ShaderStages, StencilState, StoreOp, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
    TextureViewDimension, VertexState,
};

use crate::{
    renderer::{
        pipelines::{GlobalsBinding, terrain::TerrainBinding},
        vertex_buffer::QuadInstance,
    },
    shaders,
};

const SHADOW_MAP_WIDTH: u32 = 1024 * 4;
const SHADOW_MAP_HEIGHT: u32 = 1024 * 4;

pub struct ShadowMapBinding {
    pub layout: BindGroupLayout,
    pub binding: BindGroup,
}

impl ShadowMapBinding {
    fn new(device: &Device, shadow_map_view: &TextureView) -> Self {
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("shadow map binding layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let binding = device.create_bind_group(&BindGroupDescriptor {
            label: Some("shadow map binding"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(shadow_map_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&device.create_sampler(
                        &SamplerDescriptor {
                            label: Some("shadow map sampler"),
                            // todo bets repeast value
                            address_mode_u: AddressMode::ClampToBorder,
                            address_mode_v: AddressMode::ClampToBorder,
                            address_mode_w: AddressMode::ClampToBorder,
                            border_color: Some(SamplerBorderColor::OpaqueWhite),
                            mag_filter: FilterMode::Linear,
                            min_filter: FilterMode::Linear,
                            mipmap_filter: FilterMode::Nearest,
                            // TODO ?
                            compare: Some(CompareFunction::LessEqual),
                            ..Default::default()
                        },
                    )),
                },
            ],
        });

        ShadowMapBinding { layout, binding }
    }
}

pub struct ShadowMappingPipeline {
    pub binding: ShadowMapBinding,
    pipeline: RenderPipeline,
    // todo not pub
    pub shadow_map_view: TextureView,
    pub render_target_view: TextureView,
}

impl ShadowMappingPipeline {
    pub fn new(
        device: &Device,
        globals_binding: &GlobalsBinding,
        terrain_binding: &TerrainBinding,
    ) -> Self {
        let shadow_map = device.create_texture(&TextureDescriptor {
            label: Some("shadow map texture"),
            size: Extent3d {
                width: SHADOW_MAP_WIDTH,
                height: SHADOW_MAP_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_map_view = shadow_map.create_view(&TextureViewDescriptor::default());
        let render_target = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: SHADOW_MAP_WIDTH,
                height: SHADOW_MAP_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let render_target_view = render_target.create_view(&TextureViewDescriptor::default());

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("shadow mapping pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("shadow mapping pipeline layout"),
                    bind_group_layouts: &[
                        &globals_binding.layout,
                        &terrain_binding.buffers.layout,
                        &terrain_binding.textures.layout,
                    ],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: VertexState {
                module: &device.create_shader_module(shaders::SHADER_SHADOW_MAPPING),
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[QuadInstance::desc()],
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
                module: &device.create_shader_module(shaders::SHADER_SHADOW_MAPPING),
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        Self {
            binding: ShadowMapBinding::new(device, &shadow_map_view),
            pipeline,
            shadow_map_view,
            render_target_view,
        }
    }

    pub fn render(
        &self,
        encoder: &mut CommandEncoder,
        globals_binding: &GlobalsBinding,
        terrain_binding: &TerrainBinding,
        vertex_buffer: &Buffer,
        indirect_buffer: &Buffer,
        indirect_offset: u64,
        indirect_count: u32,
    ) {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("shadow mapping render pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &self.render_target_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &self.shadow_map_view,
                depth_ops: Some(Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_bind_group(0, Some(&globals_binding.binding), &[]);
        render_pass.set_bind_group(1, Some(&terrain_binding.buffers.binding), &[]);
        render_pass.set_bind_group(2, Some(&terrain_binding.textures.binding), &[]);
        render_pass.multi_draw_indirect(indirect_buffer, indirect_offset, indirect_count);
    }
}
