use std::array;

use bytemuck::{Pod, Zeroable};
use egui::Slider;
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBindingType, BufferUsages, CommandEncoder, CompareFunction, DepthBiasState,
    DepthStencilState, Device, Extent3d, Face, FilterMode, FragmentState, FrontFace, LoadOp,
    MultisampleState, Operations, PolygonMode, PrimitiveState, PrimitiveTopology, Queue,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor, ShaderStages, StencilState,
    StoreOp, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexState,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::{
    camera::{OrthographicProj, PerspectiveProj, ToMatrix, View},
    renderer::{
        buffers::AsBytes,
        pipelines::{GlobalsBinding, terrain::TerrainBinding},
        vertex_buffer::QuadInstance,
    },
    shaders,
    ui::GuiModule,
    world::chunk::CHUNK_WIDTH,
};
use glam::Vec3;

// Implementation of casacaded shadow maps, based on https://developer.download.nvidia.com/SDK/10.5/opengl/src/cascaded_shadow_maps/doc/cascaded_shadow_maps.pdf
pub const NUM_CASCADES: usize = 4;
// Interpolation factor between linear frustum slice depth (if 0.0) or logarithmic depth (if 1.0)
const LAMBDA: f32 = 0.75;
const SHADOW_MAP_WIDTH: u32 = 2048;
const SHADOW_MAP_HEIGHT: u32 = SHADOW_MAP_WIDTH;

const MAX_SHADOW_DISTANCE: f32 = (4 * CHUNK_WIDTH) as f32;

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ShadowCascadeUniform {
    index: u32,
    _padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ShadowSettingsRaw {
    bias_min: f32,
    bias_max: f32,
    // We use u32 because bool is not Pod
    // All values != 0 represent true
    pcf: u32,
}

#[derive(Clone, Copy, PartialEq)]
struct ShadowSettings {
    bias_min: f32,
    bias_max: f32,
    pcf: bool,
}

impl From<ShadowSettings> for ShadowSettingsRaw {
    fn from(value: ShadowSettings) -> Self {
        let ShadowSettings {
            bias_min,
            bias_max,
            pcf,
        } = value;
        Self {
            bias_min,
            bias_max,
            pcf: if pcf { 1 } else { 0 },
        }
    }
}

pub struct ShadowMapBinding {
    pub layout: BindGroupLayout,
    pub binding: BindGroup,
}

impl ShadowMapBinding {
    fn new(device: &Device, shadow_maps: &Texture, bias_buffer: &Buffer) -> Self {
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("shadow map binding layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2Array,
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
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
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
                    resource: BindingResource::TextureView(
                        &shadow_maps.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&device.create_sampler(
                        &SamplerDescriptor {
                            label: Some("shadow map sampler"),
                            // Important so PCF out-of-bounds access are sensible
                            address_mode_u: AddressMode::ClampToEdge,
                            address_mode_v: AddressMode::ClampToEdge,
                            address_mode_w: AddressMode::ClampToEdge,
                            // Nearest produces pixelated shadows that might look good given a sufficient shadow map size and no light projection movement
                            // Linear adds PCF so smoother and less obvious flickering
                            mag_filter: FilterMode::Linear,
                            min_filter: FilterMode::Linear,
                            mipmap_filter: FilterMode::Nearest,
                            // Tests if fragment's depth is less than or equal to depth of occluder
                            compare: Some(CompareFunction::LessEqual),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(bias_buffer.as_entire_buffer_binding()),
                },
            ],
        });

        ShadowMapBinding { layout, binding }
    }
}

pub struct ShadowMappingPipeline {
    pub binding: ShadowMapBinding,
    queue: Queue,
    pipeline: RenderPipeline,
    shadow_maps: Texture,
    cascade_bind_groups: Vec<BindGroup>,
    settings_buffer: Buffer,
    settings: ShadowSettings,
}

impl ShadowMappingPipeline {
    pub fn new(
        device: &Device,
        queue: &Queue,
        globals_binding: &GlobalsBinding,
        terrain_binding: &TerrainBinding,
    ) -> Self {
        let shadow_maps = device.create_texture(&TextureDescriptor {
            label: Some("shadow mapping depth texture"),
            size: Extent3d {
                width: SHADOW_MAP_WIDTH,
                height: SHADOW_MAP_HEIGHT,
                depth_or_array_layers: NUM_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let cascade_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("shadow cascade layout"),
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

        let mut cascade_bind_groups = Vec::with_capacity(NUM_CASCADES);
        for cascade in 0..NUM_CASCADES {
            let cascade_buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("shadow cascade buffer"),
                contents: (cascade as u32).get_bytes(),
                usage: BufferUsages::UNIFORM,
            });
            let cascade_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("shadow cascade binding"),
                layout: &cascade_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: cascade_buffer.as_entire_binding(),
                }],
            });
            cascade_bind_groups.push(cascade_bind_group);
        }

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("shadow mapping pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("shadow mapping pipeline layout"),
                    bind_group_layouts: &[
                        &globals_binding.layout,
                        &terrain_binding.buffers.layout,
                        &terrain_binding.textures.layout,
                        &cascade_layout,
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
                // Use front face culling so shadows still work even if the front face of a mountain or similar are too far away to be rendered.
                // Also this eliminates shadow acne on fragments outside the shadows, and acne on shadowed sides is irrelevant for us.
                cull_mode: Some(Face::Front),
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
                targets: &[],
            }),
            multiview: None,
            cache: None,
        });

        let settings = ShadowSettings {
            bias_min: 0.00001,
            bias_max: 0.00001,
            pcf: false,
        };
        let settings_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&ShadowSettingsRaw::from(settings)),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        Self {
            binding: ShadowMapBinding::new(device, &shadow_maps, &settings_buffer),
            queue: queue.clone(),
            pipeline,
            shadow_maps,
            cascade_bind_groups,
            settings_buffer,
            settings,
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
        cascade: usize,
    ) {
        if cascade >= NUM_CASCADES {
            panic!("shadow map cascade too large");
        }
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("shadow mapping render pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &self.shadow_maps.create_view(&TextureViewDescriptor {
                    base_array_layer: cascade as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                }),
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
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
        render_pass.set_bind_group(3, Some(&self.cascade_bind_groups[cascade]), &[]);
        render_pass.multi_draw_indirect(indirect_buffer, indirect_offset, indirect_count);
    }
}

impl GuiModule for ShadowMappingPipeline {
    fn title(&self) -> Option<&str> {
        Some("Shadow mapping")
    }

    fn add_contents(&mut self, ui: &mut egui::Ui) {
        ui.label("bias min:");
        ui.add(Slider::new(&mut self.settings.bias_min, -0.0001..=0.0001));
        ui.label("bias max:");
        ui.add(Slider::new(&mut self.settings.bias_max, -0.0001..=0.0001));
        ui.checkbox(&mut self.settings.pcf, "pcf");
        // todo only write if changed
        self.queue.write_buffer(
            &self.settings_buffer,
            0,
            bytemuck::bytes_of(&ShadowSettingsRaw::from(self.settings)),
        );
    }
}

/// Compute the upper z limits of all frustum slices in view space, so within (z_near, z_far].
pub fn compute_frustum_slice_z_far(camera_projection: PerspectiveProj) -> [f32; NUM_CASCADES] {
    array::from_fn(|n| {
        let z_near = camera_projection.z_near;
        let z_far = camera_projection
            .z_far
            .clamp(0.0, z_near + MAX_SHADOW_DISTANCE);

        let cascade_end = (n as f32 + 1.0) / NUM_CASCADES as f32;

        LAMBDA * z_near * (z_far / z_near).powf(cascade_end)
            + (1.0 - LAMBDA) * (z_near + cascade_end * (z_far - z_near))
    })
}

pub fn create_shadow_projections(
    light_view: View,
    camera_view: View,
    camera_projection: PerspectiveProj,
) -> [OrthographicProj; NUM_CASCADES] {
    // The n-th frustum slice is between frustum_slice_bounds[n], frustum_slice_bounds[n + 1] (for n in 0..NUM_CASCADES)
    // All values are in view space
    let mut frustum_slice_bounds = [0.0; NUM_CASCADES + 1];
    frustum_slice_bounds[1..].copy_from_slice(&compute_frustum_slice_z_far(camera_projection));

    array::from_fn(|n| {
        let View {
            eye, direction, up, ..
        } = camera_view;
        let PerspectiveProj {
            fov_y_rad,
            aspect_ratio,
            ..
        } = camera_projection;

        let sub_frustum_near = frustum_slice_bounds[n];
        let sub_frustum_far = frustum_slice_bounds[n + 1];

        let right = up.cross(direction);
        let tan_fov_y_near = f32::tan(fov_y_rad / 2.0) * sub_frustum_near;
        let tan_fov_x_near = tan_fov_y_near * aspect_ratio;

        let tan_fov_y_far = f32::tan(fov_y_rad / 2.0) * sub_frustum_far;
        let tan_fov_x_far = tan_fov_y_far * aspect_ratio;

        let direction_near = eye + direction * sub_frustum_near;
        let direction_far = eye + direction * sub_frustum_far;

        let near_bl = direction_near - tan_fov_y_near * up - tan_fov_x_near * right;
        let near_tl = direction_near + tan_fov_y_near * up - tan_fov_x_near * right;
        let near_br = direction_near - tan_fov_y_near * up + tan_fov_x_near * right;
        let near_tr = direction_near + tan_fov_y_near * up + tan_fov_x_near * right;

        let far_bl = direction_far - tan_fov_y_far * up - tan_fov_x_far * right;
        let far_tl = direction_far + tan_fov_y_far * up - tan_fov_x_far * right;
        let far_br = direction_far - tan_fov_y_far * up + tan_fov_x_far * right;
        let far_tr = direction_far + tan_fov_y_far * up + tan_fov_x_far * right;

        let light_view = light_view.matrix();

        let frustum_corners = [
            near_bl, near_tl, near_br, near_tr, far_bl, far_tl, far_br, far_tr,
        ];

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        for corner in frustum_corners {
            let corner_light_space = light_view.transform_point3(corner);
            min = min.min(corner_light_space);
            max = max.max(corner_light_space);
        }

        OrthographicProj {
            left: min.x,
            right: max.x,
            bottom: min.y,
            top: max.y,
            // TODO sensible values
            near: min.z - 100.0,
            far: max.z,
        }
    })
}
