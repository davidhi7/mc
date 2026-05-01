use std::collections::HashMap;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoder, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineLayoutDescriptor, Queue, ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::{
    camera::{CameraPlanes, ToPlanes, View},
    renderer::pipelines::shadow_mapping::NUM_CASCADES,
    shaders,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CullingPass {
    ShadowMapping { cascade: usize },
    MainPass,
}

impl CullingPass {
    fn all() -> Vec<CullingPass> {
        let mut values = Vec::with_capacity(NUM_CASCADES + 1);
        for i in 0..NUM_CASCADES {
            values.push(CullingPass::ShadowMapping { cascade: i });
        }
        values.push(CullingPass::MainPass);

        values
    }
}

struct CullingDataBinding {
    layout: BindGroupLayout,
    bindings: HashMap<CullingPass, BindGroup>,
}

impl CullingDataBinding {
    fn new(
        device: &Device,
        uniform_buffer: &Buffer,
        draw_buffer: &Buffer,
        uniform_count: u32,
        draw_count: u32,
        frustum_buffers: &HashMap<CullingPass, Buffer>,
    ) -> Self {
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("frustum culling data layout"),
            entries: &[
                // Frustum planes buffer
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniform and indirect draw count buffer
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniform buffer
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Indirect draw buffer
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bounds_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("culling bounds buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(&[uniform_count, draw_count]),
        });

        let mut bindings = HashMap::new();
        for pass in CullingPass::all() {
            let binding = device.create_bind_group(&BindGroupDescriptor {
                label: Some("frustum culling data binding"),
                layout: &layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: frustum_buffers[&pass].as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: bounds_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: draw_buffer.as_entire_binding(),
                    },
                ],
            });
            bindings.insert(pass, binding);
        }

        Self { layout, bindings }
    }
}

pub struct FrustumCullingComputePass {
    frustum_buffers: HashMap<CullingPass, Buffer>,
    culling_data_binding: CullingDataBinding,
    visibility_check_pipeline: ComputePipeline,
    visibility_writeback_pipeline: ComputePipeline,
    uniform_count: u32,
    draw_count: u32,
}

impl FrustumCullingComputePass {
    pub fn new(
        device: &Device,
        uniform_buffer: &Buffer,
        indirect_draw_buffer: &Buffer,
        uniform_count: u32,
        draw_count: u32,
    ) -> FrustumCullingComputePass {
        let mut frustum_buffers = HashMap::new();
        for pass in CullingPass::all() {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("frustum culling frustum buffer {pass:?}")),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                size: std::mem::size_of::<CameraPlanes>() as u64,
                mapped_at_creation: false,
            });
            frustum_buffers.insert(pass, buffer);
        }

        let culling_data_binding = CullingDataBinding::new(
            device,
            uniform_buffer,
            indirect_draw_buffer,
            uniform_count,
            draw_count,
            &frustum_buffers,
        );

        let shader = device.create_shader_module(shaders::SHADER_FRUSTUM_CULLING);

        let visibility_check_pipeline =
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("chunk visibility check pipeline"),
                layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("chunk visibility check pipeline layout"),
                    bind_group_layouts: &[&culling_data_binding.layout],
                    push_constant_ranges: &[],
                })),
                module: &shader,
                entry_point: Some("compute_chunk_visibility"),
                compilation_options: Default::default(),
                cache: None,
            });

        let visibility_writeback_pipeline =
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("chunk visibility writeback pipeline"),
                layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("chunk visibility writeback pipeline layout"),
                    bind_group_layouts: &[&culling_data_binding.layout],
                    push_constant_ranges: &[],
                })),
                module: &shader,
                entry_point: Some("write_chunk_data"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            culling_data_binding,
            visibility_check_pipeline,
            visibility_writeback_pipeline,
            uniform_count,
            draw_count,
            frustum_buffers,
        }
    }

    pub fn write_planes(
        &self,
        queue: &Queue,
        pass: CullingPass,
        view: View,
        projection: impl ToPlanes,
    ) {
        queue.write_buffer(
            &self.frustum_buffers[&pass],
            0,
            bytemuck::bytes_of(&projection.planes(view)),
        );
    }

    pub fn run(&self, encoder: &mut CommandEncoder, pass: CullingPass) {
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("culling compute pass"),
            timestamp_writes: None,
        });
        cpass.set_bind_group(0, &self.culling_data_binding.bindings[&pass], &[]);

        cpass.set_pipeline(&self.visibility_check_pipeline);
        cpass.dispatch_workgroups(self.uniform_count.div_ceil(64), 1, 1);

        cpass.set_pipeline(&self.visibility_writeback_pipeline);
        cpass.dispatch_workgroups(self.draw_count.div_ceil(64), 1, 1);
    }
}
