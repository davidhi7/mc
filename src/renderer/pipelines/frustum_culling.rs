use wgpu::{
    include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoder, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineLayoutDescriptor, Queue, ShaderStages,
};

use crate::world::camera::{CameraController, CameraFrustum};

struct CullingDataBinding {
    layout: BindGroupLayout,
    binding: BindGroup,
}

impl CullingDataBinding {
    fn new(
        device: &Device,
        frustum_buffer: &Buffer,
        bounds_buffer: &Buffer,
        uniform_buffer: &Buffer,
        draw_buffer: &Buffer,
    ) -> Self {
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("culling data layout"),
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
        let binding = device.create_bind_group(&BindGroupDescriptor {
            label: Some("culling data binding"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: frustum_buffer.as_entire_binding(),
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
        Self { layout, binding }
    }
}

pub struct FrustumCullingComputePass {
    culling_data_binding: CullingDataBinding,
    frustum_buffer: Buffer,
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
        let frustum_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("culling frustum buffer"),
            size: std::mem::size_of::<CameraFrustum>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bounds_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("culling bounds buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(&[uniform_count, draw_count]),
        });

        let culling_data_binding = CullingDataBinding::new(
            device,
            &frustum_buffer,
            &bounds_buffer,
            uniform_buffer,
            indirect_draw_buffer,
        );

        let shader =
            device.create_shader_module(include_wgsl!("../../../res/shaders/frustum-culling.wgsl"));

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
            frustum_buffer,
            visibility_check_pipeline,
            visibility_writeback_pipeline,
            uniform_count,
            draw_count,
        }
    }

    pub fn run(&self, queue: &Queue, encoder: &mut CommandEncoder, camera: &CameraController) {
        queue.write_buffer(
            &self.frustum_buffer,
            0,
            bytemuck::bytes_of(&CameraFrustum::from_camera(camera)),
        );

        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("culling compute pass"),
            timestamp_writes: None,
        });

        cpass.set_bind_group(0, &self.culling_data_binding.binding, &[]);

        cpass.set_pipeline(&self.visibility_check_pipeline);
        cpass.dispatch_workgroups(self.uniform_count.div_ceil(64), 1, 1);

        cpass.set_pipeline(&self.visibility_writeback_pipeline);
        cpass.dispatch_workgroups(self.draw_count.div_ceil(64), 1, 1);
    }
}
