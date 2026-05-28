use std::mem::size_of;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoder, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::{
    camera::{CameraPlanes, ToPlanes, View},
    renderer::{
        indirect_buffer_array::{IndirectBufferArray, IndirectBufferBinding},
        pipelines::shadow_mapping::NUM_CASCADES,
    },
    shaders,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CullingPass {
    // todo evaluate type
    ShadowMapping { cascade: usize },
    MainPass,
}

impl CullingPass {
    pub fn shadow(cascade: usize) -> Self {
        if cascade >= NUM_CASCADES {
            panic!("Invalid cascade {cascade}")
        }
        CullingPass::ShadowMapping { cascade }
    }

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
    binding: BindGroup,
}

impl CullingDataBinding {
    fn new(
        device: &Device,
        frustum_buffer: &Buffer,
        descriptor_count_buffer: &Buffer,
        chunk_descriptor_buffer: &Buffer,
        chunk_uniform_buffer: &Buffer,
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
                // Chunk descriptor count buffer
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
                // Chunk descriptor buffer
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Chunk uniform buffer
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
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
                    resource: descriptor_count_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: chunk_descriptor_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: chunk_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        Self { layout, binding }
    }
}

pub struct CullingComputePass {
    frustum_buffer: Buffer,
    descriptor_count_buffer: Buffer,
    culling_data_binding: CullingDataBinding,
    indirect_buffer_binding: IndirectBufferBinding,
    pipeline: ComputePipeline,
}

impl CullingComputePass {
    pub fn new(
        device: &Device,
        chunk_descriptor_buffer: &Buffer,
        chunk_uniform_buffer: &Buffer,
        indirect_buffer_array: &IndirectBufferArray,
    ) -> CullingComputePass {
        let perspectives = CullingPass::all().len() as u64;
        let frustum_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("frustum culling frustum buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            size: perspectives * size_of::<CameraPlanes>() as u64,
            mapped_at_creation: false,
        });

        let descriptor_count_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("descriptor count buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: &0u32.to_ne_bytes(),
        });

        let culling_data_binding = CullingDataBinding::new(
            device,
            &frustum_buffer,
            &descriptor_count_buffer,
            chunk_descriptor_buffer,
            chunk_uniform_buffer,
        );

        let indirect_buffer_binding = IndirectBufferBinding::new(device, indirect_buffer_array);

        let shader = device.create_shader_module(shaders::SHADER_FRUSTUM_CULLING);

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("culling pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("culling pipeline layout"),
                bind_group_layouts: &[
                    &culling_data_binding.layout,
                    &indirect_buffer_binding.layout,
                ],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: Some("run"),
            compilation_options: PipelineCompilationOptions {
                constants: &[
                    // todo some are not used
                    ("PASS_COUNT", indirect_buffer_array.pass_count() as f64),
                    ("BUCKET_COUNT", indirect_buffer_array.bucket_count() as f64),
                    (
                        "INDIRECT_BUFFER_SLOTS",
                        indirect_buffer_array.indirect_buffer_slots() as f64,
                    ),
                ],
                ..Default::default()
            },
            cache: None,
        });

        Self {
            frustum_buffer,
            descriptor_count_buffer,
            culling_data_binding,
            indirect_buffer_binding,
            pipeline,
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
            &self.frustum_buffer,
            pass.offset().0 * size_of::<CameraPlanes>() as u64,
            bytemuck::bytes_of(&projection.planes(view)),
        );
    }

    pub fn run(
        &self,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        indirect_buffer_array: &IndirectBufferArray,
        descriptor_count: u32,
    ) {
        // TODO is this already done during last iteration's readback?
        // TODO write count of descriptors to buffer
        indirect_buffer_array.clear_counts(encoder);
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("culling compute pass"),
            timestamp_writes: None,
        });
        println!("{descriptor_count}");
        queue.write_buffer(
            &self.descriptor_count_buffer,
            0,
            &descriptor_count.to_ne_bytes(),
        );

        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.culling_data_binding.binding, &[]);
        cpass.set_bind_group(1, &self.indirect_buffer_binding.binding, &[]);
        cpass.dispatch_workgroups(descriptor_count.div_ceil(64).into(), 1, 1);
    }
}
