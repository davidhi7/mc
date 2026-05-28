use std::{mem::size_of, sync::mpsc};

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages, CommandEncoder,
    Device, PollType, Queue, ShaderStages,
    wgt::{BufferDescriptor, DrawIndirectArgs},
};

use crate::renderer::{
    indirect_buffer_manager::TerrainBuckets, pipelines::frustum_culling::CullingPass,
};

#[derive(Debug, Clone, Copy)]
pub struct PassId(pub u64);
#[derive(Debug, Clone, Copy)]
pub struct BucketId(pub u64);

impl CullingPass {
    pub fn offset(self) -> PassId {
        match self {
            CullingPass::ShadowMapping { cascade } => PassId(cascade as u64 + 1),
            CullingPass::MainPass => PassId(0),
        }
    }
}

impl TerrainBuckets {
    pub const fn offset(self) -> BucketId {
        match self {
            TerrainBuckets::Solid => BucketId(0),
            TerrainBuckets::Transparent => BucketId(1),
        }
    }
}

pub enum RenderingStrategy {
    GpuCount,
    CpuCount,
}

pub struct IndirectBufferBinding {
    pub layout: BindGroupLayout,
    pub binding: BindGroup,
}

impl IndirectBufferBinding {
    pub fn new(device: &Device, indirect_buffer_array: &IndirectBufferArray) -> Self {
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("indirect buffer binding layout"),
            entries: &[
                // Indirect draw buffer
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Draw counts buffer
                BindGroupLayoutEntry {
                    binding: 1,
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
            label: Some("indirect buffer binding"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: indirect_buffer_array.indirect_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: indirect_buffer_array.counts_buffer.as_entire_binding(),
                },
            ],
        });

        Self { layout, binding }
    }
}

pub struct IndirectBufferArray {
    bucket_count: u64,
    pass_count: u64,
    indirect_buffer_slots: u64,
    // todo maybe not pub
    pub indirect_buffer: Buffer,
    pub counts_buffer: Buffer,
    counts_readback_buffer: Buffer,
    counts_state: Box<[u32]>,
}

impl IndirectBufferArray {
    pub fn new(
        device: &Device,
        bucket_count: u64,
        pass_count: u64,
        indirect_buffer_slots: u64,
    ) -> Self {
        let counts_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("counts array buffer"),
            size: bucket_count * pass_count * size_of::<u32>() as u64,
            // Atomics cannot be used in uniform buffers
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counts_readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("counts array staging buffer"),
            size: bucket_count * pass_count * size_of::<u32>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let indirect_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("indirect array buffer"),
            size: bucket_count
                * pass_count
                * indirect_buffer_slots
                * size_of::<DrawIndirectArgs>() as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Self {
            bucket_count,
            pass_count,
            indirect_buffer_slots,
            indirect_buffer,
            counts_buffer,
            counts_readback_buffer,
            counts_state: vec![0u32; (bucket_count * pass_count) as usize].into_boxed_slice(),
        }
    }

    pub fn bucket_count(&self) -> u64 {
        self.bucket_count
    }
    pub fn pass_count(&self) -> u64 {
        self.pass_count
    }
    pub fn indirect_buffer_slots(&self) -> u64 {
        self.indirect_buffer_slots
    }

    pub fn indirect_offset(&self, pass: PassId, bucket: BucketId) -> u64 {
        (pass.0 * self.bucket_count + bucket.0)
            * self.indirect_buffer_slots
            * size_of::<DrawIndirectArgs>() as u64
    }

    pub fn counts_offset(&self, pass: PassId, bucket: BucketId) -> u64 {
        (pass.0 * self.bucket_count + bucket.0)
            * self.indirect_buffer_slots
            * size_of::<u32>() as u64
    }

    pub fn count(&self, pass: PassId, bucket: BucketId) -> u32 {
        self.counts_state[(pass.0 * self.bucket_count + bucket.0) as usize]
    }

    pub fn clear_counts(&self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.counts_buffer, 0, None);
    }

    pub fn readback_counts(&mut self, device: &Device, queue: &Queue, mut encoder: CommandEncoder) {
        // TODO: don't panic don't fail if not needed
        let (tx, rx) = mpsc::channel::<Box<[u32]>>();
        let counts_readback_buffer = self.counts_readback_buffer.clone();

        encoder.copy_buffer_to_buffer(
            &self.counts_buffer,
            0,
            &self.counts_readback_buffer,
            0,
            self.counts_buffer.size(),
        );
        encoder.map_buffer_on_submit(
            &self.counts_readback_buffer,
            wgpu::MapMode::Read,
            ..,
            move |result| {
                if result.is_err() {
                    log::error!("Failed to readback counts buffer");
                    panic!();
                }
                let mapped = counts_readback_buffer.get_mapped_range(..);
                tx.send(bytemuck::cast_slice(&mapped).into())
                    .expect("Failed to send buffer contents");
                drop(mapped);
                counts_readback_buffer.unmap();
            },
        );

        let submission_index = queue.submit(std::iter::once(encoder.finish()));
        device
            .poll(PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            })
            .expect("Failed to poll device");

        self.counts_state = rx.recv().expect("Failed to receive buffer contents");
        println!("counts: {:?}", self.counts_state);
    }
}
