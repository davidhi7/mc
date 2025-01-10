use core::panic;
use std::{collections::BTreeMap, marker::PhantomData};

use std::fmt::Debug;

use wgpu::util::DrawIndirectArgs;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoder, Device, Queue, ShaderStages,
};

use crate::renderer::buffers::block_allocator::{BlockAllocator, RcBlockAllocator, RcBlockHandle};
use crate::renderer::buffers::pool_allocator::{PoolAllocator, SegmentHandle};
use crate::renderer::buffers::{AsBytes, BufferMemoryTarget};
use crate::renderer::vertex_buffer::QUAD_VERTEX_COUNT;

pub struct UniformBinding {
    pub layout: BindGroupLayout,
    pub binding: BindGroup,
}

impl UniformBinding {
    fn new(device: &Device, buffer: &Buffer) -> Self {
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("uniform bind group layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let binding = device.create_bind_group(&BindGroupDescriptor {
            label: Some("uniform bind group layout"),
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        UniformBinding { layout, binding }
    }
}

/// Trait representing values that act as a bucket identifier for a class of draw calls.
pub trait DrawCallBucket {
    /// Size of a single instance, in bytes
    fn instance_size(&self) -> u64;
}

/// Identifier for a single draw call, that is, one uniform and one bucket value.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DrawCallHandle<Uniform: Clone + Ord, Bucket: Clone + Copy> {
    pub uniform: Uniform,
    pub bucket: Bucket,
}

/// Struct of all data associated with a single draw call.
#[derive(Debug)]
pub struct DrawCallData {
    indirect_buffer_handle: u64,
    vertex_buffer_handle: SegmentHandle,
    uniform_buffer_handle: RcBlockHandle,
    first_instance: u32,
    instance_count: u32,
}

pub struct MultiDrawIndirectBuffer<
    Uniform: Clone + AsBytes + Ord,
    Bucket: Copy + Debug + DrawCallBucket,
    const BUCKET_COUNT: usize,
> {
    pub indirect_buffer: Buffer,
    pub vertex_buffer: Buffer,
    /// Despite its name, this ultimately leads to a storage buffer
    pub uniform_layout: UniformBinding,
    uniform_buffer: Buffer,
    indirect_buffer_allocator: BlockAllocator<DrawIndirectArgs>,
    vertex_buffer_allocator: PoolAllocator,
    uniform_buffer_allocator: RcBlockAllocator<Uniform>,
    draw_calls: BTreeMap<DrawCallHandle<Uniform, Bucket>, DrawCallData>,
    chunks_per_bucket: u64,
    buckets: [Bucket; BUCKET_COUNT],
    draw_count_per_bucket: [u64; BUCKET_COUNT],
    phantom_uniform: PhantomData<Uniform>,
    phantom_bucket: PhantomData<Bucket>,
}

const DRAW_ARGS_SIZE: usize = std::mem::size_of::<DrawIndirectArgs>();

impl<
        Uniform: Clone + Ord + AsBytes,
        Bucket: Copy + Debug + Ord + DrawCallBucket,
        const BUCKET_COUNT: usize,
    > MultiDrawIndirectBuffer<Uniform, Bucket, BUCKET_COUNT>
{
    pub fn new(
        device: &Device,
        label: &str,
        buckets: [Bucket; BUCKET_COUNT],
        chunks_per_bucket: u64,
        max_batch_size_map: &BTreeMap<Bucket, u64>,
    ) -> Self {
        let mut vertex_buffer_size_bytes = 0;
        for bucket in buckets {
            vertex_buffer_size_bytes += chunks_per_bucket as u64
                * bucket.instance_size()
                * *max_batch_size_map
                    .get(&bucket)
                    .expect("Bucket not valid key in max_batch_size_map");
        }

        // Indirect buffer contains one slot for every batch and bucket combination
        let indirect_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("indirect buffer ".to_owned() + label)),
            size: chunks_per_bucket * DRAW_ARGS_SIZE as u64 * BUCKET_COUNT as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("vertex buffer ".to_owned() + label)),
            size: vertex_buffer_size_bytes,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Storage buffer contains one slot for every batch, referred to by draw calls of multiple buckets
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("chunk uniform buffer ".to_owned() + label)),
            size: chunks_per_bucket * std::mem::size_of::<Uniform>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_buffer_allocator =
            BlockAllocator::new(chunks_per_bucket * BUCKET_COUNT as u64);
        let vertex_buffer_allocator = PoolAllocator::new(vertex_buffer_size_bytes);
        let uniform_buffer_allocator = RcBlockAllocator::new(chunks_per_bucket);

        Self {
            vertex_buffer,
            indirect_buffer,
            uniform_layout: UniformBinding::new(device, &uniform_buffer),
            uniform_buffer,
            indirect_buffer_allocator,
            vertex_buffer_allocator,
            uniform_buffer_allocator,
            draw_calls: BTreeMap::new(),
            chunks_per_bucket,
            buckets,
            draw_count_per_bucket: [0; BUCKET_COUNT],
            phantom_uniform: PhantomData,
            phantom_bucket: PhantomData,
        }
    }

    pub fn drop_region(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        handle: DrawCallHandle<Uniform, Bucket>,
    ) {
        let draw_call_data = self
            .draw_calls
            .remove(&handle)
            .expect("Invalid draw call handle provided");

        // If the region doesn't own the last indirect/uniform buffer slot, fill the slot with another region
        if (draw_call_data.indirect_buffer_handle % self.chunks_per_bucket) + 1
            < self.draw_count(handle.bucket)
        {
            // Find BufferRegionData instance in the same bucket with highest indirect buffer slot
            let (last_draw_call_handle, last_draw_call_data) = self
                .draw_calls
                .iter()
                .filter(|&(key, ..)| key.bucket == handle.bucket)
                .max_by_key(|(.., data)| data.indirect_buffer_handle)
                .unwrap();

            let new_indirect_buffer_handle = draw_call_data.indirect_buffer_handle;

            // Move last draw call to new empty slot
            self.indirect_buffer_allocator.allocate_block(
                &mut BufferMemoryTarget::new(&self.indirect_buffer, queue, command_encoder),
                &DrawIndirectArgs {
                    vertex_count: QUAD_VERTEX_COUNT,
                    instance_count: last_draw_call_data.instance_count,
                    first_vertex: QUAD_VERTEX_COUNT
                        * *last_draw_call_data.uniform_buffer_handle as u32,
                    first_instance: last_draw_call_data.first_instance,
                },
                new_indirect_buffer_handle,
            );

            self.draw_calls
                .entry(last_draw_call_handle.clone())
                .and_modify(|entry| entry.indirect_buffer_handle = new_indirect_buffer_handle);
        }

        self.vertex_buffer_allocator
            .deallocate(&draw_call_data.vertex_buffer_handle);

        self.draw_count_per_bucket[self.bucket_id(handle.bucket)] -= 1;
    }

    pub fn insert_region(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        bucket: Bucket,
        batch_vb: &Buffer,
        instance_count: u32,
        uniform: Uniform,
    ) -> DrawCallHandle<Uniform, Bucket> {
        if self.chunks_per_bucket < self.draw_count_per_bucket[self.bucket_id(bucket)] + 1 {
            panic!(
                "Not enough indirect buffer space available for {} regions in bucket {:?}",
                self.draw_count_per_bucket[self.bucket_id(bucket)] as usize + 1,
                bucket
            );
        }

        let vertex_buffer_handle = self.vertex_buffer_allocator.allocate_from_buffer(
            &mut BufferMemoryTarget::new(&self.vertex_buffer, queue, command_encoder),
            batch_vb,
            instance_count as u64 * bucket.instance_size(),
            bucket.instance_size(),
        );
        let first_instance = (vertex_buffer_handle.offset / bucket.instance_size()) as u32;

        // TODO use block_allocator functionality
        let indirect_buffer_handle = self.indirect_buffer_offset(bucket, self.draw_count(bucket));

        let uniform_buffer_handle = if let Some(entry) = self
            .draw_calls
            .iter()
            .find(|(key, _)| key.uniform == uniform)
        {
            entry.1.uniform_buffer_handle.clone()
        } else {
            let handle = self.uniform_buffer_allocator.allocate_first_free_block(
                &mut BufferMemoryTarget::new(&self.uniform_buffer, queue, command_encoder),
                &uniform,
            );

            handle
        };

        // TODO validate in block_allocator
        self.indirect_buffer_allocator.allocate_block(
            &mut BufferMemoryTarget::new(&self.indirect_buffer, queue, command_encoder),
            &DrawIndirectArgs {
                vertex_count: QUAD_VERTEX_COUNT,
                instance_count,
                first_vertex: QUAD_VERTEX_COUNT * *uniform_buffer_handle as u32,
                first_instance,
            },
            indirect_buffer_handle,
        );

        let draw_call_handle = DrawCallHandle { uniform, bucket };

        self.draw_calls.insert(
            draw_call_handle.clone(),
            DrawCallData {
                indirect_buffer_handle,
                vertex_buffer_handle,
                uniform_buffer_handle,
                first_instance,
                instance_count,
            },
        );

        self.draw_count_per_bucket[self.bucket_id(bucket)] += 1;

        draw_call_handle
    }

    pub fn draw_count(&self, bucket: Bucket) -> u64 {
        self.draw_count_per_bucket[self.bucket_id(bucket)]
    }

    fn bucket_id(&self, bucket: Bucket) -> usize {
        self.buckets
            .iter()
            .position(|&b| b == bucket)
            .expect("Invalid bucket provided")
    }

    pub fn indirect_buffer_offset(&self, bucket: Bucket, position: u64) -> u64 {
        self.chunks_per_bucket * self.bucket_id(bucket) as u64 + position
    }

    pub fn indirect_buffer_offset_bytes(&self, bucket: Bucket, position: u64) -> u64 {
        self.indirect_buffer_offset(bucket, position) * DRAW_ARGS_SIZE as u64
    }
}
