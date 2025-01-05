use core::panic;
use std::collections::HashMap;
use std::hash::Hash;
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

// TODO assess if traits are required
pub trait DrawCallBucket: Copy + Debug + Eq + Hash {
    /// Size of a single instance, in bytes
    fn instance_size(&self) -> u64;
}

// TODO assess if traits are required
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DrawCallHandle<Uniform: Clone + Copy, Bucket: Clone + Copy> {
    pub uniform: Uniform,
    pub bucket: Bucket,
}

#[derive(Debug)]
pub struct DrawCallData {
    indirect_buffer_handle: u64,
    vertex_allocator_handle: SegmentHandle,
    uniform_allocator_handle: RcBlockHandle,
    first_instance: u32,
    instance_count: u32,
}

pub struct MultiDrawIndirectBuffer<
    Uniform: Clone + Copy + Debug + Hash + Eq + AsBytes,
    Bucket: DrawCallBucket,
    const BUCKET_COUNT: usize,
> {
    pub indirect_buffer: Buffer,
    pub vertex_buffer: Buffer,
    uniform_buffer: Buffer,
    pub uniform_bind_group_layout: BindGroupLayout,
    pub uniform_bind_group: BindGroup,
    indirect_buffer_allocator: BlockAllocator<DrawIndirectArgs>,
    vertex_buffer_allocator: PoolAllocator,
    uniform_buffer_allocator: RcBlockAllocator<Uniform>,
    draw_calls: HashMap<DrawCallHandle<Uniform, Bucket>, DrawCallData>,
    // TOOD rename
    batches_count: u64,
    buckets: [Bucket; BUCKET_COUNT],
    draw_count_per_bucket: [u64; BUCKET_COUNT],
    phantom_uniform: PhantomData<Uniform>,
    phantom_bucket: PhantomData<Bucket>,
}

const DRAW_ARGS_SIZE: usize = std::mem::size_of::<DrawIndirectArgs>();

impl<
        Uniform: Clone + Copy + Debug + Hash + Eq + AsBytes,
        Bucket: DrawCallBucket,
        const BUCKET_COUNT: usize,
    > MultiDrawIndirectBuffer<Uniform, Bucket, BUCKET_COUNT>
{
    pub fn new(
        device: &Device,
        label: &str,
        buckets: [Bucket; BUCKET_COUNT],
        batches_count: u64,
        max_batch_size_map: &HashMap<Bucket, u64>,
    ) -> Self {
        let mut empty_regions = BTreeMap::new();

        let mut vertex_buffer_size_bytes = 0;
        for bucket in buckets {
            vertex_buffer_size_bytes += batches_count as u64
                * bucket.instance_size()
                * *max_batch_size_map
                    .get(&bucket)
                    .expect("Bucket not valid key in max_batch_size_map");
        }

        // Indirect buffer contains one slot for every batch and bucket combination
        let indirect_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("indirect buffer ".to_owned() + label)),
            size: batches_count * DRAW_ARGS_SIZE as u64 * BUCKET_COUNT as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("vertex buffer ".to_owned() + label)),
            size: vertex_buffer_size_bytes,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Uniform/storage buffer contains one slot for every batch, optionally referred to by draw calls of multiple buckets
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("chunk uniform buffer ".to_owned() + label)),
            size: batches_count * std::mem::size_of::<Uniform>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        empty_regions.insert(0, vertex_buffer_size_bytes);

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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

        let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("uniform bind group layout"),
            layout: &uniform_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let indirect_buffer_allocator = BlockAllocator::new(batches_count * BUCKET_COUNT as u64);
        let vertex_buffer_allocator = PoolAllocator::new(vertex_buffer_size_bytes);
        let uniform_buffer_allocator = RcBlockAllocator::new(batches_count);

        Self {
            indirect_buffer_allocator,
            indirect_buffer,
            uniform_bind_group_layout,
            uniform_bind_group,
            // occupied_regions: HashMap::new(),
            uniform_buffer_allocator,
            vertex_buffer_allocator,
            batches_count,
            buckets,
            draw_count_per_bucket: [0; BUCKET_COUNT],
            // stored_uniform_values: HashMap::new(),
            phantom_uniform: PhantomData,
            phantom_bucket: PhantomData,
            vertex_buffer,
            uniform_buffer,
            draw_calls: HashMap::new(),
        }
    }

    pub fn drop_region(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        handle: DrawCallHandle<Uniform, Bucket>,
    ) {
        // TODO is clone neccessary here?
        let draw_call_data = self.draw_calls.get(&handle).expect("TODO");

        // If the region doesn't own the last indirect/uniform buffer slot, fill the slot with another region
        if draw_call_data.indirect_buffer_handle % self.batches_count + 1
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
                    vertex_count: 4,
                    instance_count: last_draw_call_data.instance_count,
                    first_vertex: 4 * *last_draw_call_data.uniform_allocator_handle as u32,
                    first_instance: last_draw_call_data.first_instance,
                },
                new_indirect_buffer_handle,
            );

            // TODO improve
            self.draw_calls
                .entry(last_draw_call_handle.clone())
                .and_modify(|entry| entry.indirect_buffer_handle = new_indirect_buffer_handle);
        }

        self.vertex_buffer_allocator.deallocate(
            &self
                .draw_calls
                .remove(&handle)
                .expect("Region `target_region` key missing in `occupied_regions`")
                .vertex_allocator_handle,
        );

        self.draw_count_per_bucket[self.bucket_id(handle.bucket)] -= 1;

        self.draw_calls.remove(&handle);
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
        if self.batches_count < self.draw_count_per_bucket[self.bucket_id(bucket)] + 1 {
            panic!(
                "Not enough indirect buffer space available for {} regions in bucket {:?}",
                self.draw_count_per_bucket[self.bucket_id(bucket)] as usize + 1,
                bucket
            );
        }

        let vertex_allocator_handle = self.vertex_buffer_allocator.allocate_from_buffer(
            &mut BufferMemoryTarget::new(&self.vertex_buffer, queue, command_encoder),
            batch_vb,
            instance_count as u64 * bucket.instance_size(),
            bucket.instance_size(),
        );
        let first_instance = (vertex_allocator_handle.offset / bucket.instance_size()) as u32;

        // TODO use block_allocator functionality
        let indirect_buffer_handle =
            self.indirect_buffer_bucket_position_offset(bucket, self.draw_count(bucket));

        let uniform_allocator_handle = if let Some(entry) = self
            .draw_calls
            .iter()
            .find(|(key, value)| key.uniform == uniform)
        {
            entry.1.uniform_allocator_handle.clone()
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
                vertex_count: 4,
                instance_count: instance_count,
                first_vertex: 4 * *uniform_allocator_handle as u32,
                first_instance: first_instance,
            },
            indirect_buffer_handle,
        );

        let draw_call_handle = DrawCallHandle { uniform, bucket };

        self.draw_calls.insert(
            draw_call_handle,
            DrawCallData {
                indirect_buffer_handle,
                vertex_allocator_handle,
                uniform_allocator_handle,
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

    // TODO function für position mit offset
    pub fn indirect_buffer_bucket_offset(&self, bucket: Bucket) -> u64 {
        self.batches_count * self.bucket_id(bucket) as u64
    }

    pub fn indirect_buffer_bucket_position_offset(&self, bucket: Bucket, position: u64) -> u64 {
        self.indirect_buffer_bucket_offset(bucket) + position
    }

    pub fn indirect_buffer_bucket_offset_bytes(&self, bucket: Bucket) -> u64 {
        self.indirect_buffer_bucket_offset(bucket) * DRAW_ARGS_SIZE as u64
    }
}
