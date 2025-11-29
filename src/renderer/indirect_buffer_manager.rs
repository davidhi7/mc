use core::panic;
use std::collections::HashMap;
use std::hash::Hash;

use std::fmt::Debug;

use itertools::Itertools;
use wgpu::util::DrawIndirectArgs;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Queue};

pub use enum_map::{Enum, EnumArray, EnumMap};

use crate::renderer::buffers::block_allocator::{BlockAllocator, RcBlockAllocator, RcBlockHandle};
use crate::renderer::buffers::pool_allocator::{PoolAllocator, SegmentHandle};
use crate::renderer::buffers::{AsBytes, BufferMemoryTarget};
use crate::renderer::vertex_buffer::QUAD_VERTEX_COUNT;

/// Trait representing values that act as a bucket identifier for a class of draw calls.
pub trait InstanceSize: Copy {
    /// Size of a single instance, in bytes
    fn instance_size(self) -> u64;
}

/// Identifier for a single draw call, that is, one uniform and one bucket value.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct DrawCallHandle<Uniform, Bucket> {
    pub bucket: Bucket,
    pub uniform: Uniform,
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

/// Data for a prepared but not yet written draw call.
struct DrawCallCreationArgs<Uniform, Bucket> {
    bucket: Bucket,
    uniform: Uniform,
    vertex_buffer: Buffer,
    instance_count: u32,
}

pub struct IndirectBufferManager<
    Uniform: Clone + Debug + Hash + AsBytes,
    Bucket: Copy + Debug + Hash + Eq + InstanceSize + EnumArray<u64>,
> {
    /// Indirect buffer, containing all draw calls.
    pub indirect_buffer: Buffer,
    /// Vertex/instance buffer.
    pub vertex_buffer: Buffer,
    /// Storage buffer with per-draw call uniform values.
    pub uniform_buffer: Buffer,
    indirect_buffer_allocator: BlockAllocator<DrawIndirectArgs>,
    vertex_buffer_allocator: PoolAllocator,
    uniform_buffer_allocator: RcBlockAllocator<Uniform>,
    draw_calls: HashMap<DrawCallHandle<Uniform, Bucket>, DrawCallData>,
    chunks_per_bucket: u64,
    draw_count_per_bucket: EnumMap<Bucket, u64>,
}

const DRAW_ARGS_SIZE: u64 = std::mem::size_of::<DrawIndirectArgs>() as u64;

impl<Uniform, Bucket> IndirectBufferManager<Uniform, Bucket>
where
    Uniform: Clone + Debug + Hash + Eq + AsBytes,
    Bucket: Copy + Debug + Hash + Eq + Ord + InstanceSize + EnumArray<u64>,
{
    pub fn new(
        device: &Device,
        label: &str,
        buckets: &[Bucket],
        chunks_per_bucket: u64,
        max_batch_size_map: &HashMap<Bucket, u64>,
    ) -> Self {
        let mut vertex_buffer_size_bytes = 0;
        for bucket in buckets {
            vertex_buffer_size_bytes += chunks_per_bucket
                * bucket.instance_size()
                * *max_batch_size_map
                    .get(bucket)
                    .expect("Bucket not valid key in max_batch_size_map");
        }

        // Indirect buffer contains one slot for every batch and bucket combination
        let indirect_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("indirect buffer ".to_owned() + label)),
            size: chunks_per_bucket * DRAW_ARGS_SIZE * Bucket::LENGTH as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
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
            BlockAllocator::new(chunks_per_bucket * Bucket::LENGTH as u64);
        let vertex_buffer_allocator = PoolAllocator::new(vertex_buffer_size_bytes);
        let uniform_buffer_allocator = RcBlockAllocator::new(chunks_per_bucket);

        Self {
            indirect_buffer,
            vertex_buffer,
            uniform_buffer,
            indirect_buffer_allocator,
            vertex_buffer_allocator,
            uniform_buffer_allocator,
            draw_calls: HashMap::new(),
            chunks_per_bucket,
            draw_count_per_bucket: EnumMap::default(),
        }
    }

    pub fn submit(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        mut update_pass: IndirectBufferUpdatePass<Uniform, Bucket>,
    ) {
        update_pass
            .buf_new_draws
            .sort_unstable_by_key(|draw_call| draw_call.bucket);
        update_pass
            .buf_dropped_draws
            .sort_unstable_by_key(|draw_call| draw_call.bucket);

        // Try to match as many old with new draw calls with the same bucket, so we can prevent unneccessary indirect buffer moves
        for entry in update_pass
            .buf_new_draws
            .into_iter()
            .zip_longest(update_pass.buf_dropped_draws)
        {
            match entry {
                itertools::EitherOrBoth::Both(new_drawcall, old_handle) => {
                    if new_drawcall.bucket == old_handle.bucket {
                        let draw_call_data = self
                            .draw_calls
                            .remove(&old_handle)
                            .expect("replace: Invalid draw call handle provided");

                        self.vertex_buffer_allocator
                            .deallocate(draw_call_data.vertex_buffer_handle);
                        self.draw_count_per_bucket[old_handle.bucket] -= 1;
                        let handle = draw_call_data.indirect_buffer_handle;
                        drop(draw_call_data);

                        self.insert_region_at(
                            queue,
                            command_encoder,
                            handle,
                            new_drawcall.bucket,
                            &new_drawcall.vertex_buffer,
                            new_drawcall.instance_count,
                            new_drawcall.uniform,
                        );
                        println!("both updated");
                    } else {
                        self.drop_region(queue, command_encoder, old_handle);
                        self.insert_region(queue, command_encoder, new_drawcall);
                    }
                }
                itertools::EitherOrBoth::Left(new) => {
                    self.insert_region(queue, command_encoder, new);
                }
                itertools::EitherOrBoth::Right(old) => {
                    self.drop_region(queue, command_encoder, old)
                }
            }
        }
    }

    fn drop_region(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        handle: DrawCallHandle<Uniform, Bucket>,
    ) {
        let draw_call_data = self
            .draw_calls
            .remove(&handle)
            .expect("drop_region: Invalid draw call handle provided");

        // If the draw call doesn't own the last indirect/uniform buffer slot, fill the slot with another region
        if (draw_call_data.indirect_buffer_handle % self.chunks_per_bucket) + 1
            < self.draw_count(handle.bucket)
        {
            // Find BufferRegionData instance in the same bucket with highest indirect buffer slot
            let (_, last_draw_call_data) = self
                .draw_calls
                .iter_mut()
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

            last_draw_call_data.indirect_buffer_handle = new_indirect_buffer_handle;
        }

        self.vertex_buffer_allocator
            .deallocate(draw_call_data.vertex_buffer_handle);

        self.draw_count_per_bucket[handle.bucket] -= 1;
    }

    fn insert_region_at(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        indirect_buffer_handle: u64,
        bucket: Bucket,
        vertex_buffer: &Buffer,
        instance_count: u32,
        uniform: Uniform,
    ) {
        let vertex_buffer_handle = self.vertex_buffer_allocator.allocate_from_buffer(
            vertex_buffer,
            &mut BufferMemoryTarget::new(&self.vertex_buffer, queue, command_encoder),
            instance_count as u64 * bucket.instance_size(),
            bucket.instance_size(),
        );
        let first_instance = (vertex_buffer_handle.offset / bucket.instance_size()) as u32;

        let uniform_buffer_handle = if let Some(entry) = self
            .draw_calls
            .iter()
            .find(|(key, _)| key.uniform == uniform)
        {
            entry.1.uniform_buffer_handle.clone()
        } else {
            self.uniform_buffer_allocator.allocate_first_free_block(
                &mut BufferMemoryTarget::new(&self.uniform_buffer, queue, command_encoder),
                &uniform,
            )
        };

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
            draw_call_handle,
            DrawCallData {
                indirect_buffer_handle,
                vertex_buffer_handle,
                uniform_buffer_handle,
                first_instance,
                instance_count,
            },
        );

        self.draw_count_per_bucket[bucket] += 1;
    }

    fn insert_region(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        DrawCallCreationArgs {
            bucket,
            vertex_buffer,
            instance_count,
            uniform,
        }: DrawCallCreationArgs<Uniform, Bucket>,
        // ) -> DrawCallHandle<Uniform, Bucket> {
    ) {
        if self.chunks_per_bucket < self.draw_count_per_bucket[bucket] + 1 {
            panic!(
                "Not enough indirect buffer space available for {} regions in bucket {:?}",
                self.draw_count(bucket) as usize + 1,
                bucket
            );
        }

        let indirect_buffer_handle =
            self.indirect_buffer_offset_draw_calls(bucket) + self.draw_count(bucket);

        self.insert_region_at(
            queue,
            command_encoder,
            indirect_buffer_handle,
            bucket,
            &vertex_buffer,
            instance_count,
            uniform,
        );
    }

    pub fn draw_count(&self, bucket: Bucket) -> u64 {
        self.draw_count_per_bucket[bucket]
    }

    fn indirect_buffer_offset_draw_calls(&self, bucket: Bucket) -> u64 {
        self.chunks_per_bucket * bucket.into_usize() as u64
    }

    pub fn indirect_buffer_offset(&self, bucket: Bucket) -> u64 {
        self.indirect_buffer_offset_draw_calls(bucket)
            * std::mem::size_of::<DrawIndirectArgs>() as u64
    }
}

pub struct IndirectBufferUpdatePass<Uniform, Bucket> {
    buf_new_draws: Vec<DrawCallCreationArgs<Uniform, Bucket>>,
    buf_dropped_draws: Vec<DrawCallHandle<Uniform, Bucket>>,
}

impl<Uniform: Clone, Bucket: Copy> IndirectBufferUpdatePass<Uniform, Bucket> {
    pub fn new() -> Self {
        Self {
            buf_new_draws: Vec::new(),
            buf_dropped_draws: Vec::new(),
        }
    }
    pub fn prepare_drop_region(&mut self, handle: DrawCallHandle<Uniform, Bucket>) {
        self.buf_dropped_draws.push(handle);
    }

    pub fn prepare_insert_region(
        &mut self,
        bucket: Bucket,
        vertex_buffer: Buffer,
        instance_count: u32,
        uniform: Uniform,
    ) -> DrawCallHandle<Uniform, Bucket> {
        // TODO somehow this should not dirctly return a complete draw call handle
        self.buf_new_draws.push(DrawCallCreationArgs {
            bucket,
            vertex_buffer,
            instance_count,
            uniform: uniform.clone(),
        });

        DrawCallHandle { uniform, bucket }
    }
}
