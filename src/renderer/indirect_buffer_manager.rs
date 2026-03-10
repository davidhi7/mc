use core::panic;
use std::collections::HashMap;
use std::hash::Hash;

use std::fmt::Debug;
use std::marker::PhantomData;

use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use itertools::Itertools;
use wgpu::util::DrawIndirectArgs;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Queue};

use enum_map::{EnumArray, EnumMap};

use crate::logging::ReadableBytes;
use crate::renderer::buffers::block_allocator::{
    BlockAllocator, BlockHandle, CountedBlockAllocator, CountedBlockHandle,
};
use crate::renderer::buffers::pool_allocator::{PoolAllocator, SegmentHandle};
use crate::renderer::buffers::{AllocationError, BufferMemoryTarget};
use crate::renderer::vertex_buffer::QUAD_VERTEX_COUNT;
use crate::world::chunk::ChunkUVW;

/// Trait representing values that act as a bucket identifier for a class of draw calls.
// EnumArray<T> is implemented if T derives Enum
#[expect(private_bounds)]
pub trait DrawCallBucket:
    Copy
    + Debug
    + Hash
    + Eq
    + Ord
    + EnumArray<u64>
    + EnumArray<IndirectBufferConfig>
    + EnumArray<HashMap<ChunkUniform, DrawCallData<Self>>>
{
    /// Size of a single instance, in bytes.
    fn instance_size(self) -> u64;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Zeroable, Pod)]
#[repr(C)]
pub struct ChunkUniform {
    uvw: IVec3,
    // padding is used for temporary state in compute shaders but not meant to be read by the CPU
    _padding: i32,
}

impl From<ChunkUVW> for ChunkUniform {
    fn from(value: ChunkUVW) -> Self {
        Self {
            uvw: value.into(),
            _padding: 0,
        }
    }
}

/// Identifier for a draw call, consisting of a chunk uniform and a bucket.
/// evaluate what traits are needed
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DrawCallHandle<Bucket> {
    pub bucket: Bucket,
    pub uniform: ChunkUniform,
}

/// All data associated with a draw call.
pub struct DrawCallData<Bucket: DrawCallBucket> {
    indirect_buffer_handle: IndirectBufferHandle<Bucket>,
    vertex_buffer_handle: SegmentHandle,
    uniform_buffer_handle: CountedBlockHandle<ChunkUniform>,
    instance_count: u32,
}

/// Data for a prepared but not yet written draw call.
struct DrawCallCreationArgs<Bucket> {
    bucket: Bucket,
    uniform: ChunkUniform,
    vertex_buffer: Buffer,
    instance_count: u32,
}

/// Data for a draw call that is prepared to be updated, that is the vertex buffer contents are to be replaced.
struct DrawCallUpdateArgs<Bucket> {
    draw_call: DrawCallHandle<Bucket>,
    vertex_buffer: Buffer,
    instance_count: u32,
}

/// Indirect buffer allocator and metadata for one bucket type.
struct IndirectBufferConfig {
    allocator: BlockAllocator<DrawIndirectArgs>,
    draw_count: u64,
}

/// Bucket and block handle for accessing indirect buffer slots.
struct IndirectBufferHandle<Bucket: DrawCallBucket> {
    bucket: Bucket,
    handle: BlockHandle<DrawIndirectArgs>,
}

/// Utility struct that manages allocations to an indirect buffer, aware of multiple draw call classes (bucket).
struct IndirectBufferAllocator<Bucket: DrawCallBucket> {
    chunks_per_bucket: u64,
    allocators: EnumMap<Bucket, IndirectBufferConfig>,
    /// Indirect buffer, containing all draw calls.
    /// Contains one slot for every chunk and bucket combination.
    /// Stores [`wgpu::util::DrawIndirectArgs`] instances.
    buffer: Buffer,
}

impl<Bucket: DrawCallBucket> IndirectBufferAllocator<Bucket> {
    fn new(chunks_per_bucket: u64, buffer: Buffer) -> Self {
        Self {
            chunks_per_bucket,
            allocators: EnumMap::from_fn(|_| IndirectBufferConfig {
                allocator: BlockAllocator::new(chunks_per_bucket),
                draw_count: 0,
            }),
            buffer,
        }
    }

    /// Allocate the first free block for the given bucket and increments the draw count by one. Returns error if no free bucket is available.
    fn allocate_first_free_block(
        &mut self,
        bucket: Bucket,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        data: &DrawIndirectArgs,
    ) -> Result<IndirectBufferHandle<Bucket>, AllocationError> {
        let target = &mut BufferMemoryTarget::new(&self.buffer, queue, command_encoder)
            .with_global_offset(self.offset(bucket))
            .with_limit(self.bytes_per_bucket());

        let allocator = &mut self.allocators[bucket];
        let first_free_block = allocator.allocator.first_free_block()?;

        allocator
            .allocator
            .allocate_block(target, first_free_block, data)?;
        allocator.draw_count += 1;

        Ok(IndirectBufferHandle {
            bucket,
            handle: first_free_block,
        })
    }

    /// Writes data into given block, does not increment draw count. Returns error if the handle is invalid.
    fn overwrite_block(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        handle: &IndirectBufferHandle<Bucket>,
        data: &DrawIndirectArgs,
    ) -> Result<(), AllocationError> {
        let target = &mut BufferMemoryTarget::new(&self.buffer, queue, command_encoder)
            .with_global_offset(self.offset(handle.bucket))
            .with_limit(self.bytes_per_bucket());

        self.allocators[handle.bucket]
            .allocator
            .overwrite_block(target, handle.handle, data)
    }

    /// Allocate the first free block for the given bucket and increments the draw count by one. Returns error if no free bucket is available.
    fn deallocate_last_block(&mut self, bucket: Bucket) {
        let allocator = &mut self.allocators[bucket];
        assert!(
            allocator.draw_count > 0,
            "Tried to deallocate last block while no block is allocated"
        );

        allocator
            .allocator
            .deallocate_block(BlockHandle(allocator.draw_count - 1, PhantomData))
            .expect("Last block should always be allocated if draw count is greater than 0");
        allocator.draw_count -= 1;
    }

    fn clear(&mut self) {
        for allocator in self.allocators.values_mut() {
            allocator.draw_count = 0;
            allocator.allocator.clear();
        }
    }

    /// Count of active draw calls
    fn draw_count(&self, bucket: Bucket) -> u64 {
        self.allocators[bucket].draw_count
    }

    /// Size of indirect buffer segment for one bucket type, in bytes.
    fn bytes_per_bucket(&self) -> u64 {
        self.chunks_per_bucket * std::mem::size_of::<DrawIndirectArgs>() as u64
    }

    /// Offset measured in draw calls.
    fn offset_draw_calls(&self, bucket: Bucket) -> u64 {
        self.chunks_per_bucket * bucket.into_usize() as u64
    }

    /// Offset measured in bytes.
    fn offset(&self, bucket: Bucket) -> u64 {
        self.offset_draw_calls(bucket) * std::mem::size_of::<DrawIndirectArgs>() as u64
    }
}

/// Associated data to one insertion into the vertex buffer.
struct VertexBufferInsertionTask {
    vertex_buffer_resize: Option<u64>,
    vertex_buffer_segment: SegmentHandle,
    source_buffer: Buffer,
}

pub struct IndirectBufferManager<Bucket: DrawCallBucket> {
    buffer_label: String,
    /// Vertex/instance buffer. Contains all draw call geometry data.
    vertex_buffer: Buffer,
    /// Storage buffer with per-draw call uniform values.
    /// Contains one slot for every chunk.
    /// Stores [`ChunkUniform`] instances.
    uniform_buffer: Buffer,
    indirect_buffer_allocator: IndirectBufferAllocator<Bucket>,
    vertex_buffer_allocator: PoolAllocator,
    uniform_buffer_allocator: CountedBlockAllocator<ChunkUniform>,
    draw_calls: EnumMap<Bucket, HashMap<ChunkUniform, DrawCallData<Bucket>>>,
    uniforms: HashMap<ChunkUniform, CountedBlockHandle<ChunkUniform>>,
}

impl<Bucket: DrawCallBucket> IndirectBufferManager<Bucket> {
    pub fn new(device: &Device, buffer_label: String, chunks_per_bucket: u64) -> Self {
        // Start with 1MiB
        let vertex_buffer_size = 1024u64.pow(2);

        let indirect_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("indirect buffer {buffer_label}")),
            size: chunks_per_bucket
                * Bucket::LENGTH as u64
                * std::mem::size_of::<DrawIndirectArgs>() as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("vertex buffer {buffer_label}")),
            size: vertex_buffer_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("uniform buffer {buffer_label}")),
            size: chunks_per_bucket * std::mem::size_of::<ChunkUniform>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer_label,
            vertex_buffer,
            uniform_buffer,
            indirect_buffer_allocator: IndirectBufferAllocator::new(
                chunks_per_bucket,
                indirect_buffer,
            ),
            vertex_buffer_allocator: PoolAllocator::new(vertex_buffer_size),
            uniform_buffer_allocator: CountedBlockAllocator::new(chunks_per_bucket),
            draw_calls: EnumMap::default(),
            uniforms: HashMap::default(),
        }
    }

    /// Insert region.
    ///
    /// This involves:
    /// - Allocating the vertex buffer contents
    /// - Allocating the uniform if not already existing
    /// - Allocating the indirect draw call
    ///
    /// If the draw call is followed by other draw calls of the same bucket, the last draw call is moved into the now free slot to guarantee a continuous sequence of active draw calls.
    fn insert_region(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        existing_buffer_handle: Option<IndirectBufferHandle<Bucket>>,
        DrawCallCreationArgs {
            bucket,
            uniform,
            vertex_buffer,
            instance_count,
        }: DrawCallCreationArgs<Bucket>,
    ) -> VertexBufferInsertionTask {
        let insertion_task = self.reserve_from_vertex_buffer(bucket, vertex_buffer, instance_count);

        let uniform_buffer_handle = self
            .find_or_allocate_uniform(uniform, queue, command_encoder)
            .expect("Chunk uniform allocation failed");

        let draw_indirect_args = Self::construct_draw_indirect_args(
            bucket,
            insertion_task.vertex_buffer_segment,
            uniform_buffer_handle,
            instance_count,
        );

        let indirect_buffer_handle = match existing_buffer_handle {
            Some(handle) => {
                self.indirect_buffer_allocator
                    .overwrite_block(queue, command_encoder, &handle, &draw_indirect_args)
                    .expect("Invalid indirect buffer handle provided");
                handle
            }
            None => self
                .indirect_buffer_allocator
                .allocate_first_free_block(bucket, queue, command_encoder, &draw_indirect_args)
                .unwrap_or_else(|_| {
                    panic!(
                        "Not enough indirect buffer space available for {} regions in bucket {:?}",
                        self.draw_count(bucket) + 1,
                        bucket
                    )
                }),
        };

        self.draw_calls[bucket].insert(
            uniform,
            DrawCallData {
                indirect_buffer_handle,
                vertex_buffer_handle: insertion_task.vertex_buffer_segment,
                uniform_buffer_handle,
                instance_count,
            },
        );

        insertion_task
    }

    /// Drop region.
    ///
    /// This involves:
    /// - Deallocating the vertex buffer contents
    /// - Deallocating the uniform buffer contents if they are not used for another region
    /// - Deallocating the indirect draw call
    ///
    /// If the draw call is followed by other draw calls of the same bucket, the last draw call is moved into the now free slot to guarantee a continuous sequence of active draw calls.
    fn drop_region(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        handle: DrawCallHandle<Bucket>,
    ) {
        let draw_call_data = self.draw_calls[handle.bucket]
            .remove(&handle.uniform)
            .expect("Attempted to drop invalid draw call");

        self.vertex_buffer_allocator
            .deallocate(draw_call_data.vertex_buffer_handle)
            .expect("Invalid vertex buffer handle associated to dropped draw call");

        self.decrement_uniform(handle.uniform)
            .expect("Invalid uniform buffer handle associated to dropped draw call");

        // If the draw call doesn't own the last indirect/uniform buffer slot, fill the slot with another active draw call of the same bucket
        if draw_call_data.indirect_buffer_handle.handle.0
            != self.indirect_buffer_allocator.draw_count(handle.bucket) - 1
        {
            // Perform swap-and-remove
            // Find draw call in the same bucket with highest indirect buffer slot
            let (_, last_draw_call_data) = self.draw_calls[handle.bucket]
                .iter_mut()
                .max_by_key(|(_, data)| data.indirect_buffer_handle.handle.0)
                .expect("There should be at least one active draw call remaining");

            // Move last draw call to new empty slot
            self.indirect_buffer_allocator
                .overwrite_block(
                    queue,
                    command_encoder,
                    &draw_call_data.indirect_buffer_handle,
                    &Self::construct_draw_indirect_args(
                        handle.bucket,
                        last_draw_call_data.vertex_buffer_handle,
                        last_draw_call_data.uniform_buffer_handle,
                        last_draw_call_data.instance_count,
                    ),
                )
                .expect("Existing indirect buffer handle should still be valid");

            last_draw_call_data.indirect_buffer_handle = draw_call_data.indirect_buffer_handle;
        }

        self.indirect_buffer_allocator
            .deallocate_last_block(handle.bucket);
    }

    fn reserve_from_vertex_buffer(
        &mut self,
        bucket: Bucket,
        source_buffer: Buffer,
        instance_count: u32,
    ) -> VertexBufferInsertionTask {
        let mut vertex_buffer_resize = None;
        let vertex_buffer_segment = match self.vertex_buffer_allocator.reserve_segment(
            instance_count as u64 * bucket.instance_size(),
            bucket.instance_size(),
        ) {
            Ok(segment) => segment,
            Err(_) => {
                let old_size = self.vertex_buffer_allocator.size();
                let new_size = u64::max(
                    old_size * 3 / 2,
                    old_size + instance_count as u64 * bucket.instance_size(),
                );
                vertex_buffer_resize = Some(new_size);
                self.vertex_buffer_allocator.grow(new_size);
                self.vertex_buffer_allocator
                    .reserve_segment(
                        instance_count as u64 * bucket.instance_size(),
                        bucket.instance_size(),
                    )
                    .expect("Segment reservation failed even after growing the buffer")
            }
        };

        VertexBufferInsertionTask {
            vertex_buffer_resize,
            vertex_buffer_segment,
            source_buffer,
        }
    }

    fn find_or_allocate_uniform(
        &mut self,
        uniform: ChunkUniform,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
    ) -> Result<CountedBlockHandle<ChunkUniform>, AllocationError> {
        if let Some(&handle) = self.uniforms.get(&uniform) {
            self.uniform_buffer_allocator.increment_counter(handle)?;
            Ok(handle)
        } else {
            let handle = self.uniform_buffer_allocator.allocate_first_free_block(
                &mut BufferMemoryTarget::new(&self.uniform_buffer, queue, command_encoder),
                &uniform,
            )?;
            self.uniforms.insert(uniform, handle);
            Ok(handle)
        }
    }

    fn decrement_uniform(&mut self, uniform: ChunkUniform) -> Result<(), AllocationError> {
        let handle = self
            .uniforms
            .get(&uniform)
            .expect("Uniform not currently stored in buffer");
        if self
            .uniform_buffer_allocator
            .decrement_counter(*handle)?
            .is_none()
        {
            self.uniforms.remove(&uniform);
        }

        Ok(())
    }

    fn replace_region_vertex_data(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        DrawCallUpdateArgs {
            draw_call,
            vertex_buffer,
            instance_count,
        }: DrawCallUpdateArgs<Bucket>,
    ) -> VertexBufferInsertionTask {
        let insertion_task =
            self.reserve_from_vertex_buffer(draw_call.bucket, vertex_buffer, instance_count);

        let draw_call_data = self.draw_calls[draw_call.bucket]
            .get_mut(&draw_call.uniform)
            .expect("Invalid draw call provided for replace");

        self.vertex_buffer_allocator
            .deallocate(draw_call_data.vertex_buffer_handle)
            .expect("Invalid handle provided");

        self.indirect_buffer_allocator
            .overwrite_block(
                queue,
                command_encoder,
                &draw_call_data.indirect_buffer_handle,
                &Self::construct_draw_indirect_args(
                    draw_call.bucket,
                    insertion_task.vertex_buffer_segment,
                    draw_call_data.uniform_buffer_handle,
                    instance_count,
                ),
            )
            .expect("Invalid indirect buffer handle provided");

        draw_call_data.vertex_buffer_handle = insertion_task.vertex_buffer_segment;

        insertion_task
    }

    pub fn create_update_pass(&mut self) -> IndirectBufferUpdatePass<'_, Bucket> {
        IndirectBufferUpdatePass {
            owner: self,
            new_draws: Vec::new(),
            dropped_draws: Vec::new(),
            updated_draws: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.draw_calls.clear();
        self.uniforms.clear();
        self.indirect_buffer_allocator.clear();
        self.vertex_buffer_allocator.clear();
        self.uniform_buffer_allocator.clear();
    }

    fn submit(
        &mut self,
        device: &Device,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        mut new_draws: Vec<DrawCallCreationArgs<Bucket>>,
        mut dropped_draws: Vec<DrawCallHandle<Bucket>>,
        updated_draws: Vec<DrawCallUpdateArgs<Bucket>>,
    ) {
        new_draws.sort_unstable_by_key(|draw_call| draw_call.bucket);
        dropped_draws.sort_unstable_by_key(|draw_call| draw_call.bucket);

        let mut vertex_buffer_resize = None;
        let mut vertex_buffer_insertions = Vec::new();

        let mut handle_insertion_result =
            |VertexBufferInsertionTask {
                 vertex_buffer_resize: resize_required,
                 vertex_buffer_segment,
                 source_buffer,
             }: VertexBufferInsertionTask| {
                if resize_required.is_some() {
                    vertex_buffer_resize = resize_required;
                }
                vertex_buffer_insertions.push((source_buffer, vertex_buffer_segment));
            };

        for updated_draw in updated_draws {
            handle_insertion_result(self.replace_region_vertex_data(
                queue,
                command_encoder,
                updated_draw,
            ));
        }

        // Try to match as many old with new draw calls with the same bucket, so we can prevent unneccessary indirect buffer draw call moves
        for entry in new_draws.into_iter().zip_longest(dropped_draws) {
            match entry {
                itertools::EitherOrBoth::Both(new_drawcall, old_handle) => {
                    if new_drawcall.bucket == old_handle.bucket {
                        let draw_call_data = self.draw_calls[old_handle.bucket]
                            .remove(&old_handle.uniform)
                            .expect("Invalid or inactive draw call handle provided for drop");

                        let DrawCallData {
                            indirect_buffer_handle,
                            vertex_buffer_handle,
                            uniform_buffer_handle,
                            ..
                        } = draw_call_data;

                        self.vertex_buffer_allocator
                            .deallocate(vertex_buffer_handle)
                            .expect("Invalid vertex buffer handle associated to dropped draw call");
                        self.uniform_buffer_allocator
                            .decrement_counter(uniform_buffer_handle)
                            .expect(
                                "Invalid uniform buffer handle associated to dropped draw call",
                            );

                        let output = self.insert_region(
                            queue,
                            command_encoder,
                            Some(indirect_buffer_handle),
                            new_drawcall,
                        );
                        handle_insertion_result(output);
                    } else {
                        self.drop_region(queue, command_encoder, old_handle);
                        let output = self.insert_region(queue, command_encoder, None, new_drawcall);
                        handle_insertion_result(output);
                    }
                }
                itertools::EitherOrBoth::Left(new_args) => {
                    handle_insertion_result(self.insert_region(
                        queue,
                        command_encoder,
                        None,
                        new_args,
                    ));
                }
                itertools::EitherOrBoth::Right(old) => {
                    self.drop_region(queue, command_encoder, old)
                }
            }
        }

        if let Some(new_size) = vertex_buffer_resize {
            log::info!("Grow vertex buffer to {}", ReadableBytes(new_size));
            let new_vertex_buffer = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("vertex buffer {}", self.buffer_label)),
                size: new_size,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            command_encoder.copy_buffer_to_buffer(
                &self.vertex_buffer,
                0,
                &new_vertex_buffer,
                0,
                self.vertex_buffer.size(),
            );
            self.vertex_buffer = new_vertex_buffer;
        }

        for (source_buffer, vertex_buffer_segment) in vertex_buffer_insertions {
            self.vertex_buffer_allocator.insert_into_segment(
                &source_buffer,
                &mut BufferMemoryTarget::new(&self.vertex_buffer, queue, command_encoder),
                vertex_buffer_segment,
            );
        }
    }

    pub fn vertex_buffer(&self) -> &Buffer {
        &self.vertex_buffer
    }

    pub fn uniform_buffer(&self) -> &Buffer {
        &self.uniform_buffer
    }

    pub fn indirect_buffer(&self) -> &Buffer {
        &self.indirect_buffer_allocator.buffer
    }

    /// Offset measured in bytes.
    pub fn indirect_buffer_offset(&self, bucket: Bucket) -> u64 {
        self.indirect_buffer_allocator.offset(bucket)
    }

    /// Count of active draw calls
    pub fn draw_count(&self, bucket: Bucket) -> u64 {
        self.indirect_buffer_allocator.draw_count(bucket)
    }

    /// Maximum number of chunks per bucket
    pub fn chunks_per_bucket(&self) -> u64 {
        self.indirect_buffer_allocator.chunks_per_bucket
    }

    fn construct_draw_indirect_args<T>(
        bucket: Bucket,
        vertex_buffer_segment: SegmentHandle,
        uniform_buffer_handle: CountedBlockHandle<T>,
        instance_count: u32,
    ) -> DrawIndirectArgs {
        DrawIndirectArgs {
            vertex_count: QUAD_VERTEX_COUNT,
            instance_count,
            first_vertex: QUAD_VERTEX_COUNT * uniform_buffer_handle.0 as u32,
            first_instance: (vertex_buffer_segment.offset / bucket.instance_size()) as u32,
        }
    }
}

pub struct IndirectBufferUpdatePass<'a, Bucket: DrawCallBucket> {
    owner: &'a mut IndirectBufferManager<Bucket>,
    new_draws: Vec<DrawCallCreationArgs<Bucket>>,
    dropped_draws: Vec<DrawCallHandle<Bucket>>,
    updated_draws: Vec<DrawCallUpdateArgs<Bucket>>,
}

impl<'a, Bucket: DrawCallBucket> IndirectBufferUpdatePass<'a, Bucket> {
    /// Prepare inserting a new region.
    /// This function panics if there currently exsists a region with the same chunk uniform and bucket,
    /// even if this region has been prepared to be dropped within this update pass.
    pub fn prepare_insert_region(
        &mut self,
        bucket: Bucket,
        vertex_buffer: Buffer,
        instance_count: u32,
        uniform: impl Into<ChunkUniform>,
    ) -> DrawCallHandle<Bucket> {
        let uniform = uniform.into();
        let handle = DrawCallHandle { uniform, bucket };
        if self.owner.draw_calls[handle.bucket].contains_key(&handle.uniform) {
            panic!("Region prepared for insertion conflicts with an already existing region");
        }

        self.new_draws.push(DrawCallCreationArgs {
            bucket,
            vertex_buffer,
            instance_count,
            uniform,
        });

        handle
    }

    /// Prepare to drop a region.
    pub fn prepare_drop_region(&mut self, handle: DrawCallHandle<Bucket>) {
        if !self.owner.draw_calls[handle.bucket].contains_key(&handle.uniform) {
            panic!("Invalid draw call prepared for drop");
        };
        self.dropped_draws.push(handle);
    }

    /// Prepare to update a region.
    /// An update invoklves deallocating old and allocating the new contents.
    /// If the draw call is currently inactive, no data is dropped in the first step.
    pub fn prepare_replace_region(
        &mut self,
        handle: DrawCallHandle<Bucket>,
        vertex_buffer: Buffer,
        instance_count: u32,
    ) {
        if !self.owner.draw_calls[handle.bucket].contains_key(&handle.uniform) {
            panic!("Invalid draw call prepared for replace");
        };
        self.updated_draws.push(DrawCallUpdateArgs {
            draw_call: handle,
            vertex_buffer,
            instance_count,
        });
    }

    /// Submit all prepared updates.
    pub fn submit(self, device: &Device, queue: &Queue, command_encoder: &mut CommandEncoder) {
        if self.new_draws.is_empty()
            && self.dropped_draws.is_empty()
            && self.updated_draws.is_empty()
        {
            return;
        }

        self.owner.submit(
            device,
            queue,
            command_encoder,
            self.new_draws,
            self.dropped_draws,
            self.updated_draws,
        );
    }
}
