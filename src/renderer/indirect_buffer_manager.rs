use core::panic;
use std::array;
use std::collections::HashMap;
use std::hash::Hash;

use std::fmt::Debug;
use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use enum_map::Enum;
use glam::IVec3;
use itertools::Itertools;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Queue};

use crate::logging::ReadableBytes;
use crate::renderer::buffers::block_allocator::{
    BlockAllocator, BlockHandle,
};
use crate::renderer::buffers::pool_allocator::{PoolAllocator, SegmentHandle};
use crate::renderer::buffers::{AllocationError, BufferMemoryTarget};
use crate::renderer::vertex_buffer::{INSTANCE_ALIGNMENT, QuadInstance, TransparentQuadInstance};
use crate::world::chunk::ChunkUVW;

pub const BUCKET_COUNT: usize = 2;

// todo move to different crate?
#[derive(Clone, Copy)]
pub struct OffsetSize {
    pub offset: u64,
    pub size: u64
}

#[derive(Clone, Copy)]
pub struct OffsetCount {
    pub offset: u64,
    pub count: u64,
    pub instance_size: u64
}

#[derive(Clone, Copy)]
enum AllocationRequest {
    Bytes(OffsetSize),
    Count(OffsetCount)
}

impl AllocationRequest {
    fn offset(self) -> u64 {
        match self {
            AllocationRequest::Bytes(offset_size) => offset_size.offset,
            AllocationRequest::Count(offset_count) => offset_count.offset,
        }
    }

    fn size(self) -> u64 {
        match self {
            AllocationRequest::Bytes(offset_size) => offset_size.size,
            AllocationRequest::Count(offset_count) => offset_count.count * offset_count.instance_size,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Enum)]
pub enum TerrainBuckets {
    Solid,
    Transparent,
}

impl TerrainBuckets {
    pub const fn all() -> &'static [Self] {
        &[
            Self::Solid,
            Self::Transparent
        ]
    }

    pub const fn instance_size(self) -> u64 {
        match self {
            TerrainBuckets::Solid => QuadInstance::desc().array_stride,
            TerrainBuckets::Transparent => TransparentQuadInstance::desc().array_stride,
        }
    }
}

/// GPU-side descriptor for a chunk to be drawn. The GPU compute shader reads these
/// and writes `DrawIndirectArgs` into indirect draw buffers for chunks that survive culling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Zeroable, Pod)]
#[repr(C)]
pub struct ChunkDescriptor {
    /// One entry per chunk bucket
    // todo correct size
    pub entries: [ChunkDescriptorEntry; BUCKET_COUNT],
    /// Index into the chunk uniforms buffer
    pub uniform_index: u32,
    pub _padding: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Zeroable, Pod)]
#[repr(C)]
pub struct ChunkDescriptorEntry {
    /// Byte offset into the vertex buffer where this chunk's instances begin.
    pub first_instance: u32,
    /// Number of quad instances in this chunk+bucket.
    pub instance_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Zeroable, Pod)]
#[repr(C)]
pub struct ChunkUniform {
    uvw: IVec3,
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

/// CPU-side handle for a chunk to be drawn.
// TODO: evaluate what traits are needed
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkHandle {
    pub uniform: ChunkUniform,
}

/// CPU-side data for a chunk to be drawn.
// multiple vertex buffer handles?
struct Chunk {
    descriptor_buffer_handle: BlockHandle<ChunkDescriptor>,
    uniform_buffer_handle: BlockHandle<ChunkUniform>,
    vertex_buffer_handles: [Option<SegmentHandle>; BUCKET_COUNT],
}

/// Data for a prepared but not yet written draw call.
struct ChunkCreationArgs {
    uniform: ChunkUniform,
    vertex_buffer: Buffer,
    segments: [Option<OffsetSize>; BUCKET_COUNT],
}

/// Data for a draw call that is prepared to be updated, that is the vertex buffer contents are to be replaced.
struct ChunkUpdateArgs {
    handle: ChunkHandle,
    vertex_buffer: Buffer,
    segments: [Option<OffsetSize>; BUCKET_COUNT],
}

// todo comment
/// Descriptor buffer allocator and metadata for one bucket type.
struct DescriptorAllocator {
    /// Storage buffer with per-chunk [`ChunkDescriptor`] values.
    descriptor_buffer: Buffer,
    /// Storage buffer with per-chunk [`ChunkUniform`] values.
    uniform_buffer: Buffer,
    descriptor_allocator: BlockAllocator<ChunkDescriptor>,
    uniform_allocator: BlockAllocator<ChunkUniform>,
    max_chunk_count: u64,
    descriptor_count: u64
}

impl DescriptorAllocator {
    fn reserve_block(&mut self) -> Result<(BlockHandle<ChunkDescriptor>, BlockHandle<ChunkUniform>), AllocationError> {
        let descriptor_block = self.descriptor_allocator.first_free_block()?;
        let uniform_block = self.uniform_allocator.first_free_block()?;
        // todo sync into one operation?
        if descriptor_block.0 != uniform_block.0 {
            log::warn!("Chunk and uniform buffer block index mismatch!");
        }

        Ok((descriptor_block, uniform_block))
    }

    fn write_block(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        chunk: &Chunk,
        descriptor: &ChunkDescriptor,
        uniform: &ChunkUniform,
    ) -> Result<(), AllocationError> {
        self.descriptor_count += 1;
        // todo allocate or reserveblcok
        self.descriptor_allocator
            .allocate_block(&mut BufferMemoryTarget::new(&self.descriptor_buffer, queue, command_encoder), chunk.descriptor_buffer_handle, descriptor)?;
        self.uniform_allocator
            .allocate_block(&mut BufferMemoryTarget::new(&self.uniform_buffer, queue, command_encoder), chunk.uniform_buffer_handle, uniform)?;
        Ok(())
    }

    fn deallocate(&mut self, chunk: &Chunk) -> Result<(), AllocationError> {
        self.descriptor_count -= 1;
        self.descriptor_allocator.deallocate_block(chunk.descriptor_buffer_handle)?;
        self.uniform_allocator.deallocate_block(chunk.uniform_buffer_handle)?;
        Ok(())
    }

    fn clear(&mut self) {
        self.descriptor_allocator.clear();
        self.uniform_allocator.clear();
    }
}

/// Associated data to one insertion into the vertex buffer.
struct VertexBufferInsertionTask {
    vertex_buffer_resize: Option<u64>,
    vertex_buffer_segment: SegmentHandle,
    source_buffer: Buffer,
}

pub struct IndirectBufferManager {
    /// Label appended to all buffers created by this instance.
    buffer_label: String,
    /// Indirect buffer, rebuilt each frame by the GPU compute shader.
    // todo
    // indirect_buffer: Buffer,
    /// Atomic counter buffer for GPU-driven indirect buffer creation.
    /// One u32 per bucket, cleared before each culling dispatch.
    // todo
    // counter_buffer: Buffer,
    descriptor_allocator: DescriptorAllocator,
    /// Vertex/instance buffer. Contains all draw call geometry data.
    vertex_buffer: Buffer,
    vertex_buffer_allocator: PoolAllocator,
    chunks: HashMap<ChunkHandle, Chunk>,
}

// todo why static?
impl IndirectBufferManager {
    pub fn new(device: &Device, buffer_label: String, chunks_count: u64) -> Self {
        // Start with 1MiB
        let vertex_buffer_size = 1024u64.pow(2);

        // let indirect_buffer = device.create_buffer(&BufferDescriptor {
        //     label: Some(&format!("indirect buffer {buffer_label}")),
        //     size: chunks_per_bucket
        //         * Buckets::LENGTH as u64
        //         * std::mem::size_of::<DrawIndirectArgs>() as u64,
        //     usage: BufferUsages::INDIRECT | BufferUsages::STORAGE,
        //     mapped_at_creation: false,
        // });
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("vertex buffer {buffer_label}")),
            size: vertex_buffer_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let vertex_buffer_allocator = PoolAllocator::new(vertex_buffer_size);
        let chunk_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("descriptor buffer {buffer_label}")),
            size: chunks_count
                * size_of::<ChunkDescriptor>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("uniform buffer {buffer_label}")),
            size: chunks_count * size_of::<ChunkUniform>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chunk_allocator = BlockAllocator::new(chunks_count);
        let uniform_allocator = BlockAllocator::new(chunks_count);

        Self {
            buffer_label,
            vertex_buffer,
            vertex_buffer_allocator,
            descriptor_allocator: DescriptorAllocator { descriptor_buffer: chunk_buffer, uniform_buffer, descriptor_allocator: chunk_allocator, uniform_allocator, max_chunk_count: chunks_count, descriptor_count: 0 },
            chunks: HashMap::default(),
        }
    }

    /// Insert new chunk for drawing.
    fn insert_chunk(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        overwrite_chunk: Option<ChunkHandle>,
        ChunkCreationArgs {
            uniform,
            vertex_buffer,
            segments,
        }: ChunkCreationArgs,
    ) -> [Option<VertexBufferInsertionTask>; BUCKET_COUNT] {
        let insertion_tasks = segments.map(|offset_size| {
            let Some(offset_size) = offset_size else {
                return None;
            };
            Some(self.reserve_from_vertex_buffer(vertex_buffer.clone(), AllocationRequest::Bytes(offset_size), INSTANCE_ALIGNMENT))
        });

        let (descriptor_handle, uniform_handle) = match overwrite_chunk {
            None => self.descriptor_allocator.reserve_block().expect("Descriptor/uniform allocation failed"),
            Some(chunk_handle) => {
                // todo is removing correct here?
                let chunk = self.chunks.remove(&chunk_handle).expect("Invalid chunk handle provided for overwriting");
                (chunk.descriptor_buffer_handle, chunk.uniform_buffer_handle)
            }
        };

        let chunk = Chunk {
            descriptor_buffer_handle: descriptor_handle,
            uniform_buffer_handle: uniform_handle,
            vertex_buffer_handles: insertion_tasks.iter().map(|insertion| insertion.as_ref().map(|some| some.vertex_buffer_segment)).collect::<Vec<_>>().try_into().expect("TODO"),
        };
        let chunk_handle = ChunkHandle {
            uniform,
        };

        let descriptor = Self::construct_descriptor(
            chunk.vertex_buffer_handles,
            &uniform_handle
        );

        self.descriptor_allocator.write_block(queue, command_encoder, &chunk, &descriptor, &uniform).unwrap();

        self.chunks.insert(
            chunk_handle,
            chunk
        );

        insertion_tasks
    }

    /// Drop region.
    ///
    /// This involves:
    /// - Deallocating the vertex buffer contents
    /// - Deallocating the uniform buffer contents if they are not used for another region
    /// - Deallocating the descriptor buffer entry
    ///
    /// If the draw call is followed by other draw calls of the same bucket, the last descriptor is moved into the now free slot to guarantee a continuous sequence of active descriptors.
    fn drop_region(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        handle: ChunkHandle,
    ) {
        let chunk = self.chunks
            .remove(&handle)
            .expect("Attempted to drop invalid chunk");

        for handle in chunk.vertex_buffer_handles {
            if let Some(handle) = handle {
                self.vertex_buffer_allocator.deallocate(handle).expect("Invalid vertex buffer handle associated to dropped chunk");
            }
        }
        self.descriptor_allocator.deallocate(&chunk).expect("Invalid chunk buffer handles associated to dropped chunk");

        // If the draw call doesn't own the last descriptor buffer slot, fill the slot with another active draw call of the same bucket
        if chunk.descriptor_buffer_handle.0
            != self.descriptor_allocator.descriptor_count - 1
        {
            // Perform swap-and-remove
            // Find chunk in the same bucket with highest descriptor buffer slot
            let (last_chunk_key, last_chunk) = self.chunks
                .iter_mut()
                .max_by_key(|(_, data)| data.descriptor_buffer_handle.0)
                .expect("There should be at least one active chunk remaining");

            // Move last descriptor to the now empty slot
            self.descriptor_allocator
                .write_block(
                    queue,
                    command_encoder,
                    &chunk,
                    &Self::construct_descriptor(
                        chunk.vertex_buffer_handles,
                        &chunk.uniform_buffer_handle
                    ),
                    &last_chunk_key.uniform
                )
                .expect("Existing descriptor buffer handle should still be valid");

            last_chunk.descriptor_buffer_handle = chunk.descriptor_buffer_handle;
            last_chunk.uniform_buffer_handle = chunk.uniform_buffer_handle;
        }
    }

    fn reserve_from_vertex_buffer(
        &mut self,
        source_buffer: Buffer,
        allocation: AllocationRequest,
        alignment: u64
    ) -> VertexBufferInsertionTask {
        match self.vertex_buffer_allocator.reserve_segment(
            allocation.size(),
            alignment,
        ) {
            Ok(vertex_buffer_segment) => VertexBufferInsertionTask {
                vertex_buffer_resize: None,
                vertex_buffer_segment,
                source_buffer,
            },
            Err(_) => {
                let old_size = self.vertex_buffer_allocator.size();
                let new_size = u64::max(
                    old_size * 3 / 2,
                    old_size + allocation.size(),
                );
                self.vertex_buffer_allocator.grow(new_size);
                let vertex_buffer_segment = self.vertex_buffer_allocator
                    .reserve_segment(
                        allocation.size(),
                        alignment
                    )
                    .expect("Segment reservation failed even after growing the buffer");
                VertexBufferInsertionTask { vertex_buffer_resize: Some(new_size), vertex_buffer_segment, source_buffer }
            }
        }
    }

    fn replace_region_vertex_data(
        &mut self,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        ChunkUpdateArgs {
            handle,
            vertex_buffer,
            segments,
        }: ChunkUpdateArgs,
    ) -> [Option<VertexBufferInsertionTask>; BUCKET_COUNT] {
        let chunk = self.chunks
            .get_mut(&handle)
            .expect("Invalid draw call provided for replace");

        for old_segment in chunk.vertex_buffer_handles.into_iter().flatten() {
            self.vertex_buffer_allocator
                .deallocate(old_segment)
                .expect("Invalid handle provided");
        }

        drop(chunk);

        let insertion_tasks = segments.map(|offset_count| {
            let Some(offset_size) = offset_count else {
                return None;
            };
            // todo clone needed here?
            Some(self.reserve_from_vertex_buffer(vertex_buffer.clone(), AllocationRequest::Bytes(offset_size), INSTANCE_ALIGNMENT))
        });

        // todo ugly double borrow
        let chunk = self.chunks
            .get_mut(&handle)
            .expect("Invalid draw call provided for replace");

        // todo ugly
        let segments = insertion_tasks.iter().map(|insertion| insertion.as_ref().map(|some| some.vertex_buffer_segment)).collect::<Vec<_>>().try_into().expect("TODO");

        let descriptor = Self::construct_descriptor(segments, &chunk.uniform_buffer_handle);


        self.descriptor_allocator
            .write_block(
                queue,
                command_encoder,
                &chunk,
                &descriptor,
                &handle.uniform
            )
            .expect("Invalid descriptor buffer handle provided");

        chunk.vertex_buffer_handles = segments;

        insertion_tasks
    }

    pub fn create_update_pass(&mut self) -> IndirectBufferUpdatePass<'_> {
        IndirectBufferUpdatePass {
            owner: self,
            new_draws: Vec::new(),
            dropped_draws: Vec::new(),
            updated_draws: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.descriptor_allocator.clear();
        self.vertex_buffer_allocator.clear();
    }

    fn submit(
        &mut self,
        device: &Device,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        new_draws: Vec<ChunkCreationArgs>,
        dropped_draws: Vec<ChunkHandle>,
        updated_draws: Vec<ChunkUpdateArgs>,
    ) {
        let mut vertex_buffer_resize = None;
        let mut vertex_buffer_insertions = Vec::new();

        let mut handle_insertion_result =
            |insertion: VertexBufferInsertionTask| {
                if insertion.vertex_buffer_resize.is_some() {
                    vertex_buffer_resize = insertion.vertex_buffer_resize;
                }
                vertex_buffer_insertions.push((insertion.source_buffer, insertion.vertex_buffer_segment));
            };

        for updated_draw in updated_draws {
            for new_insertion in self.replace_region_vertex_data(
                queue,
                command_encoder,
                updated_draw,
            ).into_iter().flatten() {
                handle_insertion_result(new_insertion);
            }
        }

        for entry in new_draws.into_iter().zip_longest(dropped_draws) {
            match entry {
                itertools::EitherOrBoth::Both(new_chunk, old_handle) => {
                    let chunk = self.chunks
                        .remove(&old_handle)
                        .expect("Invalid or inactive draw call handle provided for drop");

                    for handle in chunk.vertex_buffer_handles.into_iter().flatten() {
                        self.vertex_buffer_allocator
                            .deallocate(handle)
                            .expect("Invalid vertex buffer handle associated to dropped draw call");
                    }

                    for new_insertion in self.insert_chunk(
                        queue,
                        command_encoder,
                        Some(old_handle),
                        new_chunk,
                    ).into_iter().flatten() {
                        handle_insertion_result(new_insertion);
                    };
                }
                itertools::EitherOrBoth::Left(new_args) => {
                    for new_insertion in self.insert_chunk(
                                            queue,
                                            command_encoder,
                                            None,
                                            new_args,
                                        ).into_iter().flatten() {
                        handle_insertion_result(new_insertion);
                    }
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
        &self.descriptor_allocator.uniform_buffer
    }

    // pub fn indirect_buffer(&self) -> &Buffer {
    //     &self.indirect_buffer
    // }

    pub fn descriptor_buffer(&self) -> &Buffer {
        &self.descriptor_allocator.descriptor_buffer
    }

    // pub fn counter_buffer(&self) -> &Buffer {
    //     &self.counter_buffer
    // }

    // /// Offset of the indirect buffer region for the given bucket, in bytes.
    // pub fn indirect_buffer_offset(&self, bucket: Bucket) -> u64 {
    //     self.descriptor_buffer_allocator.chunks_per_bucket
    //         * bucket.into_usize() as u64
    //         * std::mem::size_of::<DrawIndirectArgs>() as u64
    // }

    // /// Counter buffer offset for the given bucket, in bytes.
    // pub fn counter_buffer_offset(&self, bucket: Bucket) -> u64 {
    //     bucket.into_usize() as u64 * std::mem::size_of::<u32>() as u64
    // }

    pub fn descriptor_count(&self) -> u64 {
        self.descriptor_allocator.descriptor_count
    }

    pub fn max_descriptor_count(&self) -> u64 {
        self.descriptor_allocator.descriptor_count
    }

    fn construct_descriptor(
        vb_segments: [Option<SegmentHandle>; BUCKET_COUNT],
        uniform_handle: &BlockHandle<ChunkUniform>
    ) -> ChunkDescriptor {
        let entries  = array::from_fn(|i| {
            let segment = vb_segments[i];
            let instance_size = TerrainBuckets::all()[i].instance_size();
            match segment {
                Some(segment) => ChunkDescriptorEntry {
                    first_instance: (segment.offset / instance_size).try_into().unwrap(),
                    instance_count: (segment.size / instance_size).try_into().unwrap(),
                },
                None => ChunkDescriptorEntry {
                    first_instance: 0,
                    instance_count: 0,
                },
            }
        });
        ChunkDescriptor {
            entries,
            uniform_index: uniform_handle.0.try_into().unwrap(),
            _padding: Default::default()
        }
    }
}

pub struct IndirectBufferUpdatePass<'a> {
    owner: &'a mut IndirectBufferManager,
    new_draws: Vec<ChunkCreationArgs>,
    dropped_draws: Vec<ChunkHandle>,
    updated_draws: Vec<ChunkUpdateArgs>,
}

impl<'a> IndirectBufferUpdatePass<'a> {
    pub fn prepare_insert_region(
        &mut self,
        vertex_buffer: Buffer,
        segments: [Option<OffsetSize>; BUCKET_COUNT],
        uniform: impl Into<ChunkUniform>,
    ) -> ChunkHandle {
        let uniform = uniform.into();
        let handle = ChunkHandle { uniform };
        if self.owner.chunks.contains_key(&handle) {
            log::warn!("Chunk prepared for insertion conflicts with already loaded chunk");
        }

        self.new_draws.push(ChunkCreationArgs {
            uniform,
            vertex_buffer,
            segments
        });

        handle
    }

    pub fn prepare_drop_chunk(&mut self, handle: ChunkHandle) {
        if !self.owner.chunks.contains_key(&handle) {
            log::warn!("Not currently loaded chunk prepared for drop");
        };
        self.dropped_draws.push(handle);
    }

    pub fn prepare_replace_chunk(
        &mut self,
        handle: ChunkHandle,
        vertex_buffer: Buffer,
        segments: [Option<OffsetSize>; BUCKET_COUNT],
    ) {
        if !self.owner.chunks.contains_key(&handle) {
            log::warn!("Invalid draw call prepared for replace");
        };
        self.updated_draws.push(ChunkUpdateArgs {
            handle,
            vertex_buffer,
            segments,
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
