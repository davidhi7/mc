use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::iter;
use std::ops::RangeInclusive;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::{
    collections::HashMap,
    thread::{self},
};

use bytemuck::{Pod, Zeroable};
use glam::{IVec2, IVec3, Vec3, ivec2, ivec3};
use itertools::Itertools;
use wgpu::CommandEncoderDescriptor;
use wgpu::{Buffer, Device, Queue};

use crate::math::{self, Aabb2I, Aabb3I};
use crate::renderer::buffers::AsBytes;
use crate::renderer::indirect_buffer_manager::DrawCallHandle;
use crate::{
    renderer::{
        indirect_buffer_manager::{InstanceSize, MultiDrawIndirectBuffer},
        vertex_buffer::{QuadInstance, TransparentQuadInstance},
    },
    world::{
        self, World,
        chunk::{ChunkStack, ChunkUVW, ChunkUW, VERTICAL_CHUNK_COUNT},
    },
};

mod worker;

type InstanceCount = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Zeroable, Pod)]
#[repr(C)]
pub struct ChunkUniform {
    pub u: i32,
    pub v: i32,
    pub w: i32,
    _padding: i32,
}

impl From<ChunkUVW> for ChunkUniform {
    fn from(value: ChunkUVW) -> Self {
        let ChunkUVW { u, v, w } = value;
        Self {
            u,
            v,
            w,
            _padding: 0,
        }
    }
}

impl From<ChunkUniform> for ChunkUVW {
    fn from(value: ChunkUniform) -> Self {
        let ChunkUniform { u, v, w, .. } = value;
        ChunkUVW { u, v, w }
    }
}

impl AsBytes for ChunkUniform {
    fn get_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum TerrainBuckets {
    SOLID,
    TRANSPARENT,
}

impl InstanceSize for TerrainBuckets {
    fn instance_size(&self) -> u64 {
        match *self {
            TerrainBuckets::SOLID => QuadInstance::desc().array_stride,
            TerrainBuckets::TRANSPARENT => TransparentQuadInstance::desc().array_stride,
        }
    }
}

#[derive(Clone, Debug)]
enum ChunkJob {
    Mesh { chunk_stack: Arc<ChunkStack> },
    GenerateAndMesh { uw: ChunkUW },
}

impl ChunkJob {
    fn get_uw(&self) -> ChunkUW {
        match self {
            ChunkJob::Mesh { chunk_stack } => chunk_stack.uw,
            ChunkJob::GenerateAndMesh { uw } => *uw,
        }
    }
}

struct WorkerThreadHandle {
    sender: Sender<ChunkJob>,
    receiver: Receiver<ChunkJobResult>,
    job_count: usize,
}

impl PartialEq for WorkerThreadHandle {
    fn eq(&self, other: &Self) -> bool {
        self.job_count == other.job_count
    }
}

impl Eq for WorkerThreadHandle {}

impl PartialOrd for WorkerThreadHandle {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.job_count.cmp(&other.job_count))
    }
}

impl Ord for WorkerThreadHandle {
    fn cmp(&self, other: &Self) -> Ordering {
        self.job_count.cmp(&other.job_count)
    }
}

struct ChunkJobResult {
    uw: ChunkUW,
    chunk_stack: Option<Arc<ChunkStack>>,
    chunk_buffers: Vec<ChunkBuffers>,
}

pub struct ChunkBuffers {
    pub buffers: HashMap<TerrainBuckets, (Buffer, InstanceCount)>,
}

pub struct WorldLoader {
    pub world: World,
    render_distance: u32,
    worker_pool: Vec<WorkerThreadHandle>,
    tasked_chunk_stacks: HashSet<ChunkUW>,
    buffered_chunks: HashMap<ChunkUW, Vec<ChunkBuffers>>,
    indirect_draw_calls: Vec<DrawCallHandle<ChunkUniform, TerrainBuckets>>,
    last_camera_chunk: Option<ChunkUVW>,
    deferred_chunk_stacks: HashSet<ChunkUW>,
}

impl WorldLoader {
    pub fn new(world: World, thread_count: u32, device: Device, render_distance: u32) -> Self {
        let mut instance = Self {
            world,
            render_distance,
            worker_pool: Vec::new(),
            tasked_chunk_stacks: HashSet::new(),
            buffered_chunks: HashMap::new(),
            indirect_draw_calls: Vec::new(),
            last_camera_chunk: None,
            deferred_chunk_stacks: HashSet::new(),
        };
        for _ in 0..thread_count {
            let (job_sender, job_receiver) = channel();
            let (result_sender, result_receiver) = channel();
            let noise_clone = instance.world.noise.clone();
            let device_clone = device.clone();
            thread::spawn(move || {
                worker::launch(job_receiver, result_sender, device_clone, noise_clone);
            });

            instance.worker_pool.push(WorkerThreadHandle {
                sender: job_sender,
                receiver: result_receiver,
                job_count: 0,
            });
        }

        instance
    }

    pub fn load_chunks(
        &mut self,
        device: &Device,
        queue: &Queue,
        indirect_buffer: &mut MultiDrawIndirectBuffer<ChunkUniform, TerrainBuckets, 2>,
        position: Vec3,
    ) {
        let camera_chunk = world::get_chunk_coordinates_f32(position);

        if let Some(last_camera_chunk) = self.last_camera_chunk {
            self.handle_results();
            self.update_indirect_buffer(
                device,
                queue,
                indirect_buffer,
                camera_chunk,
                last_camera_chunk,
            );
        }

        // If the u/w coordinates are identical, there are no new chunk stacks to generate/mesh
        if self
            .last_camera_chunk
            .is_some_and(|chunk| chunk.to_uw() == camera_chunk.to_uw())
        {
            // Update anyways in case the v coordinate changed
            self.last_camera_chunk = Some(camera_chunk);
            return;
        }

        let chunks = if let Some(old_camera_chunk) = self.last_camera_chunk {
            let aabb = Self::visible_chunk_range_aabb2(camera_chunk.to_uw(), self.render_distance);
            let subtracted_aabb =
                Self::visible_chunk_range_aabb2(old_camera_chunk.to_uw(), self.render_distance);

            math::area_subtract_overlap_2d(aabb, subtracted_aabb)
                .into_iter()
                .flat_map(Self::iterate_aabb_chunks_2d)
                .map(ChunkUW::from)
                .collect::<Vec<_>>()
        } else {
            Self::visible_chunk_range_uw(camera_chunk.to_uw(), self.render_distance)
        };

        let filtered_chunks = chunks
            .into_iter()
            .filter(|chunk| self.buffered_chunks.get(&chunk).is_none())
            .filter(|chunk| !self.tasked_chunk_stacks.contains(&chunk));

        let mut jobs = filtered_chunks
            .map(|uw| match self.world.get_chunk_stack(uw) {
                Some(chunk_stack) => ChunkJob::Mesh { chunk_stack },
                None => ChunkJob::GenerateAndMesh { uw },
            })
            .collect::<Vec<_>>();

        jobs.sort_unstable_by_key(|chunk| {
            (IVec2::from(camera_chunk.to_uw()) - IVec2::from(chunk.get_uw())).length_squared()
        });

        if self.last_camera_chunk.is_none() {
            // If self.last_camera_chunk is none and this Self::load_chunks is called for the first time,
            // then simply store the chunks for the next Self::update_indirect_buffer call and skip the call itself
            self.deferred_chunk_stacks
                .extend(jobs.iter().map(|job| job.get_uw()));
        }

        self.assign_jobs_to_workers(jobs);
        self.last_camera_chunk = Some(camera_chunk);
    }

    fn assign_jobs_to_workers(&mut self, jobs: Vec<ChunkJob>) {
        if jobs.is_empty() {
            return;
        }

        // Priority queue (min-heap) to manage workers by their job count
        let mut worker_heap: BinaryHeap<Reverse<&mut _>> =
            self.worker_pool.iter_mut().map(Reverse).collect();

        for job in jobs {
            // Get the worker with the least job count
            let Reverse(worker) = worker_heap.pop().unwrap();

            // Assign the job to this worker
            self.tasked_chunk_stacks.insert(job.get_uw());
            worker
                .sender
                .send(job)
                .expect("Failed to send job to chunk worker thread");
            worker.job_count += 1;

            // Push the worker back into the heap with updated job count
            worker_heap.push(Reverse(worker));
        }
    }

    fn handle_results(&mut self) {
        for worker in self.worker_pool.iter_mut() {
            loop {
                match worker.receiver.try_recv() {
                    Ok(result) => {
                        if let Some(chunk_stack) = result.chunk_stack {
                            self.world.insert_chunks(result.uw, chunk_stack);
                        }
                        self.buffered_chunks.insert(result.uw, result.chunk_buffers);
                        self.tasked_chunk_stacks.remove(&result.uw);
                    }
                    Err(TryRecvError::Empty) => {
                        break;
                    }
                    Err(TryRecvError::Disconnected) => {
                        panic!("Worker thread disconnected")
                    }
                }
            }
        }
    }

    fn update_indirect_buffer(
        &mut self,
        device: &Device,
        queue: &Queue,
        buf: &mut MultiDrawIndirectBuffer<ChunkUniform, TerrainBuckets, 2>,
        camera_chunk: ChunkUVW,
        old_camera_chunk: ChunkUVW,
    ) {
        if old_camera_chunk == camera_chunk && self.deferred_chunk_stacks.is_empty() {
            return;
        }

        let camera_aabb2 =
            Self::visible_chunk_range_aabb2(camera_chunk.to_uw(), self.render_distance);
        let camera_aabb3 = Self::visible_chunk_range_aabb3(camera_chunk, self.render_distance);
        let old_camera_aabb3 =
            Self::visible_chunk_range_aabb3(old_camera_chunk, self.render_distance);

        let mut new_chunks = math::volume_subtract_overlap_3d(camera_aabb3, old_camera_aabb3)
            .into_iter()
            .flat_map(|aabb| Self::iterate_aabb_chunks_3d(aabb))
            .map(ChunkUVW::from)
            .collect::<Vec<_>>();

        self.deferred_chunk_stacks.retain(|chunk_stack| {
            if self.buffered_chunks.contains_key(chunk_stack) {
                // Always remove chunk stack from list, but only prepare for rendering if actually visible
                if camera_aabb2.contains_point((*chunk_stack).into()) {
                    let v_range =
                        Self::vertical_visible_chunk_range(camera_chunk, self.render_distance)
                            .clone();

                    v_range.map(|v| chunk_stack.to_uvw(v)).for_each(|uvw| {
                        if !new_chunks.contains(&uvw) {
                            new_chunks.push(uvw)
                        }
                    });
                }
                return false;
            }
            true
        });

        let old_chunks_aabb = math::volume_subtract_overlap_3d(old_camera_aabb3, camera_aabb3);
        let mut old_draw_call_handles = Vec::new();
        let mut i: usize = 0;
        while i < self.indirect_draw_calls.len() {
            let handle = &self.indirect_draw_calls[i];
            let chunk = ChunkUVW::from(handle.uniform).into();

            if old_chunks_aabb
                .iter()
                .any(|aabb| aabb.contains_point(chunk))
            {
                old_draw_call_handles.push(self.indirect_draw_calls.remove(i));
            } else {
                i += 1;
            }
        }

        let new_chunks_vertical_groups = new_chunks.into_iter().chunk_by(|chunk| chunk.to_uw());

        let new_chunk_buffers_iterator = new_chunks_vertical_groups
            .into_iter()
            .filter(|(uw, _)| {
                if self.buffered_chunks.contains_key(uw) {
                    true
                } else {
                    self.deferred_chunk_stacks.insert(*uw);
                    false
                }
            })
            .flat_map(|(_, group)| group.into_iter())
            .cartesian_product([TerrainBuckets::SOLID, TerrainBuckets::TRANSPARENT])
            .filter_map(|(chunk, bucket)| {
                let chunk_stack = self.buffered_chunks.get(&chunk.to_uw());

                chunk_stack
                    .and_then(|buffers| buffers.get(chunk.v as usize))
                    .and_then(|buffers| buffers.buffers.get(&bucket))
                    .map(|buffer| (chunk, bucket, &buffer.0, buffer.1))
            });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("indirect buffer update command encoder"),
        });

        for element in new_chunk_buffers_iterator.zip_longest(old_draw_call_handles.into_iter()) {
            match element {
                itertools::EitherOrBoth::Both(
                    (chunk, bucket, buffer, instance_count),
                    old_handle,
                ) => {
                    let new_handle = buf.drop_and_insert_region(
                        queue,
                        &mut encoder,
                        old_handle,
                        bucket,
                        &buffer,
                        instance_count,
                        chunk.into(),
                    );
                    self.indirect_draw_calls.push(new_handle);
                }
                itertools::EitherOrBoth::Left((chunk, bucket, buffer, instance_count)) => {
                    let handle = buf.insert_region(
                        queue,
                        &mut encoder,
                        bucket,
                        &buffer,
                        instance_count,
                        chunk.into(),
                    );
                    self.indirect_draw_calls.push(handle);
                }
                itertools::EitherOrBoth::Right(old_handle) => {
                    buf.drop_region(queue, &mut encoder, old_handle);
                }
            }
        }

        let command_buffer = encoder.finish();
        queue.submit(iter::once(command_buffer));
    }

    fn visible_chunk_range_uw(camera_position: ChunkUW, render_distance: u32) -> Vec<ChunkUW> {
        let ChunkUW { u, w } = camera_position;

        let mut chunks_in_order: Vec<ChunkUW> =
            Vec::with_capacity((render_distance * 2 + 1).pow(2) as usize);

        chunks_in_order.push(ChunkUW { u, w });
        for radius in 1..=render_distance as i32 {
            for x in (-radius)..=radius {
                chunks_in_order.push(ChunkUW {
                    u: u + x,
                    w: w + radius,
                });
                chunks_in_order.push(ChunkUW {
                    u: u + x,
                    w: w - radius,
                });
            }

            for z in -(radius - 1)..radius {
                chunks_in_order.push(ChunkUW {
                    u: u + radius,
                    w: w + z,
                });
                chunks_in_order.push(ChunkUW {
                    u: u - radius,
                    w: w + z,
                });
            }
        }

        chunks_in_order
    }

    fn vertical_visible_chunk_range(
        camera_position: ChunkUVW,
        render_distance: u32,
    ) -> RangeInclusive<i32> {
        let v_min = (camera_position.v - render_distance as i32).max(0);
        let v_max =
            (camera_position.v + render_distance as i32).min(VERTICAL_CHUNK_COUNT as i32 - 1);

        v_min..=v_max
    }

    #[allow(dead_code)]
    fn visible_chunk_range_uvw(camera_position: ChunkUVW, render_distance: u32) -> Vec<ChunkUVW> {
        let vec = Self::visible_chunk_range_uw(camera_position.to_uw(), render_distance)
            .into_iter()
            .cartesian_product(Self::vertical_visible_chunk_range(
                camera_position,
                render_distance,
            ))
            .map(|(uw, v)| uw.to_uvw(v))
            .collect();

        vec
    }

    fn visible_chunk_range_aabb2(position: ChunkUW, render_distance: u32) -> Aabb2I {
        Aabb2I::new(
            ivec2(
                position.u - render_distance as i32,
                position.w - render_distance as i32,
            ),
            ivec2(
                position.u + render_distance as i32,
                position.w + render_distance as i32,
            ),
        )
    }

    fn visible_chunk_range_aabb3(position: ChunkUVW, render_distance: u32) -> Aabb3I {
        Aabb3I {
            min: ivec3(
                position.u - render_distance as i32,
                position.v - render_distance as i32,
                position.w - render_distance as i32,
            ),
            max: ivec3(
                position.u + render_distance as i32,
                position.v + render_distance as i32,
                position.w + render_distance as i32,
            ),
        }
    }

    fn iterate_aabb_chunks_2d(aabb: Aabb2I) -> Vec<IVec2> {
        let extends = aabb.max - aabb.min;
        let capacity = ((extends.x + 1) * (extends.y + 1)) as usize;
        let mut result: Vec<IVec2> = Vec::with_capacity(capacity);
        for x in aabb.min.x..=aabb.max.x {
            for y in aabb.min.y..=aabb.max.y {
                result.push(ivec2(x, y));
            }
        }
        assert!(capacity == result.len());
        result
    }

    fn iterate_aabb_chunks_3d(aabb: Aabb3I) -> Vec<IVec3> {
        let extends = aabb.max - aabb.min;
        let capacity = ((extends.x + 1) * (extends.y + 1) * (extends.z + 1)) as usize;
        let mut result: Vec<IVec3> = Vec::with_capacity(capacity);
        for x in aabb.min.x..=aabb.max.x {
            for y in aabb.min.y..=aabb.max.y {
                for z in aabb.min.z..=aabb.max.z {
                    result.push(ivec3(x, y, z));
                }
            }
        }
        result
    }
}
