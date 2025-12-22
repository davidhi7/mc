use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, channel};
use std::thread::{self};

use glam::{IVec2, IVec3, Vec3, Vec3Swizzles};
use itertools::Itertools;
use wgpu::{Buffer, CommandEncoder, Device, Queue};

use crate::renderer::indirect_buffer_manager::{
    DrawCallBucket, DrawCallHandle, IndirectBufferUpdatePass,
};
use crate::world::chunk::Chunk;
use crate::world::world_loader::rolling_grid::RollingGrid;
use crate::{
    renderer::{
        indirect_buffer_manager::IndirectBufferManager,
        vertex_buffer::{QuadInstance, TransparentQuadInstance},
    },
    world::{
        self, World,
        chunk::{ChunkStack, ChunkUVW, ChunkUW, VERTICAL_CHUNK_COUNT},
    },
};

use enum_map::{Enum, EnumMap};

mod rolling_grid;
mod worker;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Enum)]
pub enum TerrainType {
    Solid,
    Transparent,
}

impl DrawCallBucket for TerrainType {
    fn instance_size(self) -> u64 {
        match self {
            TerrainType::Solid => QuadInstance::desc().array_stride,
            TerrainType::Transparent => TransparentQuadInstance::desc().array_stride,
        }
    }
}

struct DrawnChunkState {
    #[expect(dead_code)]
    buffers: EnumMap<TerrainType, Option<Buffer>>,
    draw_calls: EnumMap<TerrainType, Option<DrawCallHandle<TerrainType>>>,
}

enum ChunkState {
    BufferingInProcess,
    BufferedAndDrawn(DrawnChunkState),
    OutOfBounds,
}

struct JobCounter(u64);

impl JobCounter {
    fn new() -> Self {
        JobCounter(0)
    }

    fn next(&mut self) -> u64 {
        let old = self.0;
        self.0 += 1;
        old
    }
}

enum ChunkJobType {
    Mesh { chunk: Arc<Chunk> },
    GenerateAndMeshStack { uw: ChunkUW },
}

struct ChunkJob {
    id: u64,
    job: ChunkJobType,
}

enum ChunkJobResultType {
    Mesh {
        chunk: Arc<Chunk>,
        buffers: EnumMap<TerrainType, Option<Buffer>>,
    },
    GenerateAndMeshStack {
        chunk_stack: ChunkStack,
        buffers: Box<[EnumMap<TerrainType, Option<Buffer>>; VERTICAL_CHUNK_COUNT]>,
    },
}

struct ChunkJobResult {
    id: u64,
    result: ChunkJobResultType,
}

struct WorkerThreadHandle {
    sender: Sender<ChunkJob>,
    job_count: usize,
}

impl PartialEq for WorkerThreadHandle {
    fn eq(&self, other: &Self) -> bool {
        self.job_count == other.job_count
    }
}

impl Eq for WorkerThreadHandle {}

impl PartialOrd for WorkerThreadHandle {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkerThreadHandle {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.job_count.cmp(&other.job_count)
    }
}

pub struct WorldLoader {
    pub world: World,
    job_counter: JobCounter,
    job_id_cutoff: Arc<AtomicU64>,
    worker_pool: Box<[WorkerThreadHandle]>,
    worker_recv: Receiver<ChunkJobResult>,
    grid: RollingGrid<ChunkState>,
    ongoing_chunk_generation: HashSet<ChunkUW>,
    ongoing_chunk_meshing: HashSet<ChunkUVW>,
}

impl WorldLoader {
    pub fn new(
        world: World,
        position: Vec3,
        thread_count: u32,
        device: Device,
        render_distance: u32,
    ) -> Self {
        let mut job_counter = JobCounter::new();
        let job_id_cutoff = Arc::new(AtomicU64::new(job_counter.next()));

        let mut worker_pool = Vec::new();
        let (worker_send, worker_recv) = mpsc::channel();
        for _ in 0..thread_count {
            let (sender, receiver) = channel();
            thread::spawn({
                let noise = world.noise;
                let device = device.clone();
                let sender = worker_send.clone();
                let job_cutoff_id = Arc::clone(&job_id_cutoff);
                move || {
                    worker::launch(receiver, sender.clone(), job_cutoff_id, device, noise);
                }
            });

            worker_pool.push(WorkerThreadHandle {
                sender,
                job_count: 0,
            });
        }
        let worker_pool = worker_pool.into_boxed_slice();

        let mut jobs = Vec::new();
        let mut ongoing_chunk_generation = HashSet::new();
        let mut ongoing_chunk_meshing = HashSet::new();
        let grid = RollingGrid::new(
            render_distance as usize * 2 + 1,
            world::get_chunk_coordinates(position.as_ivec3()).into(),
            update_rolling_grid(
                &world,
                &mut ongoing_chunk_meshing,
                &mut ongoing_chunk_generation,
                &mut jobs,
                &mut job_counter,
            ),
        );

        let mut instance = Self {
            world,
            job_counter,
            job_id_cutoff,
            worker_pool,
            worker_recv,
            grid,
            ongoing_chunk_generation,
            ongoing_chunk_meshing,
        };

        instance.distribute_jobs(jobs);
        instance
    }

    /// Load and unload all chunks that moved into and outside the render distance, respectively.
    /// Additionally, explicitly provided chunks from e.g. block updates are reloaded.
    pub fn load_chunks(
        &mut self,
        device: &Device,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        indirect_buffer: &mut IndirectBufferManager<TerrainType>,
        new_position: Vec3,
        updated_chunks: Option<Vec<ChunkUVW>>,
    ) {
        let mut update_pass = indirect_buffer.create_update_pass();

        let player_chunk = world::get_chunk_coordinates(new_position.as_ivec3());
        self.update_grid(player_chunk, &mut update_pass);
        self.complete_finished_jobs(&mut update_pass);

        if let Some(updated_chunks) = updated_chunks {
            for uvw in updated_chunks {
                if !self.grid.contains(uvw.into()) {
                    continue;
                }

                self.reload_chunk(device, &mut update_pass, uvw);
            }
        }

        update_pass.submit(device, queue, command_encoder);
    }

    /// Relocate the grid, dispatch jobs for chunks that moved into the render distance and drop old chunks.
    fn update_grid(
        &mut self,
        new_center: ChunkUVW,
        update_pass: &mut IndirectBufferUpdatePass<TerrainType>,
    ) {
        let mut jobs = Vec::new();
        self.grid.reposition(
            new_center.into(),
            update_rolling_grid(
                &self.world,
                &mut self.ongoing_chunk_meshing,
                &mut self.ongoing_chunk_generation,
                &mut jobs,
                &mut self.job_counter,
            ),
            |_, state| {
                let ChunkState::BufferedAndDrawn(state) = state else {
                    return;
                };

                for (_, draw_call) in state.draw_calls.into_iter() {
                    if let Some(draw_call) = draw_call {
                        update_pass.prepare_drop_region(draw_call);
                    }
                }
            },
        );

        self.distribute_jobs(jobs);
    }

    /// Handle all results from finished jobs.
    /// 1. Inserts newly generated chunk stacks
    /// 2. Create draw calls for chunks within the render distance.
    fn complete_finished_jobs(&mut self, update_pass: &mut IndirectBufferUpdatePass<TerrainType>) {
        fn create_draw_calls(
            update_pass: &mut IndirectBufferUpdatePass<TerrainType>,
            uvw: ChunkUVW,
            buffers: &EnumMap<TerrainType, Option<Buffer>>,
        ) -> EnumMap<TerrainType, Option<DrawCallHandle<TerrainType>>> {
            let mut draw_calls = EnumMap::default();

            for (terrain_type, buffer) in buffers.iter() {
                let Some(buffer) = buffer else {
                    continue;
                };

                let draw_call = update_pass.prepare_insert_region(
                    terrain_type,
                    buffer.clone(),
                    (buffer.size() / terrain_type.instance_size())
                        .try_into()
                        .unwrap(),
                    uvw,
                );

                draw_calls[terrain_type] = Some(draw_call);
            }

            draw_calls
        }

        for ChunkJobResult { id, result } in self.worker_recv.try_iter() {
            if id <= self.job_id_cutoff.load(Ordering::Relaxed) {
                continue;
            }
            match result {
                ChunkJobResultType::Mesh { chunk, buffers } => {
                    let uvw = chunk.uvw();
                    self.ongoing_chunk_meshing.remove(&uvw);

                    let Some(state) = self.grid.at_mut(uvw.into()) else {
                        // Chunk is no longer in render distance
                        continue;
                    };

                    *state = ChunkState::BufferedAndDrawn(DrawnChunkState {
                        draw_calls: create_draw_calls(update_pass, uvw, &buffers),
                        buffers,
                    });
                }
                ChunkJobResultType::GenerateAndMeshStack {
                    chunk_stack,
                    buffers,
                } => {
                    let uw = chunk_stack.uw;
                    self.ongoing_chunk_generation.remove(&uw);

                    self.world.insert_chunk_stack(chunk_stack);

                    for (v, buffers) in buffers.into_iter().enumerate() {
                        let uvw = uw.to_uvw(v as i32);

                        let Some(state) = self.grid.at_mut(uvw.into()) else {
                            // Chunk is no longer in render distance
                            continue;
                        };

                        *state = ChunkState::BufferedAndDrawn(DrawnChunkState {
                            draw_calls: create_draw_calls(update_pass, uvw, &buffers),
                            buffers,
                        });
                    }
                }
            }
        }
    }

    /// Recreate mesh and update draw calls for the chunk at the given coordinates.
    fn reload_chunk(
        &mut self,
        device: &Device,
        update_pass: &mut IndirectBufferUpdatePass<TerrainType>,
        uvw: ChunkUVW,
    ) {
        let chunk_state = self
            .grid
            .replace(uvw.into(), ChunkState::BufferingInProcess)
            .expect("Chunk isn't within render distance");

        let old_draw_calls =
            if let ChunkState::BufferedAndDrawn(DrawnChunkState { draw_calls, .. }) = chunk_state {
                draw_calls
            } else {
                EnumMap::default()
            };

        let buffers = worker::create_mesh(
            device,
            &self
                .world
                .get_chunk(uvw)
                .expect("Chunk hasn't been generated yet"),
        );

        let mut new_draw_calls = EnumMap::default();
        for ((terrain_type, old_draw_call), (_, buffer)) in
            old_draw_calls.into_iter().zip_eq(buffers.iter())
        {
            match (old_draw_call, buffer) {
                (None, Some(buffer)) => {
                    let draw_call = update_pass.prepare_insert_region(
                        terrain_type,
                        buffer.to_owned(),
                        (buffer.size() / terrain_type.instance_size())
                            .try_into()
                            .unwrap(),
                        uvw,
                    );
                    new_draw_calls[terrain_type] = Some(draw_call);
                }
                (Some(old_draw_call), None) => {
                    update_pass.prepare_drop_region(old_draw_call);
                }
                (Some(old_draw_call), Some(new_buffer)) => {
                    update_pass.prepare_replace_region(
                        old_draw_call,
                        new_buffer.to_owned(),
                        (new_buffer.size() / terrain_type.instance_size())
                            .try_into()
                            .unwrap(),
                    );
                    new_draw_calls[terrain_type] = Some(old_draw_call);
                }
                (None, None) => (),
            }
        }

        *self
            .grid
            .at_mut(uvw.into())
            .expect("Chunk is not within render distance") =
            ChunkState::BufferedAndDrawn(DrawnChunkState {
                buffers,
                draw_calls: new_draw_calls,
            });
    }

    /// Distribute jobs to threads in the thread pool.
    /// The jobs are ordered by the horizontal distance to the chunk the player is in.
    fn distribute_jobs(&mut self, mut jobs: Vec<ChunkJob>) {
        if jobs.is_empty() {
            return;
        }

        // Sort by distance between job chunk uw and center chunk uw
        jobs.sort_unstable_by_key(|job| {
            let uw = match &job.job {
                ChunkJobType::Mesh { chunk } => chunk.uvw().to_uw(),
                ChunkJobType::GenerateAndMeshStack { uw } => *uw,
            };

            (IVec2::from(uw) - self.grid.center().xz()).length_squared()
        });

        // Priority queue (min-heap) to manage workers by their job count
        let mut worker_heap: BinaryHeap<Reverse<&mut _>> =
            self.worker_pool.iter_mut().map(Reverse).collect();

        for job in jobs {
            // Get the worker with the least job count
            let Reverse(worker) = worker_heap.pop().unwrap();

            // Assign the job to this worker
            worker
                .sender
                .send(job)
                .expect("Failed to send job to chunk worker thread");
            worker.job_count += 1;

            // Push the worker back into the heap with updated job count
            worker_heap.push(Reverse(worker));
        }
    }

    pub fn reload_world(&mut self, indirect_buffer: &mut IndirectBufferManager<TerrainType>) {
        self.world.clear();
        indirect_buffer.clear();

        self.job_id_cutoff.store(
            self.job_counter.next(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.ongoing_chunk_generation.clear();
        self.ongoing_chunk_meshing.clear();

        let mut jobs = Vec::new();
        self.grid.reset(update_rolling_grid(
            &self.world,
            &mut self.ongoing_chunk_meshing,
            &mut self.ongoing_chunk_generation,
            &mut jobs,
            &mut self.job_counter,
        ));
        self.distribute_jobs(jobs);
    }
}

/// Return closure that manages jobs and bookkeeping during grid creation and reposition.
fn update_rolling_grid(
    world: &World,
    ongoing_chunk_meshing: &mut HashSet<ChunkUVW>,
    ongoing_chunk_generation: &mut HashSet<ChunkUW>,
    job_destination: &mut Vec<ChunkJob>,
    job_counter: &mut JobCounter,
) -> impl FnMut(IVec3) -> ChunkState {
    |vec| {
        let uvw = ChunkUVW::from(vec);
        if !ChunkStack::validate_chunk_v(uvw.v) {
            return ChunkState::OutOfBounds;
        }

        match world.get_chunk(uvw) {
            Some(chunk) => {
                if ongoing_chunk_meshing.insert(uvw) {
                    job_destination.push(ChunkJob {
                        id: job_counter.next(),
                        job: ChunkJobType::Mesh { chunk },
                    });
                }
            }
            None => {
                let uw = ChunkUVW::from(vec).to_uw();
                if ongoing_chunk_generation.insert(uw) {
                    job_destination.push(ChunkJob {
                        id: job_counter.next(),
                        job: ChunkJobType::GenerateAndMeshStack { uw },
                    });
                }
            }
        }

        ChunkState::BufferingInProcess
    }
}

#[cfg(test)]
mod tests {
    use enum_map::Enum;

    use crate::world::world_loader::TerrainType;

    #[test]
    fn test_draw_call_bucket_get_usize() {
        assert_eq!(TerrainType::Solid.into_usize(), 0);
        assert_eq!(TerrainType::Transparent.into_usize(), 1);
    }
}
