use core::panic;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::sync::mpsc::{self, Receiver, Sender, channel};
use std::sync::{Arc, RwLock};
use std::thread::{self};

use bytemuck::{Pod, Zeroable};
use glam::{IVec2, IVec3, Vec3, Vec3Swizzles};
use wgpu::{Buffer, CommandEncoder, Device, Queue};

use crate::renderer::buffers::AsBytes;
use crate::renderer::indirect_buffer_manager::{DrawCallHandle, IndirectBufferUpdatePass};
use crate::world::chunk::Chunk;
use crate::world::world_loader::rolling_grid::RollingGrid;
use crate::{
    renderer::{
        indirect_buffer_manager::{IndirectBufferManager, InstanceSize},
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Zeroable, Pod)]
#[repr(C)]
pub struct ChunkUniform {
    uvw: IVec3,
    // padding is used for temporary state in the compute shaders but not meant to be read by the CPU
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

impl From<ChunkUniform> for ChunkUVW {
    fn from(value: ChunkUniform) -> Self {
        value.uvw.into()
    }
}

impl AsBytes for ChunkUniform {
    fn get_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Enum)]
pub enum TerrainType {
    SOLID,
    TRANSPARENT,
}

impl InstanceSize for TerrainType {
    fn instance_size(self) -> u64 {
        match self {
            TerrainType::SOLID => QuadInstance::desc().array_stride,
            TerrainType::TRANSPARENT => TransparentQuadInstance::desc().array_stride,
        }
    }
}

struct DrawnChunkState {
    buffers: EnumMap<TerrainType, Option<Buffer>>,
    draw_calls: EnumMap<TerrainType, Option<DrawCallHandle<ChunkUniform, TerrainType>>>,
}

enum ChunkState {
    BufferingInProcess,
    BufferedAndDrawn(DrawnChunkState),
    OutOfBounds,
}

enum ChunkJob {
    Mesh { chunk: Arc<RwLock<Chunk>> },
    GenerateAndMeshStack { uw: ChunkUW },
}

enum ChunkJobResult {
    Mesh {
        uvw: ChunkUVW,
        buffers: EnumMap<TerrainType, Option<Buffer>>,
    },
    GenerateAndMeshStack {
        chunk_stack: ChunkStack,
        buffers: [EnumMap<TerrainType, Option<Buffer>>; VERTICAL_CHUNK_COUNT],
    },
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
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.job_count.cmp(&other.job_count))
    }
}

impl Ord for WorkerThreadHandle {
    fn cmp(&self, other: &Self) -> Ordering {
        self.job_count.cmp(&other.job_count)
    }
}

pub struct WorldLoader {
    pub world: World,
    worker_pool: Vec<WorkerThreadHandle>,
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
        // TODO cleanup duplicated code
        // TODO sorting does not work?
        // TODO fix panic due to duplicated chunk stack generation
        let mut jobs = Vec::new();
        let mut ongoing_chunk_generation = HashSet::new();
        let mut ongoing_chunk_meshing = HashSet::new();
        let grid: RollingGrid<ChunkState> = RollingGrid::new(
            render_distance as usize * 2 + 1,
            world::get_chunk_coordinates(position.as_ivec3()).into(),
            Self::update_rolling_grid(
                &world,
                &mut ongoing_chunk_meshing,
                &mut ongoing_chunk_generation,
                &mut jobs,
            ),
        );

        let mut worker_pool = Vec::new();
        let (result_sender, result_receiver) = mpsc::channel();
        for _ in 0..thread_count {
            let (job_sender, job_receiver) = channel();
            thread::spawn({
                let noise = world.noise.clone();
                let device = device.clone();
                let sender = result_sender.clone();
                move || {
                    worker::launch(job_receiver, sender.clone(), device, noise);
                }
            });

            worker_pool.push(WorkerThreadHandle {
                sender: job_sender,
                job_count: 0,
            });
        }

        let mut instance = Self {
            world,
            worker_pool,
            worker_recv: result_receiver,
            ongoing_chunk_generation,
            ongoing_chunk_meshing,
            grid,
        };

        jobs.sort_unstable_by_key(|job| {
            let uw = match job {
                // TODO optimize
                ChunkJob::Mesh { chunk } => chunk.read().unwrap().uvw().to_uw(),
                ChunkJob::GenerateAndMeshStack { uw } => *uw,
            };

            // TODO does nothing?
            (IVec2::from(uw) - position.xz().as_ivec2()).length_squared()
        });

        instance.assign_jobs_to_workers(jobs);

        instance
    }

    pub fn load_chunks(
        &mut self,
        device: &Device,
        queue: &Queue,
        command_encoder: &mut CommandEncoder,
        indirect_buffer: &mut IndirectBufferManager<ChunkUniform, TerrainType>,
        new_center: Vec3,
        updated_chunks: Option<Vec<ChunkUVW>>,
    ) {
        let mut update_pass = IndirectBufferUpdatePass::new();
        self.update_grid(new_center, &mut update_pass);
        self.complete_finished_jobs(&mut update_pass);

        if let Some(updated_chunks) = updated_chunks {
            for uvw in updated_chunks {
                self.reload_chunk(device, &mut update_pass, uvw);
            }
        }

        indirect_buffer.submit(queue, command_encoder, update_pass);
    }

    fn update_rolling_grid<'a>(
        world: &World,
        ongoing_chunk_meshing: &mut HashSet<ChunkUVW>,
        ongoing_chunk_generation: &mut HashSet<ChunkUW>,
        job_destination: &mut Vec<ChunkJob>,
    ) -> impl FnMut(IVec3) -> ChunkState {
        |vec| {
            let uvw = ChunkUVW::from(vec);
            if !(0..VERTICAL_CHUNK_COUNT as i32).contains(&uvw.v) {
                return ChunkState::OutOfBounds;
            }
            match world.get_chunk_stack(uvw.to_uw()) {
                Some(chunk_stack) => {
                    if ongoing_chunk_meshing.insert(uvw) {
                        // println!("mesh chunk {:?}", vec);
                        job_destination.push(ChunkJob::Mesh {
                            chunk: chunk_stack.chunks[usize::try_from(uvw.v).unwrap()].clone(),
                        });
                    }
                }
                None => {
                    let uw = ChunkUVW::from(vec).to_uw();
                    if ongoing_chunk_generation.insert(uw) {
                        // println!("generate stack {:?}", uw);
                        job_destination.push(ChunkJob::GenerateAndMeshStack { uw });
                    }
                }
            };
            ChunkState::BufferingInProcess
        }
    }

    fn update_grid(
        &mut self,
        new_center: Vec3,
        update_pass: &mut IndirectBufferUpdatePass<ChunkUniform, TerrainType>,
    ) {
        let mut jobs = Vec::new();
        self.grid.reposition(
            world::get_chunk_coordinates(new_center.as_ivec3()).into(),
            Self::update_rolling_grid(
                &self.world,
                &mut self.ongoing_chunk_meshing,
                &mut self.ongoing_chunk_generation,
                &mut jobs,
            ),
            |_, state| {
                let ChunkState::BufferedAndDrawn(state) = state else {
                    return;
                };

                for (_, draw_call) in state.draw_calls {
                    if let Some(draw_call) = draw_call {
                        update_pass.prepare_drop_region(draw_call);
                    }
                }
            },
        );

        jobs.sort_unstable_by_key(|job| {
            let uw = match job {
                // TODO optimize
                ChunkJob::Mesh { chunk } => chunk.read().unwrap().uvw().to_uw(),
                ChunkJob::GenerateAndMeshStack { uw } => *uw,
            };

            (IVec2::from(uw) - new_center.xz().as_ivec2()).length_squared()
        });
        self.assign_jobs_to_workers(jobs);
    }

    fn complete_finished_jobs(
        &mut self,
        update_pass: &mut IndirectBufferUpdatePass<ChunkUniform, TerrainType>,
    ) {
        for job_result in self.worker_recv.try_iter() {
            match job_result {
                ChunkJobResult::Mesh { uvw, buffers } => {
                    self.ongoing_chunk_meshing.remove(&uvw);

                    let Some(state) = self.grid.at_mut(uvw.into()) else {
                        // Chunk is no longer in render distance
                        continue;
                    };

                    let mut draw_calls = EnumMap::default();

                    for (terrain_type, buffer) in buffers.iter() {
                        let Some(buffer) = buffer else {
                            continue;
                        };

                        draw_calls[terrain_type] = Some(
                            update_pass.prepare_insert_region(
                                terrain_type,
                                buffer.clone(),
                                (buffer.size() / terrain_type.instance_size())
                                    .try_into()
                                    .unwrap(),
                                uvw.into(),
                            ),
                        );
                    }

                    *state = ChunkState::BufferedAndDrawn(DrawnChunkState {
                        buffers,
                        draw_calls,
                    })
                }
                ChunkJobResult::GenerateAndMeshStack {
                    chunk_stack,
                    buffers,
                } => {
                    let uw = chunk_stack.uw;

                    self.world.insert_chunk_stack(chunk_stack);
                    self.ongoing_chunk_generation.remove(&uw);

                    for (v, buffers) in buffers.into_iter().enumerate() {
                        let uvw = uw.to_uvw(v as i32);
                        // TODO dedup
                        let Some(state) = self.grid.at_mut(uvw.into()) else {
                            // Chunk is no longer in render distance
                            continue;
                        };

                        let mut draw_calls = EnumMap::default();
                        for (terrain_type, buffer) in buffers.iter() {
                            let Some(buffer) = buffer else {
                                continue;
                            };

                            draw_calls[terrain_type] = Some(
                                update_pass.prepare_insert_region(
                                    terrain_type,
                                    buffer.clone(),
                                    (buffer.size() / terrain_type.instance_size())
                                        .try_into()
                                        .unwrap(),
                                    uvw.into(),
                                ),
                            );
                        }

                        *state = ChunkState::BufferedAndDrawn(DrawnChunkState {
                            buffers,
                            draw_calls,
                        })
                    }
                }
            }
        }
    }

    fn reload_chunk(
        &mut self,
        device: &Device,
        update_pass: &mut IndirectBufferUpdatePass<ChunkUniform, TerrainType>,
        uvw: ChunkUVW,
    ) {
        let chunk_state = self
            .grid
            .replace(uvw.into(), ChunkState::BufferingInProcess)
            .expect("Chunk isn't within render distance");

        if let ChunkState::BufferedAndDrawn(DrawnChunkState { draw_calls, .. }) = chunk_state {
            for (_, draw_call) in draw_calls {
                if let Some(draw_call) = draw_call {
                    update_pass.prepare_drop_region(draw_call);
                }
            }
        }

        let binding = self
            .world
            .get_chunk(uvw)
            .expect("Chunk hasn't been generated yet");

        let buffers = worker::create_mesh(device, &binding.read().unwrap());

        let mut draw_calls = EnumMap::default();
        for (terrain_type, buffer) in buffers.iter() {
            let Some(buffer) = buffer else {
                continue;
            };

            draw_calls[terrain_type] = Some(
                // TODO schedule replace with same uniform
                update_pass.prepare_insert_region(
                    terrain_type,
                    buffer.clone(),
                    (buffer.size() / terrain_type.instance_size())
                        .try_into()
                        .unwrap(),
                    uvw.into(),
                ),
            );
        }

        *self
            .grid
            .at_mut(uvw.into())
            .expect("Chunk is not within render distance") =
            ChunkState::BufferedAndDrawn(DrawnChunkState {
                buffers,
                draw_calls,
            });
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
            // todo
            // self.tasked_chunk_stacks.insert(job.get_uw());
            worker
                .sender
                .send(job)
                .expect("Failed to send job to chunk worker thread");
            worker.job_count += 1;

            // Push the worker back into the heap with updated job count
            worker_heap.push(Reverse(worker));
        }
    }
}
