use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use glam::{IVec2, IVec3, Vec3, Vec3Swizzles};
use itertools::Itertools;
use smallvec::SmallVec;
use thiserror::Error;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferUsages, CommandEncoder, Device, Queue};

use crate::renderer::indirect_buffer_manager::{
    DrawCallBucket, DrawCallHandle, IndirectBufferUpdatePass,
};
use crate::world::blocks::Block;
use crate::world::chunk::ChunkMeshingContext;
use crate::world::world_gen::{ChunkGenResult, WorldGenSettings};
use crate::world::world_loader::executor::{Executor, Job, ThreadPoolExecutor};
use crate::world::world_loader::rolling_grid::RollingGrid;
use crate::{
    renderer::{
        indirect_buffer_manager::IndirectBufferManager,
        vertex_buffer::{QuadInstance, TransparentQuadInstance},
    },
    world::{
        self, World,
        chunk::{ChunkStack, ChunkUVW, ChunkUW},
    },
};

use enum_map::{Enum, EnumMap};

mod executor;
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
    Mesh {
        chunk_context: ChunkMeshingContext,
        uvw: ChunkUVW,
    },
    Generate {
        // chunk_stack: Box<ChunkStack>,
        uw: ChunkUW,
    },
}

struct ChunkJob {
    job_id: u64,
    job: ChunkJobType,
}

enum ChunkJobResultType {
    Mesh {
        uvw: ChunkUVW,
        buffers: EnumMap<TerrainType, Option<Box<[u8]>>>,
    },
    Generate {
        chunk_gen: ChunkGenResult,
    },
    Cancelled,
}

struct ChunkJobResult {
    job_id: u64,
    result: ChunkJobResultType,
}

struct ExecutorContext {
    world_gen_settings: WorldGenSettings,
    job_id_cutoff: AtomicU64,
}

struct ChunkGridContext {
    ongoing_chunk_generation: HashSet<ChunkUW>,
    ongoing_chunk_meshing: HashSet<ChunkUVW>,
    job_buffer: Vec<ChunkJob>,
    job_counter: JobCounter,
}

pub struct WorldLoader {
    world: World,
    grid: RollingGrid<ChunkState, ChunkUVW>,
    grid_ctx: ChunkGridContext,
    executor: ThreadPoolExecutor<ChunkJobResult>,
    executor_ctx: Arc<ExecutorContext>,
}

impl WorldLoader {
    pub fn new(world: World, position: Vec3, render_distance: u32) -> Self {
        let mut job_counter = JobCounter::new();

        let executor_ctx = Arc::new(ExecutorContext {
            world_gen_settings: load_worldgen_settings().expect("Failed to load worldgen settings"),
            job_id_cutoff: AtomicU64::new(job_counter.next()),
        });
        let executor = ThreadPoolExecutor::new();

        let mut grid_ctx = ChunkGridContext {
            ongoing_chunk_generation: HashSet::new(),
            ongoing_chunk_meshing: HashSet::new(),
            job_buffer: Vec::new(),
            job_counter,
        };
        let grid = RollingGrid::new(
            render_distance as usize * 2 + 1,
            world::divide_world_coordinates(position.as_ivec3()).0,
            &mut grid_ctx,
            update_rolling_grid(&world),
        );

        let mut instance = Self {
            world,
            grid,
            grid_ctx,
            executor,
            executor_ctx,
        };

        instance.execute_jobs();
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
        replaced_blocks: SmallVec<[(IVec3, Block); 1]>,
    ) {
        let mut update_pass = indirect_buffer.create_update_pass();

        let player_chunk = world::divide_world_coordinates(new_position.as_ivec3()).0;
        self.update_grid(player_chunk, &mut update_pass);
        let chunks_for_meshing = self.complete_finished_jobs(device.clone(), &mut update_pass);

        let range = self.grid.bounds();
        for uw in chunks_for_meshing {
            let ctx = ChunkMeshingContext::create(&self.world, uw);
            let (v_min, v_max) = (range.0.v, range.1.v);
            for v in v_min..=v_max {
                if !ChunkStack::validate_chunk_v(v) {
                    continue;
                }

                let uvw = uw.to_uvw(v);
                self.grid_ctx.job_buffer.push(ChunkJob {
                    job_id: self.grid_ctx.job_counter.next(),
                    job: ChunkJobType::Mesh {
                        chunk_context: ctx.clone(),
                        uvw,
                    },
                });
            }
        }

        let updated_uvw: SmallVec<[ChunkUVW; 1]> = replaced_blocks
            .into_iter()
            .flat_map(|(pos, block)| self.world.replace_block(pos, block))
            .collect();
        for uvw in updated_uvw {
            if !self.grid.contains(uvw) {
                continue;
            }

            self.reload_chunk(device, &mut update_pass, uvw);
        }

        self.execute_jobs();
        update_pass.submit(device, queue, command_encoder);
    }

    /// Relocate the grid, dispatch jobs for chunks that moved into the render distance and drop old chunks.
    fn update_grid(
        &mut self,
        new_center: ChunkUVW,
        update_pass: &mut IndirectBufferUpdatePass<TerrainType>,
    ) {
        self.grid.reposition(
            new_center.into(),
            &mut self.grid_ctx,
            update_rolling_grid(&self.world),
            |_ctx, _uvw, state| {
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
    }

    /// Handle all results from finished jobs. This returns all now-completed chunks.
    fn complete_finished_jobs(
        &mut self,
        device: Device,
        update_pass: &mut IndirectBufferUpdatePass<TerrainType>,
    ) -> Vec<ChunkUW> {
        let mut chunks_for_meshing = Vec::new();
        for ChunkJobResult { job_id, result } in
            Executor::<_, ExecutorContext>::fetch(&mut self.executor)
        {
            if job_id <= self.executor_ctx.job_id_cutoff.load(Ordering::Relaxed) {
                continue;
            }

            match result {
                ChunkJobResultType::Mesh { uvw, buffers } => {
                    self.grid_ctx.ongoing_chunk_meshing.remove(&uvw);
                    let actual_buffers = EnumMap::default();

                    let Some(state) = self.grid.at_mut(uvw) else {
                        // Chunk is no longer in render distance
                        continue;
                    };

                    let mut draw_calls = EnumMap::default();

                    for (terrain_type, buffer) in buffers.iter() {
                        let Some(buffer) = buffer else {
                            continue;
                        };

                        let actual_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("test"),
                                contents: buffer,
                                usage: BufferUsages::COPY_SRC,
                            });

                        let draw_call = update_pass.prepare_insert_region(
                            terrain_type,
                            actual_buffer.clone(),
                            (actual_buffer.size() / terrain_type.instance_size())
                                .try_into()
                                .unwrap(),
                            uvw,
                        );

                        draw_calls[terrain_type] = Some(draw_call);
                    }

                    *state = ChunkState::BufferedAndDrawn(DrawnChunkState {
                        draw_calls,
                        buffers: actual_buffers,
                    });
                }
                ChunkJobResultType::Generate { chunk_gen } => {
                    let uw = chunk_gen.chunk_stack.uw();
                    self.grid_ctx.ongoing_chunk_generation.remove(&uw);

                    chunks_for_meshing.extend_from_slice(&self.world.insert_chunk_stack(chunk_gen));
                }
                ChunkJobResultType::Cancelled => {}
            }
        }

        chunks_for_meshing
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
            .replace(uvw, ChunkState::BufferingInProcess)
            .expect("Chunk isn't within render distance");

        let old_draw_calls =
            if let ChunkState::BufferedAndDrawn(DrawnChunkState { draw_calls, .. }) = chunk_state {
                draw_calls
            } else {
                EnumMap::default()
            };

        let buffers =
            worker::create_mesh(&ChunkMeshingContext::create(&self.world, uvw.to_uw()), uvw);
        let mut actual_buffers = EnumMap::default();

        let mut new_draw_calls = EnumMap::default();
        for ((terrain_type, old_draw_call), (_, buffer)) in
            old_draw_calls.into_iter().zip_eq(buffers.iter())
        {
            match (old_draw_call, buffer) {
                (None, Some(buffer)) => {
                    let actual_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: buffer,
                            usage: BufferUsages::COPY_SRC,
                        });
                    let draw_call = update_pass.prepare_insert_region(
                        terrain_type,
                        actual_buffer.to_owned(),
                        (actual_buffer.size() / terrain_type.instance_size())
                            .try_into()
                            .unwrap(),
                        uvw,
                    );
                    actual_buffers[terrain_type] = Some(actual_buffer);
                    new_draw_calls[terrain_type] = Some(draw_call);
                }
                (Some(old_draw_call), None) => {
                    update_pass.prepare_drop_region(old_draw_call);
                }
                (Some(old_draw_call), Some(new_buffer)) => {
                    let actual_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: new_buffer,
                            usage: BufferUsages::COPY_SRC,
                        });
                    update_pass.prepare_replace_region(
                        old_draw_call,
                        // TODO drop to_owned everywhere
                        actual_buffer.to_owned(),
                        (actual_buffer.size() / terrain_type.instance_size())
                            .try_into()
                            .unwrap(),
                    );
                    actual_buffers[terrain_type] = Some(actual_buffer);
                    new_draw_calls[terrain_type] = Some(old_draw_call);
                }
                (None, None) => (),
            }
        }

        *self
            .grid
            .at_mut(uvw)
            .expect("Chunk is not within render distance") =
            ChunkState::BufferedAndDrawn(DrawnChunkState {
                buffers: actual_buffers,
                draw_calls: new_draw_calls,
            });
    }

    /// Execute jobs in `grid_ctx.job_buffer` to threads in the thread pool.
    /// This method drains the jobs buffer.
    /// The jobs are ordered by the horizontal distance relative to the chunk the player is in.
    fn execute_jobs(&mut self) {
        if self.grid_ctx.job_buffer.is_empty() {
            return;
        }

        // Sort by distance between job chunk uw and center chunk uw
        self.grid_ctx.job_buffer.sort_unstable_by_key(|job| {
            let uw = match &job.job {
                ChunkJobType::Mesh { uvw, .. } => uvw.to_uw(),
                // ChunkJobType::Generate { chunk_stack } => *&chunk_stack.uw(),
                ChunkJobType::Generate { uw } => *uw,
            };

            (IVec2::from(uw) - self.grid.center().xz()).length_squared()
        });

        let jobs: Vec<Job<ChunkJobResult, ExecutorContext>> = self
            .grid_ctx
            .job_buffer
            .drain(..)
            .map(|job| worker::create_job(job))
            .collect();

        self.executor
            .dispatch(jobs.into_boxed_slice(), self.executor_ctx.clone());
    }

    pub fn reload_world(&mut self, indirect_buffer: &mut IndirectBufferManager<TerrainType>) {
        self.world.clear();
        indirect_buffer.clear();

        self.executor_ctx.job_id_cutoff.store(
            self.grid_ctx.job_counter.next(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.grid_ctx.ongoing_chunk_generation.clear();
        self.grid_ctx.ongoing_chunk_meshing.clear();
        self.grid_ctx.job_buffer.clear();

        // Set cutoff id for existing jobs
        let new_cutoff_id = self.grid_ctx.job_counter.next();
        self.executor_ctx
            .job_id_cutoff
            .store(new_cutoff_id, Ordering::Relaxed);

        // Set new context for all new jobs, with the new world gen settings
        match load_worldgen_settings() {
            Ok(world_gen_settings) => {
                self.executor_ctx = Arc::new(ExecutorContext {
                    world_gen_settings,
                    job_id_cutoff: AtomicU64::new(new_cutoff_id),
                })
            }
            Err(err) => log::warn!("Failed to parse worldgen settings: {err}"),
        }

        self.grid
            .reset(&mut self.grid_ctx, update_rolling_grid(&self.world));
        self.execute_jobs();
    }

    pub fn world(&self) -> &World {
        &self.world
    }
}

/// Return closure that manages jobs and bookkeeping during grid creation and reposition.
fn update_rolling_grid(world: &World) -> impl Fn(&mut ChunkGridContext, ChunkUVW) -> ChunkState {
    |grid_ctx, uvw| {
        if !ChunkStack::validate_chunk_v(uvw.v) {
            return ChunkState::OutOfBounds;
        }

        if world.is_complete(uvw.to_uw()) {
            if grid_ctx.ongoing_chunk_meshing.insert(uvw) {
                grid_ctx.job_buffer.push(ChunkJob {
                    job_id: grid_ctx.job_counter.next(),
                    job: ChunkJobType::Mesh {
                        chunk_context: ChunkMeshingContext::create(world, uvw.to_uw()),
                        uvw,
                    },
                });
            }
        } else {
            for (u_shift, w_shift) in (-1..=1).cartesian_product(-1..=1) {
                let mut uw = uvw.to_uw();
                // TODO prettier arithmetic
                uw.u += u_shift;
                uw.w += w_shift;
                if !world.is_generated(uw) && grid_ctx.ongoing_chunk_generation.insert(uw) {
                    grid_ctx.job_buffer.push(ChunkJob {
                        job_id: grid_ctx.job_counter.next(),
                        job: ChunkJobType::Generate {
                            // chunk_stack: Box::new(ChunkStack::empty(uw)),
                            uw,
                        },
                    });
                }
            }
        }

        ChunkState::BufferingInProcess
    }
}

#[derive(Error, Debug)]
enum SettingsLoadingError {
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error(transparent)]
    ParseError(#[from] ron::error::SpannedError),
}

fn load_worldgen_settings() -> Result<WorldGenSettings, SettingsLoadingError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        const PATH: &str = "res/config/world-gen.ron";
        Ok(ron::from_str(&std::fs::read_to_string(PATH)?)?)
    }
    #[cfg(target_arch = "wasm32")]
    Ok(ron::from_str(include_str!(
        "../../res/config/world-gen.ron"
    ))?)
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
