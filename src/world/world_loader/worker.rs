use std::sync::Arc;

use enum_map::EnumMap;

use crate::world::{
    chunk::{ChunkMeshingContext, ChunkUVW},
    world_gen::{self},
    world_loader::{
        ChunkJob, ChunkJobResult, ChunkJobResultType, ChunkJobType, ExecutorContext, TerrainType,
    },
};

pub fn create_job(
    job: ChunkJob,
) -> Box<dyn FnOnce(Arc<ExecutorContext>) -> ChunkJobResult + 'static + Send> {
    let ChunkJob { job_id, job } = job;
    match job {
        ChunkJobType::Mesh { chunk_context, uvw } => Box::new(move |ctx| {
            if ctx.job_id_cutoff.load(std::sync::atomic::Ordering::Relaxed) > job_id {
                return ChunkJobResult {
                    job_id,
                    result: ChunkJobResultType::Cancelled,
                };
            }
            let buffers = create_mesh(&chunk_context, uvw);
            ChunkJobResult {
                job_id,
                result: ChunkJobResultType::Mesh { uvw, buffers },
            }
        }),
        ChunkJobType::Generate { uw } => Box::new(move |ctx| {
            if ctx.job_id_cutoff.load(std::sync::atomic::Ordering::Relaxed) > job_id {
                return ChunkJobResult {
                    job_id,
                    result: ChunkJobResultType::Cancelled,
                };
            }
            let chunk_gen = world_gen::generate(&ctx.world_gen_settings, uw);

            ChunkJobResult {
                job_id,
                result: ChunkJobResultType::Generate { chunk_gen },
            }
        }),
    }
}

pub fn create_mesh(
    ctx: &ChunkMeshingContext,
    uvw: ChunkUVW,
) -> EnumMap<TerrainType, Option<Box<[u8]>>> {
    // if (uvw.u + uvw.w) & 1 == 0 {
    //     return Default::default();
    // }
    let (solid_instances, transparent_instances) = ctx.generate_mesh(uvw);
    let mut buffers = EnumMap::default();

    if !solid_instances.is_empty() {
        buffers[TerrainType::Solid] = Some(solid_instances.into_boxed_slice());
    }

    if !transparent_instances.is_empty() {
        buffers[TerrainType::Transparent] = Some(transparent_instances.into_boxed_slice());
    }

    buffers
}
