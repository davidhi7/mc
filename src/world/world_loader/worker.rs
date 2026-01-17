use std::{
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, Sender},
    },
    thread,
};

use enum_map::EnumMap;
use wgpu::{
    Buffer, BufferUsages, Device,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::world::{
    chunk::{ChunkMeshingContext, ChunkUVW},
    world_gen::{self, WorldGenSettings},
    world_loader::{ChunkJob, ChunkJobResult, ChunkJobResultType, ChunkJobType, TerrainType},
};

pub fn launch(
    recv: Receiver<ChunkJob>,
    send: Sender<ChunkJobResult>,
    job_cutoff_id: Arc<AtomicU64>,
    world_gen_settings: Arc<RwLock<WorldGenSettings>>,
    device: Device,
) {
    loop {
        let ChunkJob { job_id, job } = match recv.recv() {
            Ok(job) => job,
            Err(err) => {
                eprintln!("{:?}: {:?}", thread::current().id(), err);
                return;
            }
        };

        if job_id <= job_cutoff_id.load(Ordering::Relaxed) {
            continue;
        }

        let result = match &job {
            ChunkJobType::Mesh { ctx, uvw } => {
                let buffers = create_mesh(&device, ctx, *uvw);
                ChunkJobResultType::Mesh { buffers, uvw: *uvw }
            }
            ChunkJobType::Generate { uw } => {
                let chunk_gen = world_gen::generate(&world_gen_settings.read().unwrap(), *uw);

                ChunkJobResultType::Generate { chunk_gen }
            }
        };

        send.send(ChunkJobResult { job_id, result })
            .expect("Couldn't send result to main thread");
    }
}

pub fn create_mesh(
    device: &Device,
    ctx: &ChunkMeshingContext,
    uvw: ChunkUVW,
) -> EnumMap<TerrainType, Option<Buffer>> {
    let (solid_instances, transparent_instances) = ctx.generate_mesh(uvw);
    let mut buffers = EnumMap::default();

    if !solid_instances.is_empty() {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!(
                "{:?} terrain mesh at {:?}",
                TerrainType::Solid,
                uvw
            )),
            contents: bytemuck::cast_slice(solid_instances.as_slice()),
            usage: BufferUsages::COPY_SRC,
        });

        buffers[TerrainType::Solid] = Some(buffer);
    }

    if !transparent_instances.is_empty() {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!(
                "{:?} terrain mesh at {:?}",
                TerrainType::Transparent,
                uvw
            )),
            contents: bytemuck::cast_slice(transparent_instances.as_slice()),
            usage: BufferUsages::COPY_SRC,
        });

        buffers[TerrainType::Transparent] = Some(buffer);
    }

    buffers
}
