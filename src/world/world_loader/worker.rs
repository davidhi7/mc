use std::{
    array,
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
    chunk::Chunk,
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
        let ChunkJob { id, job } = match recv.recv() {
            Ok(job) => job,
            Err(err) => {
                eprintln!("{:?}: {:?}", thread::current().id(), err);
                return;
            }
        };

        if id <= job_cutoff_id.load(Ordering::Relaxed) {
            continue;
        }

        let result = match job {
            ChunkJobType::Mesh { chunk } => {
                let buffers = create_mesh(&device, &chunk);
                ChunkJobResultType::Mesh { chunk, buffers }
            }
            ChunkJobType::GenerateAndMeshStack { uw } => {
                let chunk_stack = world_gen::generate(&world_gen_settings.read().unwrap(), uw);
                let buffers = Box::new(array::from_fn(|v| {
                    create_mesh(&device, &chunk_stack.chunks[v])
                }));

                ChunkJobResultType::GenerateAndMeshStack {
                    chunk_stack,
                    buffers,
                }
            }
        };

        send.send(ChunkJobResult { id, result })
            .expect("Couldn't send result to main thread");
    }
}

pub fn create_mesh(device: &Device, chunk: &Chunk) -> EnumMap<TerrainType, Option<Buffer>> {
    let (solid_instances, transparent_instances) = chunk.generate_mesh();
    let mut buffers = EnumMap::default();

    if !solid_instances.is_empty() {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!(
                "{:?} terrain mesh at {:?}",
                TerrainType::Solid,
                chunk.uvw()
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
                chunk.uvw()
            )),
            contents: bytemuck::cast_slice(transparent_instances.as_slice()),
            usage: BufferUsages::COPY_SRC,
        });

        buffers[TerrainType::Transparent] = Some(buffer);
    }

    buffers
}
