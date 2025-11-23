use std::{
    array,
    sync::mpsc::{Receiver, Sender},
    thread,
};

use enum_map::EnumMap;
use noise::Simplex;
use wgpu::{
    Buffer, BufferUsages, Device,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::world::{
    chunk::Chunk,
    world_loader::{ChunkJob, ChunkJobResult, TerrainType},
};

pub fn launch(
    recv: Receiver<ChunkJob>,
    send: Sender<ChunkJobResult>,
    device: Device,
    noise: Simplex,
) {
    loop {
        let job = match recv.recv() {
            Ok(job) => job,
            Err(err) => {
                eprintln!("{:?}: {:?}", thread::current().id(), err);
                return;
            }
        };

        let result = match job {
            ChunkJob::Mesh { chunk } => {
                let chunk = chunk.read().unwrap();
                ChunkJobResult::Mesh {
                    uvw: chunk.uvw(),
                    buffers: create_mesh(&device, &chunk),
                }
            }
            ChunkJob::GenerateAndMeshStack { uw } => {
                let chunk_stack = Chunk::generate_stack(&noise, uw);
                let buffers = array::from_fn(|v| create_mesh(&device, &chunk_stack.chunks[v]));

                ChunkJobResult::GenerateAndMeshStack {
                    chunk_stack,
                    buffers,
                }
            }
        };

        send.send(result)
            .expect("Couldn't send result to main thread");
    }
}

pub fn create_mesh(device: &Device, chunk: &Chunk) -> EnumMap<TerrainType, Option<Buffer>> {
    let (solid_instances, transparent_instances) = chunk.generate_mesh();
    let mut buffers = EnumMap::default();

    if solid_instances.len() > 0 {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!(
                "{:?} terrain mesh at {:?}",
                TerrainType::SOLID,
                chunk.uvw()
            )),
            contents: bytemuck::cast_slice(solid_instances.as_slice()),
            usage: BufferUsages::COPY_SRC,
        });

        buffers[TerrainType::SOLID] = Some(buffer);
    }

    if transparent_instances.len() > 0 {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!(
                "{:?} terrain mesh at {:?}",
                TerrainType::TRANSPARENT,
                chunk.uvw()
            )),
            contents: bytemuck::cast_slice(transparent_instances.as_slice()),
            usage: BufferUsages::COPY_SRC,
        });

        buffers[TerrainType::TRANSPARENT] = Some(buffer);
    }

    buffers
}
