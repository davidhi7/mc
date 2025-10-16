use std::{
    collections::HashMap,
    sync::{
        Arc, RwLock,
        mpsc::{Receiver, Sender},
    },
    thread,
    time::Duration,
};

use noise::Simplex;
use wgpu::{
    BufferUsages, Device,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::world::{
    chunk::{Chunk, VERTICAL_CHUNK_COUNT},
    world_loader::{ChunkBuffers, ChunkJob, ChunkJobResult, TerrainBuckets},
};

pub fn launch(
    job_receiver: Receiver<ChunkJob>,
    result_sender: Sender<ChunkJobResult>,
    device: Device,
    noise: Simplex,
) {
    loop {
        let job = match job_receiver.recv_timeout(Duration::MAX) {
            Ok(job) => job,
            Err(err) => {
                eprintln!("{:?}: {:?}", thread::current().id(), err);
                return;
            }
        };

        let mut chunk_stack_created = false;
        let chunk_stack = match job {
            ChunkJob::Mesh { chunk_stack } => chunk_stack,
            ChunkJob::GenerateAndMesh { uw } => {
                chunk_stack_created = true;
                Arc::new(RwLock::new(Chunk::generate_stack(&noise, uw)))
            }
        };

        let uw = chunk_stack.read().unwrap().uw;

        let chunk_buffers = (0..VERTICAL_CHUNK_COUNT)
            .map(|v| {
                let (solid_instances, transparent_instances) =
                    chunk_stack.read().unwrap().chunks[v].generate_mesh();
                let mut buffers = HashMap::with_capacity(2);

                if solid_instances.len() > 0 {
                    buffers.insert(
                        TerrainBuckets::SOLID,
                        (
                            device.create_buffer_init(&BufferInitDescriptor {
                                label: Some(
                                    format!(
                                        "{:?} terrain mesh at {:?}",
                                        TerrainBuckets::SOLID,
                                        uw.to_uvw(v as i32)
                                    )
                                    .as_str(),
                                ),
                                contents: bytemuck::cast_slice(solid_instances.as_slice()),
                                usage: BufferUsages::COPY_SRC,
                            }),
                            solid_instances.len() as u32,
                        ),
                    );
                }

                if transparent_instances.len() > 0 {
                    buffers.insert(
                        TerrainBuckets::TRANSPARENT,
                        (
                            device.create_buffer_init(&BufferInitDescriptor {
                                label: Some(
                                    format!(
                                        "{:?} terrain mesh at {:?}",
                                        TerrainBuckets::TRANSPARENT,
                                        uw.to_uvw(v as i32)
                                    )
                                    .as_str(),
                                ),
                                contents: bytemuck::cast_slice(transparent_instances.as_slice()),
                                usage: BufferUsages::COPY_SRC,
                            }),
                            transparent_instances.len() as u32,
                        ),
                    );
                }
                ChunkBuffers { buffers }
            })
            .collect::<Vec<ChunkBuffers>>();

        result_sender
            .send(ChunkJobResult {
                uw,
                chunk_stack: if chunk_stack_created {
                    Some(chunk_stack)
                } else {
                    None
                },
                chunk_buffers,
            })
            .expect("Couldn't send result to main thread");
    }
}
