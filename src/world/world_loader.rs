use std::{
    collections::HashMap,
    ops::Range,
    sync::Arc,
    thread::{self, JoinHandle},
    time::Instant,
};

use noise::Simplex;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer, BufferUsages, Device,
};

use crate::{
    renderer::CubeFaceInstance,
    world::{
        camera::CameraController,
        chunk::{Chunk, ChunkStack, ChunkUW},
        World, CHUNK_DIMENSIONS, VERTICAL_CHUNK_COUNT,
    },
};

const CHUNKS_PER_TASK: usize = 19;

struct ChunkMeshingTaskInput {
    uw: ChunkUW,
    chunk_stack: Option<ChunkStack>,
}

struct ChunkMeshingTaskOutput {
    uw: ChunkUW,
    chunk_stack: ChunkStack,
    chunk_meshes: [Vec<CubeFaceInstance>; VERTICAL_CHUNK_COUNT],
}

struct ChunkMeshingTask {
    uw_list: Vec<ChunkUW>,
    handle: JoinHandle<Vec<ChunkMeshingTaskOutput>>,
}

pub struct ChunkBuffers {
    pub instance_buffer: Buffer,
    pub chunk_bind_group: BindGroup,
    pub instance_count: u32,
}

pub struct WorldLoader {
    world: World,
    chunk_meshes: HashMap<ChunkUW, [Vec<CubeFaceInstance>; VERTICAL_CHUNK_COUNT]>,
    buffered_chunks: HashMap<ChunkUW, Vec<Option<ChunkBuffers>>>,
    tasks: Vec<ChunkMeshingTask>,
    chunk_view_distance: u32,
}

impl WorldLoader {
    pub fn new(world: World, chunk_view_distance: u32) -> WorldLoader {
        WorldLoader {
            world,
            chunk_meshes: HashMap::new(),
            buffered_chunks: HashMap::new(),
            chunk_view_distance,
            tasks: Vec::new(),
        }
    }

    pub fn complete_finished_threads(&mut self) {
        for i in (0..self.tasks.len()).rev() {
            if self.tasks[i].handle.is_finished() {
                let task = self.tasks.swap_remove(i);
                let result = task
                    .handle
                    .join()
                    .expect("Chunk generation/meshing thread panicked");

                for element in result {
                    self.world.insert_chunks(element.uw, element.chunk_stack);
                    self.chunk_meshes.insert(element.uw, element.chunk_meshes);
                }
            }
        }
    }

    pub fn update(&mut self, camera: &CameraController, device: Arc<Device>) {
        self.complete_finished_threads();

        let mut chunks_to_mesh: Vec<ChunkMeshingTaskInput> = Vec::new();

        let (range_u, _, range_w) = self.visible_chunk_range(camera);

        for u in range_u {
            for w in range_w.clone() {
                let tuple: ChunkUW = (u, w);
                if self.tasks.iter().any(|task| task.uw_list.contains(&tuple)) {
                    // If chunk is currently generated and/or meshed, continue
                    continue;
                }
                if self.chunk_meshes.get(&tuple).is_none() {
                    // If chunk hasn't been meshed, do so
                    chunks_to_mesh.push(ChunkMeshingTaskInput {
                        uw: tuple.to_owned(),
                        chunk_stack: self
                            .world
                            .chunk_stacks
                            .get(&tuple)
                            .map_or(None, |chunks| Some(chunks.clone())),
                    });
                }
            }
        }

        let mut batches: Vec<Vec<ChunkMeshingTaskInput>> = Vec::new();
        let mut last_batch = Vec::new();
        for item in chunks_to_mesh.into_iter() {
            last_batch.push(item);
            if last_batch.len() == CHUNKS_PER_TASK {
                batches.push(last_batch);
                last_batch = Vec::new();
            }
        }
        if !last_batch.is_empty() {
            batches.push(last_batch);
        }

        let noise: Simplex = self.world.noise;

        for batch in batches.into_iter() {
            let chunk_coordinates: Vec<ChunkUW> = batch.iter().map(|item| item.uw).collect();

            let handle = thread::spawn(move || {
                let start_time = Instant::now();

                let mut output: Vec<ChunkMeshingTaskOutput> = Vec::new();

                for chunk in batch {
                    let chunk_stack = chunk
                        .chunk_stack
                        .unwrap_or_else(|| Chunk::generate_stack(&noise, chunk.uw));

                    let chunk_meshes = (0..VERTICAL_CHUNK_COUNT)
                        .map(|v| chunk_stack[v].generate_mesh())
                        .collect::<Vec<Vec<CubeFaceInstance>>>()
                        .try_into()
                        .unwrap();

                    output.push(ChunkMeshingTaskOutput {
                        uw: chunk.uw,
                        chunk_stack,
                        chunk_meshes,
                    });
                }

                println!(
                    "Processed {} chunk stacks in {}ms",
                    output.len(),
                    start_time.elapsed().as_millis()
                );

                output
            });

            println!(
                "Spawned thread for meshing chunks at uw = {:?}",
                chunk_coordinates
            );

            self.tasks.push(ChunkMeshingTask {
                uw_list: chunk_coordinates,
                handle,
            });
        }
    }

    pub fn create_buffers(
        &mut self,
        camera: &CameraController,
        device: Arc<Device>,
        chunk_bind_group_layout: &BindGroupLayout,
    ) {
        // TODO deduplicate code with update function
        let camera_u: i32 = camera.get_position().x as i32 / CHUNK_DIMENSIONS;
        let camera_w = camera.get_position().z as i32 / CHUNK_DIMENSIONS;

        let view_distance_i32 = self.chunk_view_distance as i32;
        let chunk_range_u = camera_u - view_distance_i32..camera_u + view_distance_i32 + 1;
        let chunk_range_w = camera_w - view_distance_i32..camera_w + view_distance_i32 + 1;

        for u in chunk_range_u {
            for w in chunk_range_w.clone() {
                if self
                    .tasks
                    .iter()
                    .any(|task: &ChunkMeshingTask| task.uw_list.contains(&(camera_u, camera_w)))
                {
                    // If chunk is currently generated or meshed, continue
                    continue;
                }
                if !self.buffered_chunks.contains_key(&(u, w))
                    && self.chunk_meshes.contains_key(&(u, w))
                {
                    // If chunk is meshed but not stored in a wgpu buffer, buffer it
                    let meshed_chunks = self.chunk_meshes.get(&(u, w)).unwrap();
                    let mut chunk_buffers = Vec::new();
                    for v in 0..VERTICAL_CHUNK_COUNT {
                        if meshed_chunks[v].len() == 0 {
                            chunk_buffers.push(None);
                            continue;
                        }
                        let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
                            label: Some(format!("u={u} v={v} w={w} instance buffer").as_str()),
                            contents: bytemuck::cast_slice(meshed_chunks[v].as_slice()),
                            usage: BufferUsages::VERTEX,
                        });
                        let chunk_uniform: Buffer =
                            device.create_buffer_init(&BufferInitDescriptor {
                                label: Some(format!("u={u} v={v} w={w} uniform buffer").as_str()),
                                contents: bytemuck::cast_slice(&[
                                    u, v as i32, w, /* alignmnet */ 0,
                                ]),
                                usage: BufferUsages::UNIFORM,
                            });
                        let chunk_bind_group = device.create_bind_group(&BindGroupDescriptor {
                            label: Some(format!("u={u} v={v} w={w} uniform bind group").as_str()),
                            layout: chunk_bind_group_layout,
                            entries: &[BindGroupEntry {
                                binding: 0,
                                resource: chunk_uniform.as_entire_binding(),
                            }],
                        });
                        chunk_buffers.push(Some(ChunkBuffers {
                            instance_buffer,
                            chunk_bind_group,
                            instance_count: meshed_chunks[v].len() as u32,
                        }));
                    }
                    self.buffered_chunks.insert((u, w), chunk_buffers);
                }
            }
        }
    }

    pub fn get_buffer(&self, u: i32, v: i32, w: i32) -> Option<&ChunkBuffers> {
        if self.buffered_chunks.contains_key(&(u, w)) {
            let buffered_chunks = self.buffered_chunks.get(&(u, w));
            let buf = buffered_chunks.unwrap().get(v as usize).unwrap();
            return buf.as_ref();
        }
        None
    }

    pub fn visible_chunk_range(
        &self,
        camera: &CameraController,
    ) -> (Range<i32>, Range<u32>, Range<i32>) {
        let camera_u: i32 = camera.get_position().x as i32 / CHUNK_DIMENSIONS;
        let camera_v: i32 = camera.get_position().y as i32 / CHUNK_DIMENSIONS;
        let camera_w = camera.get_position().z as i32 / CHUNK_DIMENSIONS;

        let view_distance_i32 = self.chunk_view_distance as i32;
        let range_u = camera_u - view_distance_i32..camera_u + view_distance_i32 + 1;
        let range_w = camera_w - view_distance_i32..camera_w + view_distance_i32 + 1;

        let range_v: Range<u32> = i32::max(0, camera_v - view_distance_i32) as u32
            ..u32::min(
                VERTICAL_CHUNK_COUNT as u32,
                (camera_v + view_distance_i32 + 1) as u32,
            );
        (range_u, range_v, range_w)
    }
}
