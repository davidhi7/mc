use std::{array, collections::HashMap};

use noise::NoiseFn;

use crate::{
    world::CubeFaceInstance,
    world::{
        blocks::{Block, Direction},
        CHUNK_DIMENSIONS, VERTICAL_CHUNK_COUNT, WORLD_HEIGHT,
    },
};

pub struct Chunk {
    pub u: i32,
    pub v: u32,
    pub w: i32,
    data: Box<[Block]>,
}

impl Chunk {
    pub fn generate_stack(
        noise: &impl NoiseFn<f64, 2>,
        u: i32,
        w: i32,
    ) -> [Self; VERTICAL_CHUNK_COUNT] {
        const TOTAL_BLOCK_COUNT: i32 = (CHUNK_DIMENSIONS + 2).pow(3);

        let mut blocks = Vec::with_capacity(TOTAL_BLOCK_COUNT as usize);
        for _ in 0..TOTAL_BLOCK_COUNT {
            blocks.push(Block::AIR);
        }

        let mut chunks: [Chunk; VERTICAL_CHUNK_COUNT] = array::from_fn(|v| Chunk {
            u,
            v: v as u32,
            w,
            data: blocks.clone().into_boxed_slice(),
        });

        for x in (-1)..CHUNK_DIMENSIONS + 1 {
            for z in (-1)..CHUNK_DIMENSIONS + 1 {
                let nx = u as f64 + (x as f64 / CHUNK_DIMENSIONS as f64) - 0.5;
                let nz = w as f64 + (z as f64 / CHUNK_DIMENSIONS as f64) - 0.5;

                let mut height = noise.get([0.3 * nx, 0.3 * nz])
                    + 0.5 * noise.get([nx, nz])
                    + 0.25 * noise.get([3.0 * nx, 3.0 * nz]);
                height /= 1.75 * 2.0;
                height += 0.5;
                height = height.powf(2.5 * (2.0 + noise.get([nx / 10.0, nx / 10.0])));
                height *= WORLD_HEIGHT as f64;
                let height = height.round() as i32;

                let mut current_v = 0;
                for y in 0..height {
                    if y % CHUNK_DIMENSIONS == CHUNK_DIMENSIONS - 1
                        && current_v < VERTICAL_CHUNK_COUNT
                    {
                        *chunks[current_v + 1].at_mut(x, -1, z) = Block::STONE;
                    } else if y % CHUNK_DIMENSIONS == 0 && y != 0 {
                        *chunks[current_v].at_mut(x, CHUNK_DIMENSIONS, z) = Block::STONE;
                        current_v += 1;
                    }

                    *chunks[current_v].at_mut(x, y % CHUNK_DIMENSIONS, z) = Block::STONE;
                }

                *chunks[(height / CHUNK_DIMENSIONS) as usize].at_mut(
                    x,
                    height % CHUNK_DIMENSIONS,
                    z,
                ) = Block::GRASS;
            }
        }

        chunks
    }

    fn validate_chunk_coordinates(x: i32, y: i32, z: i32) -> bool {
        !(x < -1
            || x > CHUNK_DIMENSIONS
            || y < -1
            || y > CHUNK_DIMENSIONS
            || z < -1
            || z > CHUNK_DIMENSIONS)
    }

    pub fn at(&self, x: i32, y: i32, z: i32) -> &Block {
        if !Chunk::validate_chunk_coordinates(x, y, z) {
            panic!("Invalid chunk coordinates x={} y={} z={} ", x, y, z);
        } else {
            &self.data[(((x + 1) * (CHUNK_DIMENSIONS + 2) + y + 1) * (CHUNK_DIMENSIONS + 2) + z + 1)
                as usize]
        }
    }

    pub fn at_mut(&mut self, x: i32, y: i32, z: i32) -> &mut Block {
        if !Chunk::validate_chunk_coordinates(x, y, z) {
            panic!("Invalid chunk coordinates x={} y={} z={} ", x, y, z);
        } else {
            &mut self.data[(((x + 1) * (CHUNK_DIMENSIONS + 2) + y + 1) * (CHUNK_DIMENSIONS + 2)
                + z
                + 1) as usize]
        }
    }

    pub fn generate_mesh(&self) -> Vec<CubeFaceInstance> {
        let mut instances = Vec::new();

        for x in 0..CHUNK_DIMENSIONS {
            for z in 0..CHUNK_DIMENSIONS {
                for y in 0..CHUNK_DIMENSIONS {
                    if let Block::AIR = self.at(x, y, z) {
                        continue;
                    }

                    let mut directions = Vec::with_capacity(6);

                    if let Block::AIR = self.at(x - 1, y, z) {
                        directions.push(Direction::NegX)
                    }
                    if let Block::AIR = self.at(x + 1, y, z) {
                        directions.push(Direction::X)
                    }
                    if let Block::AIR = self.at(x, y - 1, z) {
                        directions.push(Direction::NegY)
                    }
                    if let Block::AIR = self.at(x, y + 1, z) {
                        directions.push(Direction::Y)
                    }
                    if let Block::AIR = self.at(x, y, z - 1) {
                        directions.push(Direction::NegZ)
                    }
                    if let Block::AIR = self.at(x, y, z + 1) {
                        directions.push(Direction::Z)
                    }

                    let tex_index = self.at(x, y, z).texture_index();
                    for direction in directions {
                        instances.push(CubeFaceInstance {
                            position: glam::vec3(
                                (x + self.u * CHUNK_DIMENSIONS) as f32,
                                (y + self.v as i32 * CHUNK_DIMENSIONS) as f32,
                                (z + self.w * CHUNK_DIMENSIONS) as f32,
                            ),
                            direction,
                            tex_index,
                        })
                    }
                }
            }
        }

        instances
    }
}
