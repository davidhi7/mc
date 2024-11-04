use std::array;

use noise::NoiseFn;

use crate::{
    world::CubeFaceInstance,
    world::{
        blocks::{Block, Direction},
        CHUNK_DIMENSIONS, CHUNK_WIDTH_BITS, VERTICAL_CHUNK_COUNT, WORLD_HEIGHT,
    },
};

pub type ChunkStack = [Chunk; VERTICAL_CHUNK_COUNT];
pub type ChunkUW = (i32, i32);
pub type ChunkUVW = (i32, i32, i32);

#[derive(Clone)]
pub struct Chunk {
    pub u: i32,
    pub v: i32,
    pub w: i32,
    data: Box<[Block]>,
}

impl Chunk {
    pub fn generate_stack(
        noise: &impl NoiseFn<f64, 2>,
        uw: ChunkUW,
    ) -> [Self; VERTICAL_CHUNK_COUNT] {
        const TOTAL_BLOCK_COUNT: usize = (CHUNK_DIMENSIONS as usize + 2).pow(3);

        // Directly generating an array with something like [Block::AIR; TOTAL_BLOCK_COUNT] on the stack could cause a stack overflow
        let mut blocks = Vec::with_capacity(TOTAL_BLOCK_COUNT);
        for _ in 0..TOTAL_BLOCK_COUNT {
            blocks.push(Block::AIR);
        }

        let mut chunks: [Chunk; VERTICAL_CHUNK_COUNT] = array::from_fn(|v| Chunk {
            u: uw.0,
            v: v as i32,
            w: uw.1,
            data: blocks.clone().into_boxed_slice(),
        });

        for x in (-1)..CHUNK_DIMENSIONS + 1 {
            for z in (-1)..CHUNK_DIMENSIONS + 1 {
                let nx = uw.0 as f64 + (x as f64 / CHUNK_DIMENSIONS as f64) - 0.5;
                let nz = uw.1 as f64 + (z as f64 / CHUNK_DIMENSIONS as f64) - 0.5;

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
        !(!(-1..=CHUNK_DIMENSIONS).contains(&x)
            || !(-1..=CHUNK_DIMENSIONS).contains(&y)
            || !(-1..=CHUNK_DIMENSIONS).contains(&z))
    }

    pub fn at(&self, x: i32, y: i32, z: i32) -> &Block {
        if !Chunk::validate_chunk_coordinates(x, y, z) {
            panic!("Invalid chunk coordinates x={} y={} z={} ", x, y, z);
        }
        let index =
            (((x + 1) * (CHUNK_DIMENSIONS + 2) + y + 1) * (CHUNK_DIMENSIONS + 2) + z + 1) as usize;
        &self.data[index]
    }

    pub fn at_mut(&mut self, x: i32, y: i32, z: i32) -> &mut Block {
        if !Chunk::validate_chunk_coordinates(x, y, z) {
            panic!("Invalid chunk coordinates x={} y={} z={} ", x, y, z);
        }
        let index =
            (((x + 1) * (CHUNK_DIMENSIONS + 2) + y + 1) * (CHUNK_DIMENSIONS + 2) + z + 1) as usize;
        &mut self.data[index]
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

                    let common_packed_bits: u32 = x as u32
                        | ((y as u32) << CHUNK_WIDTH_BITS)
                        | ((z as u32) << (CHUNK_WIDTH_BITS * 2))
                        | ((tex_index as u32) << (CHUNK_WIDTH_BITS * 3));

                    for direction in directions {
                        let packed_bits =
                            common_packed_bits | ((direction as u32) << (CHUNK_WIDTH_BITS * 3 + 8));

                        instances.push(CubeFaceInstance {
                            attributes: packed_bits,
                        })
                    }
                }
            }
        }

        instances
    }
}
