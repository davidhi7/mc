use std::array;

use noise::NoiseFn;

use crate::{
    renderer::vertex_buffer::{QuadInstance, TransparentQuadInstance},
    world::{
        blocks::{Block, BlockType, Direction},
        coordinates::Coordinates,
    },
};

pub const CHUNK_WIDTH_BITS: u32 = 5;
// TODO make into usize
pub const CHUNK_WIDTH: u32 = 2_u32.pow(CHUNK_WIDTH_BITS);
pub const CHUNK_WIDTH_I32: i32 = CHUNK_WIDTH as i32;
const CHUNK_WIDTH_P: u32 = CHUNK_WIDTH + 2;
const CHUNK_WIDTH_P_I32: i32 = CHUNK_WIDTH_P as i32;

pub const VERTICAL_CHUNK_COUNT: usize = 8;
pub const WORLD_HEIGHT: u32 = CHUNK_WIDTH * VERTICAL_CHUNK_COUNT as u32;

const MIN_HEIGHT: u32 = 8;
const SEA_LEVEL: u32 = 24;

pub type ChunkUW = (i32, i32);
pub type ChunkUVW = (i32, i32, i32);

#[allow(dead_code)]
#[derive(Clone)]
pub struct ChunkStack {
    pub u: i32,
    pub w: i32,
    pub chunks: [Chunk; VERTICAL_CHUNK_COUNT],
    pub height_map: [u32; (CHUNK_WIDTH * CHUNK_WIDTH) as usize],
}

#[derive(Clone)]
pub struct Chunk {
    data: Box<[Block]>,
}

impl Chunk {
    pub fn generate_stack(noise: &impl NoiseFn<f64, 2>, uw: ChunkUW) -> ChunkStack {
        const TOTAL_BLOCK_COUNT: usize = (CHUNK_WIDTH as usize + 2).pow(3);

        // Directly generating an array with something like [Block::AIR; TOTAL_BLOCK_COUNT] on the stack could cause a stack overflow
        let mut blocks = Vec::with_capacity(TOTAL_BLOCK_COUNT);
        for _ in 0..TOTAL_BLOCK_COUNT {
            blocks.push(Block::AIR);
        }

        let chunks: [Chunk; VERTICAL_CHUNK_COUNT] = array::from_fn(|_| Chunk {
            data: blocks.clone().into_boxed_slice(),
        });

        let mut chunk_stack = ChunkStack {
            u: uw.0,
            w: uw.1,
            chunks,
            height_map: [0; CHUNK_WIDTH.pow(2) as usize],
        };

        for x in (-1)..CHUNK_WIDTH_I32 + 1 {
            for z in (-1)..CHUNK_WIDTH_I32 + 1 {
                let nx = uw.0 as f64 + (x as f64 / CHUNK_WIDTH as f64) - 0.5;
                let nz = uw.1 as f64 + (z as f64 / CHUNK_WIDTH as f64) - 0.5;

                let mut height = noise.get([0.3 * nx, 0.3 * nz])
                    + 0.5 * noise.get([nx, nz])
                    + 0.25 * noise.get([3.0 * nx, 3.0 * nz]);
                height /= 1.75 * 2.0;
                height += 0.5;
                height = height.powf(2.5 * (2.0 + noise.get([nx / 10.0, nx / 10.0])));
                height *= (WORLD_HEIGHT - MIN_HEIGHT - 1) as f64;
                // Always have a height >= MIN_HEIGHT
                let height = height.round() as u32 + MIN_HEIGHT;

                let mut block_array = Vec::new();
                block_array.push((0..height, Block::STONE));
                if height < SEA_LEVEL {
                    block_array.push((height..height + 1, Block::SAND));
                    block_array.push((height + 1..SEA_LEVEL, Block::WATER));
                } else {
                    block_array.push((height..height + 1, Block::GRASS));
                }

                for (range, block) in block_array {
                    for y in range {
                        Chunk::insert_into_chunk_stack(&mut chunk_stack, x, y, z, block);
                    }
                }

                if (0..CHUNK_WIDTH as i32).contains(&z) && (0..CHUNK_WIDTH as i32).contains(&x) {
                    chunk_stack.height_map[(z as u32 * CHUNK_WIDTH + x as u32) as usize] = height;
                }
            }
        }

        chunk_stack
    }

    fn insert_into_chunk_stack(
        chunk_stack: &mut ChunkStack,
        x: i32,
        global_y: u32,
        z: i32,
        block: Block,
    ) {
        let y = global_y % CHUNK_WIDTH;
        let v = (global_y / CHUNK_WIDTH) as usize;

        *chunk_stack.chunks[v].at_mut(x, y as i32, z) = block;

        if y == 0 && v > 0 {
            *chunk_stack.chunks[v - 1].at_mut(x, CHUNK_WIDTH_I32, z) = block;
        } else if y == CHUNK_WIDTH - 1 && v < VERTICAL_CHUNK_COUNT - 1 {
            *chunk_stack.chunks[v + 1].at_mut(x, -1, z) = block;
        }
    }

    fn validate_chunk_coordinates(x: i32, y: i32, z: i32) -> bool {
        let range = -1..=CHUNK_WIDTH_I32;
        range.contains(&x) && range.contains(&y) && range.contains(&z)
    }

    pub fn at(&self, x: i32, y: i32, z: i32) -> &Block {
        if !Chunk::validate_chunk_coordinates(x, y, z) {
            panic!("Invalid chunk coordinates x={} y={} z={} ", x, y, z);
        }
        let index = (((x + 1) * CHUNK_WIDTH_P_I32 + y + 1) * CHUNK_WIDTH_P_I32 + z + 1) as usize;
        &self.data[index]
    }

    pub fn at_coords(&self, coords: Coordinates) -> &Block {
        self.at(coords.x(), coords.y(), coords.z())
    }

    pub fn at_mut(&mut self, x: i32, y: i32, z: i32) -> &mut Block {
        if !Chunk::validate_chunk_coordinates(x, y, z) {
            panic!("Invalid chunk coordinates x={} y={} z={} ", x, y, z);
        }
        let index = (((x + 1) * CHUNK_WIDTH_P_I32 + y + 1) * CHUNK_WIDTH_P_I32 + z + 1) as usize;
        &mut self.data[index]
    }

    pub fn generate_mesh(&self) -> (Vec<QuadInstance>, Vec<TransparentQuadInstance>) {
        let mut solid_instances = Vec::new();
        let mut transparent_instances = Vec::new();

        for x in 0..CHUNK_WIDTH_I32 {
            for z in 0..CHUNK_WIDTH_I32 {
                for y in 0..CHUNK_WIDTH_I32 {
                    let block_type = self.at(x, y, z).get_block_type();
                    if let BlockType::INVISIBLE = block_type {
                        continue;
                    }

                    let mut directions = Vec::with_capacity(6);
                    if Chunk::is_face_visible(block_type, self.at(x - 1, y, z).get_block_type()) {
                        directions.push(Direction::NegX)
                    }
                    if Chunk::is_face_visible(block_type, self.at(x + 1, y, z).get_block_type()) {
                        directions.push(Direction::X)
                    }
                    if Chunk::is_face_visible(block_type, self.at(x, y - 1, z).get_block_type()) {
                        directions.push(Direction::NegY)
                    }
                    if Chunk::is_face_visible(block_type, self.at(x, y + 1, z).get_block_type()) {
                        directions.push(Direction::Y)
                    }
                    if Chunk::is_face_visible(block_type, self.at(x, y, z - 1).get_block_type()) {
                        directions.push(Direction::NegZ)
                    }
                    if Chunk::is_face_visible(block_type, self.at(x, y, z + 1).get_block_type()) {
                        directions.push(Direction::Z)
                    }

                    let tex_index = self.at(x, y, z).texture_index();

                    let common_packed_bits: u32 = x as u32
                        | ((y as u32) << CHUNK_WIDTH_BITS)
                        | ((z as u32) << (CHUNK_WIDTH_BITS * 2))
                        | ((tex_index as u32) << (CHUNK_WIDTH_BITS * 3));

                    for direction in directions {
                        let attributes =
                            common_packed_bits | ((direction as u32) << (CHUNK_WIDTH_BITS * 3 + 8));

                        if let BlockType::SOLID = block_type {
                            let instance = QuadInstance {
                                attributes,
                                ao_attributes: self
                                    .get_ao_attributes(Coordinates::new(x, y, z), direction),
                            };
                            solid_instances.push(instance);
                        } else if let BlockType::TRANSPARENT = block_type {
                            let instance = TransparentQuadInstance { attributes };
                            transparent_instances.push(instance);
                        }
                    }
                }
            }
        }

        (solid_instances, transparent_instances)
    }

    fn is_face_visible(block: BlockType, adjacent_block: BlockType) -> bool {
        // If the block is solid, all sides adjacent to transparent or invisible blocks are visible
        // If the block is transparent, only sides adjacent to transparent blocks are visible
        match block {
            BlockType::INVISIBLE => false,
            BlockType::SOLID => match adjacent_block {
                BlockType::SOLID => false,
                BlockType::TRANSPARENT | BlockType::INVISIBLE => true,
            },
            BlockType::TRANSPARENT => match adjacent_block {
                BlockType::SOLID | BlockType::TRANSPARENT => false,
                BlockType::INVISIBLE => true,
            },
        }
    }

    fn get_ao_attributes(&self, block: Coordinates, direction: Direction) -> u32 {
        let cross_directions = match direction {
            Direction::NegX => (Direction::Y, Direction::Z),
            Direction::X => (Direction::Z, Direction::Y),

            Direction::NegY => (Direction::Z, Direction::X),
            Direction::Y => (Direction::X, Direction::Z),

            Direction::NegZ => (Direction::X, Direction::Y),
            Direction::Z => (Direction::Y, Direction::X),
        };
        let air_block = block.go(direction, 1);

        let mut factor = 0;

        for i in 0..4 {
            // step 0 is -/-/+/+
            // step 1 is -/+/-/+
            let step_0 = if i < 2 { -1 } else { 1 };
            let step_1 = if i & 1 == 1 { 1 } else { -1 };

            let side_1 = self
                .at_coords(air_block.go(cross_directions.0, step_0))
                .is_solid();
            let side_2 = self
                .at_coords(air_block.go(cross_directions.1, step_1))
                .is_solid();

            let corner = self
                .at_coords(
                    air_block
                        .go(cross_directions.0, step_0)
                        .go(cross_directions.1, step_1),
                )
                .is_solid();

            let value = if side_1 && side_2 {
                3
            } else {
                (side_1 as u32) + (side_2 as u32) + (corner as u32)
            };

            factor |= value << (2 * i);
        }

        factor
    }
}
