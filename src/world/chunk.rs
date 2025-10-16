use std::array;

use glam::{IVec2, IVec3, ivec2, ivec3};
use noise::NoiseFn;

use crate::{
    renderer::vertex_buffer::{QuadInstance, TransparentQuadInstance},
    world::blocks::{Block, BlockRenderType, Direction},
};

pub const CHUNK_WIDTH_BITS: u32 = 5;
pub const CHUNK_WIDTH: usize = 2_usize.pow(CHUNK_WIDTH_BITS);
pub const CHUNK_WIDTH_I32: i32 = CHUNK_WIDTH as i32;

const CHUNK_WIDTH_P: usize = CHUNK_WIDTH + 2;
const CHUNK_WIDTH_P_I32: i32 = CHUNK_WIDTH_P as i32;

pub const VERTICAL_CHUNK_COUNT: usize = 4;

pub const WORLD_HEIGHT: usize = CHUNK_WIDTH * VERTICAL_CHUNK_COUNT;

const MIN_HEIGHT: usize = 8;
const SEA_LEVEL: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkUW {
    pub u: i32,
    pub w: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkUVW {
    pub u: i32,
    pub v: i32,
    pub w: i32,
}

impl ChunkUW {
    pub fn to_uvw(&self, v: i32) -> ChunkUVW {
        ChunkUVW {
            u: self.u,
            v,
            w: self.w,
        }
    }
}

impl From<IVec2> for ChunkUW {
    fn from(value: IVec2) -> Self {
        ChunkUW {
            u: value.x,
            w: value.y,
        }
    }
}

impl From<ChunkUW> for IVec2 {
    fn from(value: ChunkUW) -> Self {
        ivec2(value.u, value.w)
    }
}

impl ChunkUVW {
    pub fn to_uw(&self) -> ChunkUW {
        ChunkUW {
            u: self.u,
            w: self.w,
        }
    }
}

impl From<IVec3> for ChunkUVW {
    fn from(value: IVec3) -> Self {
        ChunkUVW {
            u: value.x,
            v: value.y,
            w: value.z,
        }
    }
}

impl From<ChunkUVW> for IVec3 {
    fn from(value: ChunkUVW) -> Self {
        ivec3(value.u, value.v, value.w)
    }
}

#[derive(Clone, Debug)]
pub struct ChunkStack {
    pub uw: ChunkUW,
    pub chunks: [Chunk; VERTICAL_CHUNK_COUNT],
}

impl ChunkStack {
    pub fn validate_chunk_v(v: i32) -> bool {
        v >= 0 && v < VERTICAL_CHUNK_COUNT as i32
    }
}

#[derive(Clone, Debug)]
pub struct Chunk {
    data: Box<[Block]>,
}

impl Chunk {
    pub fn generate_stack(noise: &impl NoiseFn<f64, 2>, uw: ChunkUW) -> ChunkStack {
        const TOTAL_BLOCK_COUNT: usize = CHUNK_WIDTH_P.pow(3);

        let blocks = vec![Block::AIR; TOTAL_BLOCK_COUNT];

        let chunks: [Chunk; VERTICAL_CHUNK_COUNT] = array::from_fn(|_| Chunk {
            data: blocks.clone().into_boxed_slice(),
        });

        let mut chunk_stack = ChunkStack { uw, chunks };

        for x in (-1)..CHUNK_WIDTH_I32 + 1 {
            for z in (-1)..CHUNK_WIDTH_I32 + 1 {
                let nx = uw.u as f64 + (x as f64 / CHUNK_WIDTH as f64) - 0.5;
                let nz = uw.w as f64 + (z as f64 / CHUNK_WIDTH as f64) - 0.5;

                let mut height = noise.get([0.3 * nx, 0.3 * nz])
                    + 0.5 * noise.get([nx, nz])
                    + 0.25 * noise.get([3.0 * nx, 3.0 * nz]);
                height /= 1.75 * 2.0;
                height += 0.5;
                height = height.powf(2.5 * (2.0 + noise.get([nx / 10.0, nx / 10.0])));
                height *= (WORLD_HEIGHT - MIN_HEIGHT - 1) as f64;
                // Always have a height >= MIN_HEIGHT
                let height = height.round() as usize + MIN_HEIGHT;

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
            }
        }

        chunk_stack
    }

    fn insert_into_chunk_stack(
        chunk_stack: &mut ChunkStack,
        x: i32,
        global_y: usize,
        z: i32,
        block: Block,
    ) {
        let y = global_y % CHUNK_WIDTH;
        let v = global_y / CHUNK_WIDTH;

        *chunk_stack.chunks[v].at_mut_with_padding(ivec3(x, y as i32, z)) = block;

        if y == 0 && v > 0 {
            *chunk_stack.chunks[v - 1].at_mut_with_padding(ivec3(x, CHUNK_WIDTH_I32, z)) = block;
        } else if y == CHUNK_WIDTH - 1 && v < VERTICAL_CHUNK_COUNT - 1 {
            *chunk_stack.chunks[v + 1].at_mut_with_padding(ivec3(x, -1, z)) = block;
        }
    }

    fn validate_chunk_coordinates(block: IVec3) -> bool {
        let IVec3 { x, y, z } = block;
        let range = 0..CHUNK_WIDTH_I32;
        range.contains(&x) && range.contains(&y) && range.contains(&z)
    }

    fn validate_chunk_coordinates_with_padding(block: IVec3) -> bool {
        let IVec3 { x, y, z } = block;
        let range = -1..=CHUNK_WIDTH_I32;
        range.contains(&x) && range.contains(&y) && range.contains(&z)
    }

    fn array_index(x: i32, y: i32, z: i32) -> usize {
        (((x + 1) * CHUNK_WIDTH_P_I32 + y + 1) * CHUNK_WIDTH_P_I32 + z + 1) as usize
    }

    pub fn at(&self, block: IVec3) -> &Block {
        debug_assert!(
            Chunk::validate_chunk_coordinates(block),
            "Invalid chunk coordinates {}",
            block
        );
        let IVec3 { x, y, z } = block;
        &self.data[Chunk::array_index(x, y, z)]
    }

    pub fn at_mut(&mut self, block: IVec3) -> &mut Block {
        debug_assert!(
            Chunk::validate_chunk_coordinates(block),
            "Invalid chunk coordinates {}",
            block
        );
        let IVec3 { x, y, z } = block;
        &mut self.data[Chunk::array_index(x, y, z)]
    }

    pub fn at_with_padding(&self, block: IVec3) -> &Block {
        debug_assert!(
            Chunk::validate_chunk_coordinates_with_padding(block),
            "Invalid chunk coordinates {}",
            block
        );
        let IVec3 { x, y, z } = block;
        &self.data[Chunk::array_index(x, y, z)]
    }

    pub fn at_mut_with_padding(&mut self, block: IVec3) -> &mut Block {
        debug_assert!(
            Chunk::validate_chunk_coordinates_with_padding(block),
            "Invalid chunk coordinates {}",
            block
        );
        let IVec3 { x, y, z } = block;
        &mut self.data[Chunk::array_index(x, y, z)]
    }

    pub fn generate_mesh(&self) -> (Vec<QuadInstance>, Vec<TransparentQuadInstance>) {
        let mut solid_instances = Vec::new();
        let mut transparent_instances = Vec::new();

        for x in 0..CHUNK_WIDTH_I32 {
            for y in 0..CHUNK_WIDTH_I32 {
                for z in 0..CHUNK_WIDTH_I32 {
                    let coords = ivec3(x, y, z);
                    let block = self.at_with_padding(coords);
                    if let BlockRenderType::INVISIBLE = block.render_type() {
                        continue;
                    }

                    let common_packed_bits: u32 = x as u32
                        | ((y as u32) << CHUNK_WIDTH_BITS)
                        | ((z as u32) << (CHUNK_WIDTH_BITS * 2))
                        | ((block.texture_index() as u32) << (CHUNK_WIDTH_BITS * 3));

                    for direction in Direction::iter() {
                        if !Chunk::is_face_visible(
                            block.render_type(),
                            self.at_with_padding(coords + direction.get_unit_ivec())
                                .render_type(),
                        ) {
                            continue;
                        }

                        let attributes =
                            common_packed_bits | ((direction as u32) << (CHUNK_WIDTH_BITS * 3 + 8));

                        match block.render_type() {
                            BlockRenderType::OPAQUE => {
                                solid_instances.push(QuadInstance {
                                    attributes,
                                    ao_attributes: self.get_ao_attributes(coords, direction),
                                });
                            }
                            BlockRenderType::TRANSPARENT => {
                                transparent_instances.push(TransparentQuadInstance { attributes });
                            }
                            BlockRenderType::INVISIBLE => unreachable!(),
                        };
                    }
                }
            }
        }

        (solid_instances, transparent_instances)
    }

    fn is_face_visible(block: BlockRenderType, adjacent_block: BlockRenderType) -> bool {
        // If the block is solid, all sides adjacent to transparent or invisible blocks are visible
        // If the block is transparent, only sides adjacent to transparent blocks are visible
        match block {
            BlockRenderType::INVISIBLE => false,
            BlockRenderType::OPAQUE => match adjacent_block {
                BlockRenderType::OPAQUE => false,
                BlockRenderType::TRANSPARENT | BlockRenderType::INVISIBLE => true,
            },
            BlockRenderType::TRANSPARENT => match adjacent_block {
                BlockRenderType::OPAQUE | BlockRenderType::TRANSPARENT => false,
                BlockRenderType::INVISIBLE => true,
            },
        }
    }

    fn get_ao_attributes(&self, coords: IVec3, direction: Direction) -> u32 {
        let cross_directions = match direction {
            Direction::NegX => (Direction::Y, Direction::Z),
            Direction::X => (Direction::Z, Direction::Y),

            Direction::NegY => (Direction::Z, Direction::X),
            Direction::Y => (Direction::X, Direction::Z),

            Direction::NegZ => (Direction::X, Direction::Y),
            Direction::Z => (Direction::Y, Direction::X),
        };
        let air_block = coords + direction.get_unit_ivec();

        let mut factor = 0;

        for i in 0..4 {
            // step 0 is -/-/+/+
            // step 1 is -/+/-/+
            let step_0 = if i < 2 { -1 } else { 1 };
            let step_1 = if i & 1 == 1 { 1 } else { -1 };

            let side_1 = self
                .at_with_padding(air_block + step_0 * cross_directions.0.get_unit_ivec())
                .render_type()
                == BlockRenderType::OPAQUE;
            let side_2 = self
                .at_with_padding(air_block + step_1 * cross_directions.1.get_unit_ivec())
                .render_type()
                == BlockRenderType::OPAQUE;

            let corner = self
                .at_with_padding(
                    air_block
                        + step_0 * cross_directions.0.get_unit_ivec()
                        + step_1 * cross_directions.1.get_unit_ivec(),
                )
                .render_type()
                == BlockRenderType::OPAQUE;

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
