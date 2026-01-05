use std::{panic, sync::RwLock};

use glam::{IVec2, IVec3, ivec2, ivec3};

use crate::{
    renderer::vertex_buffer::{QuadInstance, TransparentQuadInstance},
    world::blocks::{Block, BlockRenderType, Direction},
};

pub const CHUNK_WIDTH_BITS: u32 = 5;
pub const CHUNK_WIDTH: usize = 2_usize.pow(CHUNK_WIDTH_BITS);
pub const CHUNK_WIDTH_I32: i32 = CHUNK_WIDTH as i32;

pub const CHUNK_WIDTH_P: usize = CHUNK_WIDTH + 2;
pub const CHUNK_WIDTH_P_I32: i32 = CHUNK_WIDTH_P as i32;

pub const VERTICAL_CHUNK_COUNT: usize = 4;

pub const WORLD_HEIGHT: usize = CHUNK_WIDTH * VERTICAL_CHUNK_COUNT;

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
    pub fn to_uvw(self, v: i32) -> ChunkUVW {
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
    pub fn to_uw(self) -> ChunkUW {
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

pub struct ChunkStack {
    pub uw: ChunkUW,
    pub chunks: [Chunk; VERTICAL_CHUNK_COUNT],
}

impl ChunkStack {
    pub fn validate_chunk_v(v: i32) -> bool {
        v >= 0 && v < VERTICAL_CHUNK_COUNT as i32
    }

    pub fn insert(&mut self, pos: IVec3, block: Block) {
        let y = pos.y % CHUNK_WIDTH_I32;
        let v = pos.y as usize / CHUNK_WIDTH;

        if !Self::validate_chunk_v(v as i32) {
            panic!("Invalid vertical chunk component");
        }

        self.chunks[v].set_including_padding(pos.with_y(y), block);

        if y == 0 && v > 0 {
            self.chunks[v - 1].set_including_padding(pos.with_y(CHUNK_WIDTH_I32), block);
        } else if y == CHUNK_WIDTH_I32 - 1 && v < VERTICAL_CHUNK_COUNT - 1 {
            self.chunks[v + 1].set_including_padding(pos.with_y(-1), block);
        }
    }

    pub fn get(&self, pos: IVec3) -> Block {
        let y = pos.y % CHUNK_WIDTH_I32;
        let v = pos.y as usize / CHUNK_WIDTH;

        if !Self::validate_chunk_v(v as i32) {
            panic!("Invalid vertical chunk component");
        }

        self.chunks[v].get_including_padding(pos.with_y(y))
    }
}

pub struct Chunk {
    uvw: ChunkUVW,
    data: RwLock<Box<[Block]>>,
}

impl Chunk {
    pub fn empty(uvw: ChunkUVW) -> Self {
        Chunk {
            uvw,
            data: RwLock::new(vec![Block::Air; CHUNK_WIDTH_P.pow(3)].into_boxed_slice()),
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

    pub fn uvw(&self) -> ChunkUVW {
        self.uvw
    }

    /// Get the block at the given location.
    pub fn get(&self, location: IVec3) -> Block {
        debug_assert!(
            Chunk::validate_chunk_coordinates(location),
            "Invalid chunk coordinates {location}",
        );
        self.get_including_padding(location)
    }

    /// Set the block at the given location, returning the old block.
    #[expect(dead_code)]
    pub fn set(&self, location: IVec3, block: Block) -> Block {
        debug_assert!(
            Chunk::validate_chunk_coordinates(location),
            "Invalid chunk coordinates {location}",
        );
        self.set_including_padding(location, block)
    }

    /// Get the block at the given location.
    /// This function allows to set the blocks copied from adjacent chunks, stored at xy/z/ indexes -1 and CHUNK_WIDTH, respectively.
    pub fn get_including_padding(&self, location: IVec3) -> Block {
        debug_assert!(
            Chunk::validate_chunk_coordinates_with_padding(location),
            "Invalid chunk coordinates {location}",
        );
        let IVec3 { x, y, z } = location;
        self.data.read().unwrap()[Chunk::array_index(x, y, z)]
    }

    /// Set the block at the given location, returning the old block.
    /// This function allows to set the blocks copied from adjacent chunks, stored at xy/z/ indexes -1 and CHUNK_WIDTH, respectively.
    pub fn set_including_padding(&self, location: IVec3, block: Block) -> Block {
        debug_assert!(
            Chunk::validate_chunk_coordinates_with_padding(location),
            "Invalid chunk coordinates {location}",
        );
        let IVec3 { x, y, z } = location;
        std::mem::replace(
            &mut self.data.write().unwrap()[Chunk::array_index(x, y, z)],
            block,
        )
    }

    pub fn generate_mesh(&self) -> (Vec<QuadInstance>, Vec<TransparentQuadInstance>) {
        let mut solid_instances = Vec::new();
        let mut transparent_instances = Vec::new();

        for x in 0..CHUNK_WIDTH_I32 {
            for y in 0..CHUNK_WIDTH_I32 {
                for z in 0..CHUNK_WIDTH_I32 {
                    let coords = ivec3(x, y, z);
                    let block = self.get_including_padding(coords);
                    if let BlockRenderType::Invisible = block.render_type() {
                        continue;
                    }

                    let common_packed_bits: u32 = x as u32
                        | ((y as u32) << CHUNK_WIDTH_BITS)
                        | ((z as u32) << (CHUNK_WIDTH_BITS * 2));

                    for direction in Direction::iter() {
                        if !Chunk::is_face_visible(
                            block,
                            self.get_including_padding(coords + direction.get_unit_ivec()),
                        ) {
                            continue;
                        }

                        let attributes = common_packed_bits
                            | ((block.texture_index(direction) as u32) << (CHUNK_WIDTH_BITS * 3))
                            | ((direction as u32) << (CHUNK_WIDTH_BITS * 3 + 8));

                        match block {
                            Block::Water => {
                                transparent_instances.push(TransparentQuadInstance { attributes });
                            }
                            Block::Air => unreachable!(),
                            _ => {
                                solid_instances.push(QuadInstance {
                                    attributes,
                                    ao_attributes: self.get_ao_attributes(coords, direction),
                                });
                            }
                        };
                    }
                }
            }
        }

        (solid_instances, transparent_instances)
    }

    /// Returns true if `block`'s face that is adjacent to `adjacent_block`'s face is visible.
    fn is_face_visible(block: Block, adjacent_block: Block) -> bool {
        // If the block is solid, all sides adjacent to transparent or invisible blocks are visible
        // If the block is transparent, only sides adjacent to transparent blocks are visible
        match block.render_type() {
            // If block is invisible, its faces are by definition never visible
            BlockRenderType::Invisible => false,
            // If block is opaque: it is visible only if the adjacent block is not solid
            BlockRenderType::Opaque => match adjacent_block.render_type() {
                BlockRenderType::Opaque => false,
                BlockRenderType::Transparent { .. } | BlockRenderType::Invisible => true,
            },
            // If block is transparent ...
            BlockRenderType::Transparent {
                interior_face_culling,
            } => match adjacent_block.render_type() {
                // and adjacent block is solid, don't render
                BlockRenderType::Opaque => false,
                // and adjacent block is also transparent: render if the adjacent block is of a different type
                // or interior faces adjacent to the identical blocks are not configured to be culled
                BlockRenderType::Transparent { .. } => {
                    block != adjacent_block || !interior_face_culling
                }
                // render always if adjacent block is invisible
                BlockRenderType::Invisible => true,
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
                .get_including_padding(air_block + step_0 * cross_directions.0.get_unit_ivec())
                .render_type()
                == BlockRenderType::Opaque;
            let side_2 = self
                .get_including_padding(air_block + step_1 * cross_directions.1.get_unit_ivec())
                .render_type()
                == BlockRenderType::Opaque;

            let corner = self
                .get_including_padding(
                    air_block
                        + step_0 * cross_directions.0.get_unit_ivec()
                        + step_1 * cross_directions.1.get_unit_ivec(),
                )
                .render_type()
                == BlockRenderType::Opaque;

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
