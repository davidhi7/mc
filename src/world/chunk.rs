use std::{
    array, mem,
    ops::{Index, IndexMut},
    sync::{Arc, RwLock},
};

use glam::{IVec2, IVec3, USizeVec3, ivec2, ivec3};

use crate::{
    math::nd_array::{HyperCubeArray, ShiftedHyperCubeArray},
    renderer::vertex_buffer::{QuadInstance, TransparentQuadInstance},
    world::{
        self, World,
        blocks::{Block, BlockRenderType, Direction},
    },
};

pub const CHUNK_WIDTH_BITS: u32 = 5;
pub const CHUNK_WIDTH: usize = 2_usize.pow(CHUNK_WIDTH_BITS);
pub const CHUNK_WIDTH_I32: i32 = CHUNK_WIDTH as i32;

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
    uw: ChunkUW,
    chunks: [Chunk; VERTICAL_CHUNK_COUNT],
}

impl ChunkStack {
    pub fn empty(uw: ChunkUW) -> Self {
        Self {
            uw,
            chunks: array::from_fn(|v| Chunk::empty(uw.to_uvw(v as i32))),
        }
    }

    pub fn uw(&self) -> ChunkUW {
        self.uw
    }

    pub fn get_chunk(&self, v: i32) -> Option<&Chunk> {
        if !Self::validate_chunk_v(v) {
            return None;
        }

        Some(&self.chunks[v as usize])
    }

    pub fn validate_chunk_v(v: i32) -> bool {
        (0..VERTICAL_CHUNK_COUNT as i32).contains(&v)
    }
}

// TODO indexmut?
#[derive(Clone)]
pub struct ChunkMeshingContext {
    pub neighbors: ShiftedHyperCubeArray<2, 3, Arc<ChunkStack>>,
}

impl ChunkMeshingContext {
    // TODO not panic
    pub fn create(world: &World, uw: ChunkUW) -> Self {
        // array is shifted so that center chunk is in the center, not the
        let neighbors =
            ShiftedHyperCubeArray::from_fn([uw.u as isize - 1, uw.w as isize - 1], |uw| {
                Arc::clone(
                    world
                        .get_chunk_stack(ChunkUW {
                            u: uw[0] as i32,
                            w: uw[1] as i32,
                        })
                        .unwrap(),
                )
            });

        Self { neighbors }
    }

    fn get(&self, pos: IVec3) -> Option<Block> {
        let (uvw, inner_chunk_coords) = world::divide_world_coordinates(pos);
        let uw = uvw.to_uw();

        let chunk_stack = &self.neighbors[[uw.u as isize, uw.w as isize]];
        chunk_stack
            .get_chunk(uvw.v)
            .map(|chunk| chunk.get(inner_chunk_coords))
    }

    pub fn generate_mesh(&self, uvw: ChunkUVW) -> (Vec<u8>, Vec<u8>) {
        let mut solid_instances = Vec::new();
        let mut transparent_instances = Vec::new();

        for x in 0..CHUNK_WIDTH_I32 {
            for y in 0..CHUNK_WIDTH_I32 {
                for z in 0..CHUNK_WIDTH_I32 {
                    let coords = IVec3::from(uvw) * CHUNK_WIDTH_I32 + ivec3(x, y, z);
                    let block = self.get(coords).unwrap();
                    if let BlockRenderType::Invisible = block.render_type() {
                        continue;
                    }

                    let common_packed_bits: u32 = x as u32
                        | ((y as u32) << CHUNK_WIDTH_BITS)
                        | ((z as u32) << (CHUNK_WIDTH_BITS * 2));

                    for direction in Direction::iter() {
                        if let Some(adjacent_block) = self.get(coords + direction.get_unit_ivec())
                            && !Chunk::is_face_visible(block, adjacent_block)
                        {
                            continue;
                        }

                        let attributes = common_packed_bits
                            | ((block.texture_index(direction) as u32) << (CHUNK_WIDTH_BITS * 3))
                            | ((direction as u32) << (CHUNK_WIDTH_BITS * 3 + 8));

                        match block {
                            Block::Water => {
                                transparent_instances.extend_from_slice(bytemuck::bytes_of(
                                    &TransparentQuadInstance { attributes },
                                ));
                            }
                            Block::Air => unreachable!(),
                            _ => {
                                solid_instances.extend_from_slice(bytemuck::bytes_of(
                                    &QuadInstance {
                                        attributes,
                                        ao_attributes: self.get_ao_attributes(coords, direction),
                                    },
                                ));
                            }
                        };
                    }
                }
            }
        }

        (solid_instances, transparent_instances)
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

            // get(pos) should only return None if the block is vertically outside of the allowed block range, in that case we assume its air
            let side_1 = self
                .get(air_block + step_0 * cross_directions.0.get_unit_ivec())
                .unwrap_or(Block::Air)
                .render_type()
                == BlockRenderType::Opaque;

            let side_2 = self
                .get(air_block + step_1 * cross_directions.1.get_unit_ivec())
                .unwrap_or(Block::Air)
                .render_type()
                == BlockRenderType::Opaque;

            let corner = self
                .get(
                    air_block
                        + step_0 * cross_directions.0.get_unit_ivec()
                        + step_1 * cross_directions.1.get_unit_ivec(),
                )
                .unwrap_or(Block::Air)
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

pub struct Chunk {
    uvw: ChunkUVW,
    data: RwLock<HyperCubeArray<3, CHUNK_WIDTH, Block>>,
}

impl Chunk {
    pub fn empty(uvw: ChunkUVW) -> Self {
        Chunk {
            uvw,
            data: RwLock::new(HyperCubeArray::default()),
        }
    }

    #[expect(dead_code)]
    pub fn uvw(&self) -> ChunkUVW {
        self.uvw
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

    pub fn get(&self, index: USizeVec3) -> Block {
        *self.data.read().unwrap().index(index)
    }

    pub fn set(&self, index: USizeVec3, block: Block) -> Block {
        mem::replace(self.data.write().unwrap().index_mut(index), block)
    }
}
