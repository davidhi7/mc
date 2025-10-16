use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use crate::world::{
    blocks::{Block, BlockPhysicsType},
    chunk::{CHUNK_WIDTH_I32, ChunkStack, ChunkUVW, ChunkUW, VERTICAL_CHUNK_COUNT},
};
use glam::IVec3;
use itertools::Itertools;
use noise::Simplex;

pub mod blocks;
pub mod chunk;
pub mod world_loader;

pub trait LookupBlock {
    fn lookup_block(&self, block: IVec3) -> Option<Block>;

    fn is_solid(&self, block: IVec3) -> bool {
        self.lookup_block(block)
            .is_some_and(|block| matches!(block.physics_type(), BlockPhysicsType::SOLID))
    }

    fn is_liquid(&self, block: IVec3) -> bool {
        self.lookup_block(block)
            .is_some_and(|block| matches!(block.physics_type(), BlockPhysicsType::LIQUID))
    }
}

pub struct World {
    noise: Simplex,
    chunk_stacks: HashMap<ChunkUW, Arc<RwLock<ChunkStack>>>,
}

impl World {
    pub fn new(seed: u32) -> Self {
        World {
            noise: Simplex::new(seed),
            chunk_stacks: HashMap::new(),
        }
    }

    pub fn get_chunk_stack(&self, uw: ChunkUW) -> Option<Arc<RwLock<ChunkStack>>> {
        self.chunk_stacks.get(&uw).map(Arc::clone)
    }

    pub fn insert_chunks(&mut self, uw: ChunkUW, chunks: Arc<RwLock<ChunkStack>>) {
        if self.chunk_stacks.contains_key(&uw) {
            panic!("Chunks at [u={}, w={}] already generated", uw.u, uw.w);
        }

        self.chunk_stacks.insert(uw.to_owned(), chunks);
    }

    pub fn replace_block(&mut self, coords: IVec3, block: Block) -> Vec<ChunkUVW> {
        let uvw = get_chunk_coordinates(coords);
        if !ChunkStack::validate_chunk_v(uvw.v) {
            return Vec::new();
        }

        let mut updated_chunks = vec![];
        let inner_chunk_coords = get_inner_chunk_coordinates(coords);

        // Find all chunks sharing a chunk cube face with the chunk inside which the block will be replaced.
        // These neighboring chunks are represented by the vector to be added to the origin chunk's uvw coordinates.
        // Using the powerset of this set later, we also represent chunks that only share an edge or single point.
        let mut neighboring_chunks = Vec::with_capacity(3);
        if inner_chunk_coords.x == 0 {
            neighboring_chunks.push(IVec3::NEG_X);
        } else if inner_chunk_coords.x == CHUNK_WIDTH_I32 - 1 {
            neighboring_chunks.push(IVec3::X);
        };

        if inner_chunk_coords.y == 0 {
            neighboring_chunks.push(IVec3::NEG_Y);
        } else if inner_chunk_coords.y == CHUNK_WIDTH_I32 - 1 {
            neighboring_chunks.push(IVec3::Y);
        };

        if inner_chunk_coords.z == 0 {
            neighboring_chunks.push(IVec3::NEG_Z);
        } else if inner_chunk_coords.z == CHUNK_WIDTH_I32 - 1 {
            neighboring_chunks.push(IVec3::Z);
        };

        // The power set always contains the empty set, so this loop also always considers the empty directions vector
        for directions in neighboring_chunks.iter().powerset() {
            let chunk_offset: IVec3 = directions.iter().map(|vec| **vec).sum();
            let uvw: ChunkUVW = (IVec3::from(uvw) + chunk_offset).into();

            if !ChunkStack::validate_chunk_v(uvw.v) {
                continue;
            }

            let mut inner_chunk_coords = inner_chunk_coords;
            for direction in directions.into_iter() {
                match *direction {
                    IVec3::NEG_X => inner_chunk_coords.x = CHUNK_WIDTH_I32,
                    IVec3::X => inner_chunk_coords.x = -1,
                    IVec3::NEG_Y => inner_chunk_coords.y = CHUNK_WIDTH_I32,
                    IVec3::Y => inner_chunk_coords.y = -1,
                    IVec3::NEG_Z => inner_chunk_coords.z = CHUNK_WIDTH_I32,
                    IVec3::Z => inner_chunk_coords.z = -1,
                    _ => unreachable!(),
                };
            }

            self.set_block_with_padding(uvw, inner_chunk_coords, block);

            updated_chunks.push(uvw);
        }

        updated_chunks
    }

    fn set_block_with_padding(&mut self, uvw: ChunkUVW, inner_chunk_coords: IVec3, block: Block) {
        let tmp = self.get_chunk_stack(uvw.to_uw()).unwrap();
        *tmp.write().unwrap().chunks[uvw.v as usize].at_mut_with_padding(inner_chunk_coords) =
            block;
    }
}

impl LookupBlock for World {
    fn lookup_block(&self, coords: IVec3) -> Option<Block> {
        let chunk = get_chunk_coordinates(coords);

        if chunk.v < 0 || chunk.v as usize >= VERTICAL_CHUNK_COUNT {
            return None;
        }

        let chunk_stack = self.get_chunk_stack(chunk.to_uw())?;
        let binding = chunk_stack.read().unwrap();
        let chunk = binding.chunks.get(chunk.v as usize)?;

        Some(chunk.at(get_inner_chunk_coordinates(coords)).to_owned())
    }
}

pub fn get_chunk_coordinates(position: IVec3) -> ChunkUVW {
    position.div_euclid(IVec3::splat(CHUNK_WIDTH_I32)).into()
}

pub fn get_inner_chunk_coordinates(position: IVec3) -> IVec3 {
    position.rem_euclid(IVec3::splat(CHUNK_WIDTH_I32))
}
