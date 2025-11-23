use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::world::{
    blocks::{Block, BlockPhysicsType},
    chunk::{CHUNK_WIDTH_I32, Chunk, ChunkStack, ChunkUVW, ChunkUW},
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
    /// Invariant: If this HashMap contains chunks a chunk, it always contains all chunks of the same stack.
    chunks: HashMap<ChunkUVW, Arc<Chunk>>,
    chunk_stacks: HashSet<ChunkUW>,
}

impl World {
    pub fn new(seed: u32) -> Self {
        World {
            noise: Simplex::new(seed),
            chunks: HashMap::new(),
            chunk_stacks: HashSet::new(),
        }
    }

    pub fn get_chunk(&self, uvw: ChunkUVW) -> Option<Arc<Chunk>> {
        self.chunks.get(&uvw).map(Arc::clone)
    }

    pub fn insert_chunk_stack(&mut self, chunk_stack: ChunkStack) {
        let uw = chunk_stack.uw;

        if !self.chunk_stacks.insert(uw) {
            panic!("Chunk stack at {:?} already loaded", uw);
        }
        for (v, chunk) in chunk_stack.chunks.into_iter().enumerate() {
            if self.chunks.contains_key(&uw.to_uvw(v as i32)) {
                panic!("Attempted to overwrite previously loaded chunk");
            }

            self.chunks.insert(uw.to_uvw(v as i32), Arc::new(chunk));
        }
    }

    /// Replace the block at the given coordinates.
    /// The function returns all chunks that need to be reloaded.
    /// This can be more than one chunk if the replaced block is adjacent to blocks in another chunk and their visibility or AO values change.
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

            if let Some(chunk) = self.get_chunk(uvw) {
                let old_block = chunk.set_including_padding(inner_chunk_coords, block);
                if old_block.render_type() != block.render_type() {
                    // Only note chunk as updated if its mesh is potentially affected
                    updated_chunks.push(uvw);
                }
            }
        }

        updated_chunks
    }
}

impl LookupBlock for World {
    fn lookup_block(&self, coords: IVec3) -> Option<Block> {
        let optional_chunk = self.get_chunk(get_chunk_coordinates(coords));

        let Some(chunk) = optional_chunk else {
            return None;
        };

        Some(chunk.get(get_inner_chunk_coordinates(coords)))
    }
}

pub fn get_chunk_coordinates(position: IVec3) -> ChunkUVW {
    position.div_euclid(IVec3::splat(CHUNK_WIDTH_I32)).into()
}

pub fn get_inner_chunk_coordinates(position: IVec3) -> IVec3 {
    position.rem_euclid(IVec3::splat(CHUNK_WIDTH_I32))
}
