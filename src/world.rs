use std::{
    array,
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
};

use crate::world::{
    blocks::{Block, BlockPhysicsType},
    chunk::{ArcChunkStack, CHUNK_WIDTH_I32, Chunk, ChunkStack, ChunkUVW, ChunkUW},
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
    chunks: HashMap<ChunkUVW, Arc<RwLock<Chunk>>>,
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

    pub fn get_chunk(&self, uvw: ChunkUVW) -> Option<Arc<RwLock<Chunk>>> {
        self.chunks.get(&uvw).map(Arc::clone)
    }

    // TODO remove if possible
    pub fn get_chunk_stack(&self, uw: ChunkUW) -> Option<ArcChunkStack> {
        if !self.chunk_stacks.contains(&uw) {
            return None;
        }

        Some(ArcChunkStack {
            uw: uw,
            chunks: array::from_fn(|v| self.get_chunk(uw.to_uvw(v as i32)).unwrap()),
        })
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

            self.chunks
                .insert(uw.to_uvw(v as i32), Arc::new(RwLock::new(chunk)));
        }
    }

    /// Replace the block at the given coordinates, returning all all chunks that need to be reloaded.
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
        // todo remove unwrap, might fail when neighboring chunks not generated
        *self
            .get_chunk(uvw)
            .unwrap()
            .write()
            .unwrap()
            .at_mut_with_padding(inner_chunk_coords) = block;
    }
}

impl LookupBlock for World {
    fn lookup_block(&self, coords: IVec3) -> Option<Block> {
        let optional_chunk = self.get_chunk(get_chunk_coordinates(coords));

        // todo remove unwrap
        optional_chunk.map(|chunk| {
            chunk
                .read()
                .unwrap()
                .at(get_inner_chunk_coordinates(coords))
                .to_owned()
        })
    }
}

pub fn get_chunk_coordinates(position: IVec3) -> ChunkUVW {
    position.div_euclid(IVec3::splat(CHUNK_WIDTH_I32)).into()
}

pub fn get_inner_chunk_coordinates(position: IVec3) -> IVec3 {
    position.rem_euclid(IVec3::splat(CHUNK_WIDTH_I32))
}
