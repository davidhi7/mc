use std::{collections::HashMap, sync::Arc};

use crate::world::chunk::{ChunkStack, ChunkUVW, ChunkUW, CHUNK_WIDTH_I32};
use glam::Vec3;
use noise::Simplex;

pub mod blocks;
pub mod camera;
pub mod chunk;
pub mod coordinates;
pub mod world_loader;

pub struct World {
    noise: Simplex,
    chunk_stacks: HashMap<ChunkUW, Arc<ChunkStack>>,
}

impl World {
    pub fn new(seed: u32) -> Self {
        World {
            noise: Simplex::new(seed),
            chunk_stacks: HashMap::new(),
        }
    }

    pub fn get_chunk_stack(&self, uw: ChunkUW) -> Option<Arc<ChunkStack>> {
        self.chunk_stacks.get(&uw).map(Arc::clone)
    }

    pub fn insert_chunks(&mut self, uw: ChunkUW, chunks: Arc<ChunkStack>) {
        if self.chunk_stacks.contains_key(&uw) {
            panic!("Chunks at [u={}, w={}] already generated", uw.u, uw.w);
        }

        self.chunk_stacks.insert(uw.to_owned(), chunks);
    }
}

pub fn get_chunk_coordinates(position: Vec3) -> ChunkUVW {
    ChunkUVW {
        u: (position.x as i32).div_euclid(CHUNK_WIDTH_I32),
        v: (position.y as i32).div_euclid(CHUNK_WIDTH_I32),
        w: (position.z as i32).div_euclid(CHUNK_WIDTH_I32),
    }
}
