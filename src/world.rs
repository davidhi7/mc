use std::collections::HashMap;

use crate::world::chunk::{ChunkStack, ChunkUVW, ChunkUW, CHUNK_WIDTH_I32};
use glam::Vec3;
use noise::Simplex;

pub mod blocks;
pub mod camera;
pub mod chunk;
pub mod coordinates;
pub mod world_loader;

pub struct World {
    pub noise: Simplex,
    pub chunk_stacks: HashMap<ChunkUW, ChunkStack>,
}

impl World {
    pub fn new(seed: u32) -> Self {
        World {
            noise: Simplex::new(seed),
            chunk_stacks: HashMap::new(),
        }
    }

    pub fn insert_chunks(&mut self, uw: ChunkUW, chunks: ChunkStack) {
        if self.chunk_stacks.contains_key(&uw) {
            panic!("Chunks at [u={}, w={}] already generated", uw.0, uw.1);
        }

        self.chunk_stacks.insert(uw.to_owned(), chunks);
    }
}

pub fn get_chunk_coordinates(position: Vec3) -> ChunkUVW {
    let u = position.x as i32 / CHUNK_WIDTH_I32;
    let v = position.y as i32 / CHUNK_WIDTH_I32;
    let w = position.z as i32 / CHUNK_WIDTH_I32;

    return (u, v, w);
}
