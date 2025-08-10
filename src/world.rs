use std::{collections::HashMap, sync::Arc};

use crate::world::{
    blocks::Block,
    chunk::{ChunkStack, ChunkUVW, ChunkUW, CHUNK_WIDTH_I32, VERTICAL_CHUNK_COUNT},
};
use glam::{ivec3, IVec3, Vec3};
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

    pub fn get_block(&self, block: IVec3) -> Option<Block> {
        let chunk = get_chunk_coordinates_i32(block);

        if chunk.v < 0 || chunk.v as usize >= VERTICAL_CHUNK_COUNT {
            return None;
        }

        let chunk = &self.get_chunk_stack(chunk.to_uw())?.chunks[chunk.v as usize];
        let (x, y, z) = get_inner_chunk_coordinates_i32(block).into();

        Some(chunk.at(x, y, z).to_owned())
    }
}

pub fn get_chunk_coordinates_f32(position: Vec3) -> ChunkUVW {
    get_chunk_coordinates_i32(ivec3(
        position.x.floor() as i32,
        position.y.floor() as i32,
        position.z.floor() as i32,
    ))
}

pub fn get_chunk_coordinates_i32(position: IVec3) -> ChunkUVW {
    ChunkUVW {
        u: position.x.div_euclid(CHUNK_WIDTH_I32),
        v: position.y.div_euclid(CHUNK_WIDTH_I32),
        w: position.z.div_euclid(CHUNK_WIDTH_I32),
    }
}

pub fn get_inner_chunk_coordinates_f32(position: Vec3) -> IVec3 {
    get_inner_chunk_coordinates_i32(ivec3(
        position.x as i32,
        position.y as i32,
        position.z as i32,
    ))
}

pub fn get_inner_chunk_coordinates_i32(position: IVec3) -> IVec3 {
    position.rem_euclid(ivec3(CHUNK_WIDTH_I32, CHUNK_WIDTH_I32, CHUNK_WIDTH_I32))
}
