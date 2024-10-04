use std::{collections::HashMap, time::Instant};

use noise::Simplex;

use crate::world::{chunk::Chunk, renderer::CubeFaceInstance};

pub mod blocks;
pub mod camera;
pub mod chunk;
pub mod renderer;

const CHUNK_WIDTH_BITS: u32 = 5;
const CHUNK_DIMENSIONS: i32 = 2_i32.pow(CHUNK_WIDTH_BITS);
const WORLD_HEIGHT: i32 = 256;
const VERTICAL_CHUNK_COUNT: usize = (WORLD_HEIGHT / CHUNK_DIMENSIONS) as usize;

pub struct World {
    noise: Simplex,
    pub chunk_columns: HashMap<(i32, i32), [Chunk; VERTICAL_CHUNK_COUNT]>,
    pub meshed_chunks: HashMap<(i32, i32, i32), Vec<CubeFaceInstance>>,
}

impl World {
    pub fn new(seed: u32) -> Self {
        World {
            noise: Simplex::new(seed),
            chunk_columns: HashMap::new(),
            meshed_chunks: HashMap::new(),
        }
    }

    pub fn create_chunks(&mut self, u: i32, w: i32) {
        let start_instant = Instant::now();

        if self.chunk_columns.contains_key(&(u, w)) {
            panic!("Chunks at [u={}, w={}] already generated", u, w);
        }

        let chunk_column = Chunk::generate_stack(&self.noise, u, w);
        self.chunk_columns
            .insert((u, w), Chunk::generate_stack(&self.noise, u, w));
        for chunk in chunk_column {
            self.meshed_chunks
                .insert((chunk.u, chunk.v, chunk.w), chunk.generate_mesh());
        }

        println!(
            "Generating chunks at [u={}, w={}] took {}ms",
            u,
            w,
            start_instant.elapsed().as_millis()
        );
    }
}
