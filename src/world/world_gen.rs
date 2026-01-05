use std::array;

use glam::{ivec2, ivec3, usizevec2};
use itertools::Itertools;
use noise::{Fbm, MultiFractal, NoiseFn, Power, ScaleBias, ScalePoint, Simplex};
use rand_pcg::{Pcg64Mcg, rand_core::RngCore};
use serde::Deserialize;

use crate::{
    math::nd_array::HyperCubeArray,
    world::{
        blocks::Block,
        chunk::{
            CHUNK_WIDTH, CHUNK_WIDTH_I32, CHUNK_WIDTH_P, Chunk, ChunkStack, ChunkUW,
            VERTICAL_CHUNK_COUNT, WORLD_HEIGHT,
        },
    },
};

#[derive(Deserialize)]
pub struct WorldGenSettings {
    global_seed: u64,
    sea_level: i32,
    fbm: FbmSettings,
}

#[derive(Deserialize)]
pub struct FbmSettings {
    octaves: u32,
    frequency: f64,
    lacunarity: f64,
    persistance: f64,
    power: f64,
}

fn create_noise(settings: &WorldGenSettings) -> impl NoiseFn<f64, 2> {
    // TODO analyze value ranges of the noises
    // seed: xor between bits 0-31 and 32-63
    let seed = (settings.global_seed as u32) ^ ((settings.global_seed >> 32) as u32);
    let fbm = Fbm::<Simplex>::new(seed)
        .set_octaves(settings.fbm.octaves as usize)
        .set_frequency(settings.fbm.frequency)
        .set_lacunarity(settings.fbm.lacunarity)
        .set_persistence(settings.fbm.persistance);

    Power::new(
        fbm,
        ScaleBias::new(ScalePoint::new(Simplex::new(seed)).set_all_scales(0.01, 0.01, 0.01, 0.01))
            .set_bias(0.5), // .set_scale(0.5),
    )
}

fn create_chunk_rng(seed: u64, uw: ChunkUW) -> Pcg64Mcg {
    let mut chunk_seed = (seed as u128) << 64;
    chunk_seed |= ((uw.u as u128) << 32) | (uw.w as u128);
    Pcg64Mcg::new(chunk_seed)
}

fn sample_height_map(
    settings: &WorldGenSettings,
    uw: ChunkUW,
) -> HyperCubeArray<2, CHUNK_WIDTH_P, u32> {
    let mut heightmap = HyperCubeArray::default();
    let noise = create_noise(settings);

    for x in (-1)..=CHUNK_WIDTH_I32 {
        for z in (-1)..=CHUNK_WIDTH_I32 {
            let y = noise.get([
                uw.u as f64 + (x as f64 / CHUNK_WIDTH as f64),
                uw.w as f64 + (z as f64 / CHUNK_WIDTH as f64),
            ]);

            // Normalize from [-max_amplitude, +max_amplitude] to [0, 1]
            let y_norm = (y + 1.0) / 2.0;

            // Multiply from [0f64, 1f64] to [0usize, WORLD_HEIGHT - 1]
            let height = u32::clamp(
                (y_norm.powf(settings.fbm.power) * WORLD_HEIGHT as f64) as u32,
                1,
                WORLD_HEIGHT as u32 - 1,
            ) as usize;

            heightmap[ivec2(x + 1, z + 1).as_usizevec2()] = height as u32;
        }
    }

    heightmap
}

pub fn generate(settings: &WorldGenSettings, uw: ChunkUW) -> ChunkStack {
    let chunks: [Chunk; VERTICAL_CHUNK_COUNT] =
        array::from_fn(|v| Chunk::empty(uw.to_uvw(v as i32)));
    let mut chunk_stack = ChunkStack { uw, chunks };
    let height_map = sample_height_map(settings, uw);

    // Rng with unique seed over global seed and chunk uw
    let mut rng = create_chunk_rng(settings.global_seed, uw);

    for x in (-1)..=CHUNK_WIDTH_I32 {
        for z in (-1)..=CHUNK_WIDTH_I32 {
            let height = height_map[usizevec2(x as usize + 1, z as usize + 1)] as i32;
            let mut block_array = Vec::new();
            block_array.push((0..height - 1, Block::Stone));
            if height <= settings.sea_level {
                block_array.push((height - 1..height + 1, Block::Sand));
                block_array.push((height + 1..settings.sea_level, Block::Water));
            } else {
                // block_array.push((height - 1..height, Block::Dirt));
                block_array.push((height - 1..height, Block::Stone));
                block_array.push((height..height + 1, Block::Grass));
            }

            for (range, block) in block_array {
                for y in range {
                    chunk_stack.insert(ivec3(x, y, z), block);
                }
            }
        }
    }

    for _ in 0..rng.next_u32() % 16 {
        let x = rng.next_u32() % CHUNK_WIDTH as u32;
        let z = rng.next_u32() % CHUNK_WIDTH as u32;

        let height = height_map[usizevec2(x.try_into().unwrap(), z.try_into().unwrap())];

        if chunk_stack.get(ivec3(x as i32, height as i32, z as i32)) != Block::Grass {
            continue;
        }

        insert_tree(
            &mut chunk_stack,
            x as i32,
            height as i32,
            z as i32,
            rng.next_u32() & 1,
        );
    }

    chunk_stack
}

fn insert_tree(chunk_stack: &mut ChunkStack, x: i32, y: i32, z: i32, variant: u32) {
    let (log, leaves) = match variant {
        0 => (Block::LogOak, Block::LeavesOak),
        1 => (Block::LogSpruce, Block::LeavesSpruce),
        _ => unreachable!(),
    };
    for y in (y + 1)..=(y + 5) {
        chunk_stack.insert(ivec3(x, y, z), log);
    }
    for ((x_, z_), y_) in ((x - 2)..=(x + 2))
        .cartesian_product((z - 2)..=(z + 2))
        .cartesian_product((y + 3)..=(y + 6))
    {
        if x_ == x && z_ == z && y_ <= y + 5 {
            continue;
        }
        if (0..CHUNK_WIDTH_I32).contains(&x_)
            && (0..WORLD_HEIGHT as i32).contains(&y_)
            && (0..CHUNK_WIDTH_I32).contains(&z_)
            && chunk_stack.get(ivec3(x_, y_, z_)) == Block::Air
        {
            chunk_stack.insert(ivec3(x_, y_, z_), leaves);
        }
    }
}
