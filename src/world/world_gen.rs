use std::sync::Arc;

use glam::{IVec2, IVec3, Vec3Swizzles, ivec3, usizevec2};
use itertools::Itertools;
use noise::{Fbm, MultiFractal, NoiseFn, Power, ScaleBias, ScalePoint, Simplex};
use rand_pcg::{Pcg64Mcg, rand_core::RngCore};
use serde::Deserialize;

use crate::{
    math::nd_array::{HyperCubeArray, ShiftedHyperCubeArray},
    world::{
        self,
        blocks::Block,
        chunk::{CHUNK_WIDTH, CHUNK_WIDTH_I32, ChunkStack, ChunkUW, WORLD_HEIGHT},
    },
};

#[derive(Deserialize)]
pub struct WorldGenSettings {
    global_seed: u64,
    min_height: i32,
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
) -> HyperCubeArray<2, CHUNK_WIDTH, i32> {
    let mut height_map = HyperCubeArray::default();
    let noise = create_noise(settings);

    for x in 0..CHUNK_WIDTH {
        for z in 0..CHUNK_WIDTH {
            let y = noise.get([
                uw.u as f64 + (x as f64 / CHUNK_WIDTH as f64),
                uw.w as f64 + (z as f64 / CHUNK_WIDTH as f64),
            ]);

            // Normalize from [-max_amplitude, +max_amplitude] to [0, 1]
            let y_norm = (y + 1.0) / 2.0;

            // Multiply from [0f64, 1f64] to [0usize, WORLD_HEIGHT - 1]
            let height = i32::clamp(
                (y_norm.powf(settings.fbm.power) * WORLD_HEIGHT as f64) as i32,
                1,
                WORLD_HEIGHT as i32 - 1,
            );

            height_map[usizevec2(x, z)] = height;
        }
    }

    height_map
}

pub struct ChunkGenResult {
    pub chunk_stack: Arc<ChunkStack>,
    pub adjacent_chunk_changes: ShiftedHyperCubeArray<2, 3, Option<Vec<(IVec3, Block)>>>,
}

impl ChunkGenResult {
    fn new(uw: ChunkUW) -> Self {
        Self {
            chunk_stack: Arc::new(ChunkStack::empty(uw)),
            adjacent_chunk_changes: ShiftedHyperCubeArray::default([-1; 2]),
        }
    }

    fn set_block(&mut self, pos: IVec3, block: Block) {
        let (uvw_centered, inner) = world::divide_world_coordinates(pos);
        let offset = IVec3::from(uvw_centered);
        if offset.xz() == IVec2::ZERO {
            self.chunk_stack
                .get_chunk(uvw_centered.v)
                .unwrap()
                .set(inner, block);
        } else {
            // TODO prettier
            self.adjacent_chunk_changes[[offset.x as isize, offset.z as isize]]
                .get_or_insert(Vec::new())
                .push((inner.as_ivec3().with_y(pos.y), block));
        }
    }

    fn get_chunk_stack_block(&mut self, pos: IVec3) -> Block {
        let (uvw_centered, inner) = world::divide_world_coordinates(pos);
        if IVec2::from(uvw_centered.to_uw()) != IVec2::ZERO {
            // TODO should return something else
            return Block::Air;
        }
        self.chunk_stack
            .get_chunk(uvw_centered.v)
            .unwrap()
            .get(inner)
    }
}

enum TreeType {
    Oak,
    Spruce,
}

pub fn generate(settings: &WorldGenSettings, uw: ChunkUW) -> ChunkGenResult {
    let height_map = sample_height_map(settings, uw);
    let mut output = ChunkGenResult::new(uw);

    // Rng with unique seed over global seed and chunk uw
    let mut rng = create_chunk_rng(settings.global_seed, uw);

    for x in 0..CHUNK_WIDTH {
        for z in 0..CHUNK_WIDTH {
            let height = height_map[usizevec2(x, z)] as i32;

            let mut block_array = Vec::new();
            block_array.push((settings.min_height..height - 1, Block::Stone));
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
                    output.set_block(ivec3(x as i32, y, z as i32), block);
                }
            }
        }
    }

    for _ in 0..rng.next_u32() % 16 {
        let x = rng.next_u32() as usize % CHUNK_WIDTH;
        let z = rng.next_u32() as usize % CHUNK_WIDTH;

        let height = height_map[usizevec2(x, z)];

        if output.get_chunk_stack_block(ivec3(x as i32, height, z as i32)) != Block::Grass {
            continue;
        }

        insert_tree(
            &mut output,
            x,
            height,
            z,
            if rng.next_u32() & 1 == 0 {
                TreeType::Oak
            } else {
                TreeType::Spruce
            },
        );
    }

    output
}

fn insert_tree(chunk_gen: &mut ChunkGenResult, x: usize, y: i32, z: usize, tree: TreeType) {
    let (log, leaves) = match tree {
        TreeType::Oak => (Block::LogOak, Block::LeavesOak),
        TreeType::Spruce => (Block::LogSpruce, Block::LeavesSpruce),
    };

    for y in (y + 1)..=i32::min(y + 5, WORLD_HEIGHT as i32 - 1) {
        chunk_gen.set_block(ivec3(x as i32, y, z as i32), log);
    }

    // todo this causes overflows
    for ((x_leaves, z_leaves), y_leaves) in ((x as i32 - 2)..=(x as i32 + 2))
        .cartesian_product((z as i32 - 2)..=(z as i32 + 2))
        .cartesian_product((y + 3)..=y + 6)
    {
        // Skip log blocks
        if x_leaves == x as i32 && z_leaves == z as i32 && y_leaves <= y + 5 {
            continue;
        }
        if (-CHUNK_WIDTH_I32..(2 * CHUNK_WIDTH_I32)).contains(&x_leaves)
            && (0..WORLD_HEIGHT as i32 - 1).contains(&y_leaves)
            && (-CHUNK_WIDTH_I32..(2 * CHUNK_WIDTH_I32)).contains(&z_leaves)
            && chunk_gen.get_chunk_stack_block(ivec3(x_leaves, y_leaves, z_leaves)) == Block::Air
        {
            chunk_gen.set_block(ivec3(x_leaves, y_leaves, z_leaves), leaves);
        }
    }
}
