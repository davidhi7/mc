use glam::{IVec3, Vec3};

use crate::{
    math::ray_caster::{self, RaycastHit, RaycastStatus},
    world::{
        LookupBlock,
        blocks::{Block, BlockPhysicsType, Direction},
    },
};

const FOCUS_DISTANCE: f32 = 10.0;

#[derive(Clone, Copy, Debug)]
pub struct BlockInfo {
    pub coords: IVec3,
    pub block: Block,
    pub face: Option<Direction>,
}

#[derive(Clone, Copy, Debug)]
pub struct LookedAtBlockResult {
    pub solid_block: Option<BlockInfo>,
    pub liquid_block: Option<BlockInfo>,
}

pub fn find_looked_at_blocks(
    eye: Vec3,
    direction: Vec3,
    block_lookup: &impl LookupBlock,
) -> LookedAtBlockResult {
    let mut focused_blocks = LookedAtBlockResult {
        solid_block: None,
        liquid_block: None,
    };

    // TODO handle blocks inside camera
    ray_caster::cast_ray(
        eye,
        direction,
        FOCUS_DISTANCE,
        |RaycastHit {
             voxel,
             voxel_face: direction,
             ..
         }| {
            if let Some(block) = block_lookup.lookup_block(voxel) {
                match block.physics_type() {
                    BlockPhysicsType::SOLID => {
                        focused_blocks.solid_block = Some(BlockInfo {
                            coords: voxel,
                            block,
                            face: direction,
                        });
                        RaycastStatus::Stop
                    }
                    BlockPhysicsType::LIQUID => {
                        if focused_blocks.liquid_block.is_none() {
                            focused_blocks.liquid_block = Some(BlockInfo {
                                coords: voxel,
                                block,
                                face: direction,
                            });
                        }
                        RaycastStatus::Continue
                    }
                    BlockPhysicsType::GASEOUS => RaycastStatus::Continue,
                }
            } else {
                RaycastStatus::Continue
            }
        },
    );

    focused_blocks
}
