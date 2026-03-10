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
pub struct BlockHitInfo {
    pub coords: IVec3,
    pub block: Block,
    pub face: Option<Direction>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LookedAtBlocks {
    pub solid_block: Option<BlockHitInfo>,
    pub liquid_block: Option<BlockHitInfo>,
}

pub fn find_looked_at_blocks(
    eye: Vec3,
    direction: Vec3,
    block_lookup: &impl LookupBlock,
) -> LookedAtBlocks {
    let mut focused_blocks = LookedAtBlocks {
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
                    BlockPhysicsType::Solid => {
                        focused_blocks.solid_block = Some(BlockHitInfo {
                            coords: voxel,
                            block,
                            face: direction,
                        });
                        RaycastStatus::Stop
                    }
                    BlockPhysicsType::Liquid => {
                        if focused_blocks.liquid_block.is_none() {
                            focused_blocks.liquid_block = Some(BlockHitInfo {
                                coords: voxel,
                                block,
                                face: direction,
                            });
                        }
                        RaycastStatus::Continue
                    }
                    BlockPhysicsType::Gaseous => RaycastStatus::Continue,
                }
            } else {
                RaycastStatus::Continue
            }
        },
    );

    focused_blocks
}
