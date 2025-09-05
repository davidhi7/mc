use glam::IVec3;

use crate::{
    math::ray_caster::{self, RaycastHit, RaycastStatus},
    world::{
        blocks::{Block, BlockType, Direction},
        camera::CameraController,
        World,
    },
};

const FOCUS_DISTANCE: f32 = 10.0;

#[derive(Clone, Copy, Debug)]
pub struct BlockInfo {
    coords: IVec3,
    block: Block,
    face: Option<Direction>,
}

#[derive(Clone, Copy, Debug)]
pub struct LookedAtBlockResult {
    solid_block: Option<BlockInfo>,
    liquid_block: Option<BlockInfo>,
}

pub fn find_looked_at_blocks(camera: &CameraController, world: &World) -> LookedAtBlockResult {
    let mut focused_blocks = LookedAtBlockResult {
        solid_block: None,
        liquid_block: None,
    };

    // TODO handle blocks inside camera
    ray_caster::cast_ray(
        camera.view.eye,
        camera.view.direction,
        FOCUS_DISTANCE,
        |RaycastHit {
             voxel,
             voxel_face: direction,
             t,
         }| {
            if let Some(block) = world.get_block(voxel) {
                match block.get_block_type() {
                    BlockType::OPAQUE => {
                        focused_blocks.solid_block = Some(BlockInfo {
                            coords: voxel,
                            block,
                            face: Some(direction),
                        });
                        RaycastStatus::Stop
                    }
                    BlockType::TRANSPARENT => {
                        if focused_blocks.liquid_block.is_none() {
                            focused_blocks.liquid_block = Some(BlockInfo {
                                coords: voxel,
                                block,
                                face: Some(direction),
                            });
                        }
                        RaycastStatus::Continue
                    }
                    BlockType::INVISIBLE => RaycastStatus::Continue,
                }
            } else {
                RaycastStatus::Continue
            }
        },
    );

    focused_blocks
}
