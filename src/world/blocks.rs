use glam::{IVec3, Vec3};

use crate::texture::Texture;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockPhysicsType {
    Solid,
    Liquid,
    Gaseous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockRenderType {
    Opaque,
    Transparent {
        /// If multiple identical blocks are adjacent to each other, this sets whether faces adjacent to the same block are culled.
        /// Faces are never culled when adjacent to a different transparent or any invisible block.
        interior_face_culling: bool,
    },
    Invisible,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Block {
    #[default]
    Air,
    Stone,
    Grass,
    Dirt,
    Sand,
    Gravel,
    Andesite,
    Snow,
    Water,
    LogOak,
    LeavesOak,
    LogSpruce,
    LeavesSpruce,
}

impl Block {
    pub fn texture_index(&self, face: Direction) -> Option<u8> {
        let texture = match self {
            Block::Air | Block::Water => {
                return None;
            }
            Block::Stone => Texture::Stone,
            Block::Grass => match face {
                Direction::Y => Texture::GrassBlockTop,
                Direction::NegY => Texture::Dirt,
                _ => Texture::GrassBlockTop,
            },
            Block::Dirt => Texture::Dirt,
            Block::Sand => Texture::Sand,
            Block::Gravel => Texture::Gravel,
            Block::Andesite => Texture::Andesite,
            Block::Snow => Texture::Snow,
            Block::LogOak => match face {
                Direction::NegY | Direction::Y => Texture::LogOakTopBottom,
                _ => Texture::LogOakSide,
            },
            Block::LogSpruce => match face {
                Direction::NegY | Direction::Y => Texture::LogSpruceTopBottom,
                _ => Texture::LogSpruceSide,
            },
            Block::LeavesOak => Texture::LeavesOak,
            Block::LeavesSpruce => Texture::LeavesSpruce,
        };
        Some(texture as u8)
    }

    pub fn physics_type(&self) -> BlockPhysicsType {
        match self {
            Block::Air => BlockPhysicsType::Gaseous,
            Block::Water => BlockPhysicsType::Liquid,
            _ => BlockPhysicsType::Solid,
        }
    }

    pub fn render_type(&self) -> BlockRenderType {
        match self {
            Block::Air => BlockRenderType::Invisible,
            Block::Water => BlockRenderType::Transparent {
                interior_face_culling: true,
            },
            Block::LeavesOak | Block::LeavesSpruce => BlockRenderType::Transparent {
                interior_face_culling: false,
            },
            _ => BlockRenderType::Opaque,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
#[repr(u8)]
pub enum Direction {
    NegX = 0,
    X = 1,
    NegY = 2,
    Y = 3,
    NegZ = 4,
    Z = 5,
}

impl Direction {
    pub fn iter() -> impl Iterator<Item = Direction> {
        [
            Direction::NegX,
            Direction::X,
            Direction::NegY,
            Direction::Y,
            Direction::NegZ,
            Direction::Z,
        ]
        .into_iter()
    }

    #[expect(dead_code)]
    pub fn get_unit_vec(&self) -> Vec3 {
        match self {
            Direction::NegX => Vec3::NEG_X,
            Direction::X => Vec3::X,
            Direction::NegY => Vec3::NEG_Y,
            Direction::Y => Vec3::Y,
            Direction::NegZ => Vec3::NEG_Z,
            Direction::Z => Vec3::Z,
        }
    }

    pub fn get_unit_ivec(&self) -> IVec3 {
        match self {
            Direction::NegX => IVec3::NEG_X,
            Direction::X => IVec3::X,
            Direction::NegY => IVec3::NEG_Y,
            Direction::Y => IVec3::Y,
            Direction::NegZ => IVec3::NEG_Z,
            Direction::Z => IVec3::Z,
        }
    }
}
