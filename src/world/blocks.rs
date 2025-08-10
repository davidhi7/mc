use std::array::IntoIter;

use glam::{IVec3, Vec3};

#[derive(Debug, Clone, Copy)]
pub enum BlockType {
    SOLID,
    TRANSPARENT,
    INVISIBLE,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum Block {
    AIR,
    STONE,
    GRASS,
    DIRT,
    SAND,
    GRAVEL,
    ANDESITE,
    SNOW,
    WATER,
}

impl Block {
    pub fn texture_index(&self) -> u8 {
        match self {
            Block::AIR => panic!("{:?} doesn't feature a texture", self),
            Block::STONE => 0,
            Block::GRASS => 1,
            Block::DIRT => 2,
            Block::SAND => 3,
            Block::GRAVEL => 4,
            Block::ANDESITE => 5,
            Block::SNOW => 6,
            Block::WATER => 6,
        }
    }

    pub fn get_block_type(&self) -> BlockType {
        match self {
            Block::AIR => BlockType::INVISIBLE,
            Block::WATER => BlockType::TRANSPARENT,
            _ => BlockType::SOLID,
        }
    }

    pub fn is_solid(&self) -> bool {
        match self.get_block_type() {
            BlockType::SOLID => true,
            _ => false,
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
    pub fn into_iter() -> IntoIter<Direction, 6> {
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

    pub fn get_unit_vector(&self) -> Vec3 {
        match self {
            Direction::NegX => Vec3::NEG_X,
            Direction::X => Vec3::X,
            Direction::NegY => Vec3::NEG_Y,
            Direction::Y => Vec3::Y,
            Direction::NegZ => Vec3::NEG_Z,
            Direction::Z => Vec3::Z,
        }
    }

    pub fn get_unit_vector_i32(&self) -> IVec3 {
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
