use glam::{IVec3, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockPhysicsType {
    Solid,
    Liquid,
    Gaseous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockRenderType {
    Opaque,
    Transparent,
    Invisible,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Block {
    Air,
    Stone,
    Grass,
    Dirt,
    Sand,
    Gravel,
    Andesite,
    Snow,
    Water,
}

impl Block {
    pub fn texture_index(&self) -> u8 {
        match self {
            Block::Air => panic!("{:?} doesn't feature a texture", self),
            Block::Stone => 0,
            Block::Grass => 1,
            Block::Dirt => 2,
            Block::Sand => 3,
            Block::Gravel => 4,
            Block::Andesite => 5,
            Block::Snow => 6,
            Block::Water => 6,
        }
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
            Block::Water => BlockRenderType::Transparent,
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
