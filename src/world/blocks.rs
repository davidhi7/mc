use std::array::IntoIter;

#[derive(Debug, Clone, Copy)]
pub enum BlockType {
    SOLID,
    INVISIBLE,
    TRANSPARENT,
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
}
