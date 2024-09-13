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
}

impl Block {
    pub fn texture_index(&self) -> u32 {
        match self {
            Block::AIR => panic!("{:?} doesn't feature a texture", self),
            Block::STONE => 0,
            Block::GRASS => 1,
            Block::DIRT => 2,
            Block::SAND => 3,
            Block::GRAVEL => 4,
            Block::ANDESITE => 5,
            Block::SNOW => 6,
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
