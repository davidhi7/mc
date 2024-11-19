use crate::world::blocks::Direction;

pub struct Coordinate {
    coordinates: [i32; 3],
}

impl Coordinate {
    pub fn new(x: i32, y: i32, z: i32) -> Coordinate {
        Coordinate {
            coordinates: [x, y, z],
        }
    }

    pub fn go(&self, direction: Direction, distance: i32) -> Self {
        let dimension = (direction as usize) / 2;
        let coefficient = if (direction as u8) & 1 == 1 { 1 } else { -1 };

        let mut new_coordinates = self.coordinates.clone();
        new_coordinates[dimension] += distance * coefficient;

        Coordinate {
            coordinates: new_coordinates,
        }
    }

    pub fn x(&self) -> i32 {
        self.coordinates[0]
    }

    pub fn y(&self) -> i32 {
        self.coordinates[1]
    }

    pub fn z(&self) -> i32 {
        self.coordinates[2]
    }
}
