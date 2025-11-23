use std::collections::HashSet;

use glam::{IVec3, USizeVec3, ivec3};
use itertools::Itertools;

pub struct RollingGrid<T> {
    /// Invariant: width is always an uneven number
    width: usize,
    array: Box<[T]>,
    center: IVec3,
}

impl<T> RollingGrid<T> {
    /// Create new instance with the given width and center point.
    /// For every cell in the grid, the load function is called.
    pub fn new(width: usize, center: IVec3, mut load: impl FnMut(IVec3) -> T) -> Self {
        assert!(width & 1 == 1, "N must be an uneven number");
        let mut vec = Vec::with_capacity(width.pow(3));

        for z in Self::iter_1d(width, center.z) {
            for y in Self::iter_1d(width, center.y) {
                for x in Self::iter_1d(width, center.x) {
                    vec.push(load(ivec3(x, y, z)));
                }
            }
        }

        Self {
            width,
            array: vec.into_boxed_slice(),
            center,
        }
    }

    fn iter_1d(width: usize, center: i32) -> impl Iterator<Item = i32> + Clone {
        let n_half = (width / 2) as i32;
        center - n_half..=center + n_half
    }

    fn iter_2d(
        width: usize,
        center_0: i32,
        center_1: i32,
    ) -> impl Iterator<Item = (i32, i32)> + Clone {
        Self::iter_1d(width, center_0).cartesian_product(Self::iter_1d(width, center_1))
    }

    /// Returns true if the position is within `self.width / 2` from `self.center`.
    fn validate_position(&self, position: IVec3) -> bool {
        usize::try_from((self.center - position).abs().max_element()).unwrap() <= self.width / 2
    }

    /// Compute the `self.grid` index from the given vector.
    /// This function panicks if the position is more than `self.width / 2` away from `self.center`.
    fn position_to_index(&self, position: IVec3) -> usize {
        if !self.validate_position(position) {
            panic!(
                "position vector {} not within grid around {} and width {}",
                position, self.center, self.width
            );
        }

        let USizeVec3 { x, y, z } = position
            .rem_euclid(IVec3::splat(self.width.try_into().unwrap()))
            .as_usizevec3();

        (x * self.width + y) * self.width + z
    }

    /// Get an immutable reference to the grid contents of the given position.
    /// Returns None if the position is not within the grid around the current center.
    pub fn at(&self, position: IVec3) -> Option<&T> {
        if !self.validate_position(position) {
            return None;
        }
        Some(&self.array[self.position_to_index(position)])
    }

    /// Get a mutable reference to the grid contents of the given position.
    /// Returns None if the position is not within the grid around the current center.
    pub fn at_mut(&mut self, position: IVec3) -> Option<&mut T> {
        if !self.validate_position(position) {
            return None;
        }
        Some(&mut self.array[self.position_to_index(position)])
    }

    /// Insert the given value into the grid at the given position, returning the old value.
    /// Returns None and does not store the new value if the position is not within the grid around the current center.
    pub fn replace(&mut self, position: IVec3, new_value: T) -> Option<T> {
        if !self.validate_position(position) {
            return None;
        }
        Some(std::mem::replace(self.at_mut(position).unwrap(), new_value))
    }

    /// Reposition the grid so the given vector becomes the new center.
    /// This necessitates unloading cells around the old, and loading cells around the new center.
    /// For every unloaded and loaded cell, the respective function is called exactly once.
    pub fn reposition(
        &mut self,
        new_center: IVec3,
        mut load: impl FnMut(IVec3) -> T,
        mut unload: impl FnMut(IVec3, T),
    ) {
        if new_center == self.center {
            return;
        }

        // Find number of steps on each axis by which current cells where shifted out of the grid, clamped to grid width.
        // Positive number: shifted to the left, negative number: shifted to the right
        let diff = (new_center - self.center).clamp(
            IVec3::splat(-(self.width as i32)),
            IVec3::splat(self.width as i32),
        );
        let IVec3 {
            x: dx,
            y: dy,
            z: dz,
        } = diff;
        let old_position = self.center;

        let width = self.width;
        let width_i32: i32 = self.width.try_into().unwrap();
        let half_width: i32 = (self.width / 2) as i32;

        let x_range = if dx > 0 {
            (old_position.x - half_width)..(old_position.x - half_width + dx)
        } else {
            (old_position.x + half_width + dx + 1)..(old_position.x + half_width + 1)
        };

        let y_range = if dy > 0 {
            (old_position.y - half_width)..(old_position.y - half_width + dy)
        } else {
            (old_position.y + half_width + dy + 1)..(old_position.y + half_width + 1)
        };

        let z_range = if dz > 0 {
            (old_position.z - half_width)..(old_position.z - half_width + dz)
        } else {
            (old_position.z + half_width + dz + 1)..(old_position.z + half_width + 1)
        };

        let mut updated_points = HashSet::new();
        let mut update_once = |old_cell: IVec3| {
            if updated_points.insert(old_cell) {
                // compute the global coordinates of the new cell from those of the old cell
                let mut new_cell = old_cell;
                while (new_cell.x - new_center.x).abs() > half_width {
                    new_cell.x += dx.signum() * width_i32;
                }
                while (new_cell.y - new_center.y).abs() > half_width {
                    new_cell.y += dy.signum() * width_i32;
                }
                while (new_cell.z - new_center.z).abs() > half_width {
                    new_cell.z += dz.signum() * width_i32;
                }

                // TODO vectorized version fixen
                // let diff_to_point = old_cell - new_center;
                // let steps =
                //     (diff_to_point.abs() - IVec3::splat(half_width)) / width_i32 + IVec3::splat(1);
                // let mask = diff_to_point.abs().cmpgt(IVec3::splat(half_width));

                // let new_cell =
                //     old_cell + IVec3::select(mask, diff.signum() * steps * width_i32, IVec3::ZERO);

                let old_value = std::mem::replace(self.at_mut(old_cell).unwrap(), load(new_cell));
                unload(old_cell, old_value);
            }
        };

        for x in x_range {
            for (y, z) in Self::iter_2d(width, old_position.y, old_position.z) {
                update_once(ivec3(x, y, z));
            }
        }

        for y in y_range {
            for (x, z) in Self::iter_2d(width, old_position.x, old_position.z) {
                update_once(ivec3(x, y, z));
            }
        }

        for z in z_range {
            for (x, y) in Self::iter_2d(width, old_position.x, old_position.y) {
                update_once(ivec3(x, y, z));
            }
        }

        self.center = new_center;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::{IVec3, ivec3};
    use itertools::Itertools;

    use crate::world::world_loader::rolling_grid::RollingGrid;

    impl<T: Default> RollingGrid<T> {
        pub fn new_default(width: usize, position: IVec3) -> Self {
            Self::new(width, position, |_| T::default())
        }

        pub fn reposition_default(&mut self, new_position: IVec3) {
            self.reposition(new_position, |_| T::default(), |_, _| ());
        }
    }

    #[test]
    fn test() {
        let mut grid: RollingGrid<bool> = RollingGrid::new_default(3, IVec3::ZERO);
        *grid.at_mut(IVec3::ZERO).unwrap() = true;
        assert_eq!(grid.at(IVec3::ZERO), Some(&true));
        assert_eq!(grid.at(IVec3::X), Some(&false));

        grid.reposition_default(ivec3(1, 0, 0));
        assert_eq!(grid.at(IVec3::ZERO), Some(&true));
        assert_eq!(grid.at(IVec3::X), Some(&false));

        grid.reposition_default(ivec3(-1, -1, 0));
        assert_eq!(grid.at(IVec3::ZERO), Some(&true));
        assert_eq!(grid.at(IVec3::X), None);

        grid.reposition_default(ivec3(2, 0, -1));
        assert_eq!(grid.at(IVec3::ZERO), None);
        assert_eq!(grid.at(IVec3::X), Some(&false));
    }

    #[test]
    fn test_reposition_arguments() {
        let mut load = HashSet::new();
        let mut unload = HashSet::new();
        let mut grid: RollingGrid<bool> = RollingGrid::new_default(1, IVec3::ZERO);
        grid.reposition(
            ivec3(1, 0, 0),
            |vec| {
                load.insert(vec);
                false
            },
            |vec, _| {
                unload.insert(vec);
            },
        );
        assert_eq!(load, HashSet::from([ivec3(1, 0, 0)]));
        assert_eq!(unload, HashSet::from([ivec3(0, 0, 0)]));

        load.clear();
        unload.clear();

        grid.reposition(
            ivec3(10, -20, 30),
            |vec| load.insert(vec),
            |vec, _| {
                unload.insert(vec);
            },
        );
        assert_eq!(load, HashSet::from([ivec3(10, -20, 30)]));
        assert_eq!(unload, HashSet::from([ivec3(1, 0, 0)]));

        load.clear();
        unload.clear();

        let mut grid: RollingGrid<bool> = RollingGrid::new_default(3, IVec3::ZERO);
        // the grid should only retain at x == y == z == 1
        grid.reposition(
            ivec3(2, 2, 2),
            |vec| {
                load.insert(vec);
                false
            },
            |vec, _| {
                unload.insert(vec);
            },
        );

        assert_eq!(
            load,
            (1..=3)
                .cartesian_product(1..=3)
                .cartesian_product(1..=3)
                .map(|((x, y), z)| ivec3(x, y, z))
                .filter(|&vec| vec != ivec3(1, 1, 1))
                .collect(),
        );

        assert_eq!(
            unload,
            (-1..=1)
                .cartesian_product(-1..=1)
                .cartesian_product(-1..=1)
                .map(|((x, y), z)| ivec3(x, y, z))
                .filter(|&vec| vec != ivec3(1, 1, 1))
                .collect(),
        );

        load.clear();
        unload.clear();

        grid.reposition(
            ivec3(-20, 40, 60),
            |vec| {
                load.insert(vec);
                false
            },
            |vec, _| {
                unload.insert(vec);
            },
        );

        assert_eq!(
            load,
            (-21..=-19)
                .cartesian_product(39..=41)
                .cartesian_product(59..=61)
                .map(|((x, y), z)| ivec3(x, y, z))
                .collect()
        );
        assert_eq!(
            unload,
            (1..=3)
                .cartesian_product(1..=3)
                .cartesian_product(1..=3)
                .map(|((x, y), z)| ivec3(x, y, z))
                .collect()
        );
    }
}
