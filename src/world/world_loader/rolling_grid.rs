use std::{collections::HashSet, marker::PhantomData, mem::MaybeUninit};

use glam::{IVec2, IVec3, USizeVec3, Vec3Swizzles, ivec2, ivec3};
use itertools::Itertools;

pub struct RollingGrid<T, I>
where
    I: Copy + Into<IVec3> + From<IVec3>,
{
    /// Invariant: width is always an uneven number
    width: usize,
    array: Box<[T]>,
    center: IVec3,
    phantom: PhantomData<I>,
}

impl<T, I> RollingGrid<T, I>
where
    I: Copy + Into<IVec3> + From<IVec3>,
{
    /// Create new instance with the given width and center point.
    /// For every cell in the grid, the load function is called.
    pub fn new<Ctx>(
        width: usize,
        center: I,
        ctx: &mut Ctx,
        load: impl Fn(&mut Ctx, I) -> T,
    ) -> Self {
        assert!(width & 1 == 1, "N must be an uneven number");
        let mut array: Box<[MaybeUninit<T>]> = Box::new_uninit_slice(width.pow(3));

        for position in Self::iter_3d(width, center.into()) {
            // SAFETY: Position is contained in the grid.
            array[Self::position_to_index_unchecked(width, position)] =
                MaybeUninit::new(load(ctx, position.into()));
        }

        Self {
            width,
            // SAFETY: All items were initialized in the loop.
            array: unsafe { array.assume_init() },
            center: center.into(),
            phantom: PhantomData,
        }
    }

    fn iter_1d(width: usize, center: i32) -> impl Iterator<Item = i32> + Clone {
        let n_half = (width / 2) as i32;
        center - n_half..=center + n_half
    }

    fn iter_2d(width: usize, center: IVec2) -> impl Iterator<Item = IVec2> + Clone {
        Self::iter_1d(width, center.x)
            .cartesian_product(Self::iter_1d(width, center.y))
            .map(|(x, y)| ivec2(x, y))
    }

    fn iter_3d(width: usize, center: IVec3) -> impl Iterator<Item = IVec3> + Clone {
        Self::iter_1d(width, center.x)
            .cartesian_product(Self::iter_1d(width, center.y))
            .cartesian_product(Self::iter_1d(width, center.z))
            .map(|((x, y), z)| ivec3(x, y, z))
    }

    /// Returns the lower and upper bound coordinates of the grid.
    pub fn bounds(&self) -> (I, I) {
        let min = self.center - IVec3::splat(self.width as i32 / 2);
        let max = self.center + IVec3::splat(self.width as i32 / 2);

        (min.into(), max.into())
    }

    /// Returns true if the position is within `self.width / 2` from `self.center` on every axis.
    pub fn contains(&self, position: I) -> bool {
        self._contains(position.into())
    }

    fn _contains(&self, position: IVec3) -> bool {
        usize::try_from((self.center - position).abs().max_element()).unwrap() <= self.width / 2
    }

    fn position_to_index_unchecked(width: usize, position: IVec3) -> usize {
        let USizeVec3 { x, y, z } = position
            .rem_euclid(IVec3::splat(width.try_into().unwrap()))
            .as_usizevec3();

        (x * width + y) * width + z
    }

    /// Compute the `self.grid` index from the given vector.
    /// This function panicks if the position is more than `self.width / 2` away from `self.center`.
    fn position_to_index(&self, position: IVec3) -> usize {
        if !self._contains(position) {
            panic!(
                "position vector {} not within grid around {} and width {}",
                position, self.center, self.width
            );
        }

        Self::position_to_index_unchecked(self.width, position)
    }

    pub fn center(&self) -> IVec3 {
        self.center
    }

    /// Get an immutable reference to the grid contents of the given position.
    /// Returns None if the position is not within the grid around the current center.
    #[cfg_attr(not(test), expect(dead_code))]
    pub fn at(&self, position: I) -> Option<&T> {
        if !self.contains(position) {
            return None;
        }
        Some(&self.array[self.position_to_index(position.into())])
    }

    /// Get a mutable reference to the grid contents of the given position.
    /// Returns None if the position is not within the grid around the current center.
    pub fn at_mut(&mut self, position: I) -> Option<&mut T> {
        self._at_mut(position.into())
    }

    fn _at_mut(&mut self, position: IVec3) -> Option<&mut T> {
        if !self._contains(position) {
            return None;
        }
        Some(&mut self.array[self.position_to_index(position)])
    }

    /// Insert the given value into the grid cell at the given position, returning the old value.
    /// Returns None and does not store the new value if the position is not within the grid around the current center.
    pub fn replace(&mut self, position: I, new_value: T) -> Option<T> {
        if !self.contains(position) {
            return None;
        }
        Some(std::mem::replace(self.at_mut(position).unwrap(), new_value))
    }

    /// Reposition the grid so the given vector becomes the new center.
    /// This necessitates unloading cells around the old, and loading cells around the new center.
    /// For every unloaded and loaded cell, the respective function is called exactly once.
    pub fn reposition<Ctx>(
        &mut self,
        new_center: IVec3,
        ctx: &mut Ctx,
        mut load: impl FnMut(&mut Ctx, I) -> T,
        mut unload: impl FnMut(&mut Ctx, I, T),
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
                // unvectorized version:
                // let mut new_cell = old_cell;
                // while (new_cell.x - new_center.x).abs() > half_width {
                //     new_cell.x += dx.signum() * width_i32;
                // }
                // ...

                // absolute shifts by width for each dimension to move from the old to the new cell
                let shifts = ((new_center - old_cell).abs() + IVec3::splat(half_width)) / width_i32;
                let new_cell = old_cell + diff.signum() * shifts * width_i32;

                let old_value =
                    std::mem::replace(self._at_mut(old_cell).unwrap(), load(ctx, new_cell.into()));
                unload(ctx, old_cell.into(), old_value);
            }
        };

        for x in x_range {
            for IVec2 { x: y, y: z } in Self::iter_2d(width, old_position.yz()) {
                update_once(ivec3(x, y, z));
            }
        }

        for y in y_range {
            for IVec2 { x, y: z } in Self::iter_2d(width, old_position.xz()) {
                update_once(ivec3(x, y, z));
            }
        }

        for z in z_range {
            for IVec2 { x, y } in Self::iter_2d(width, old_position.xy()) {
                update_once(ivec3(x, y, z));
            }
        }

        self.center = new_center;
    }

    pub fn reset<Ctx>(&mut self, ctx: &mut Ctx, load: impl Fn(&mut Ctx, I) -> T) {
        for position in Self::iter_3d(self.width, self.center) {
            self.array[self.position_to_index(position)] = load(ctx, position.into());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::{IVec3, ivec3};
    use itertools::Itertools;

    use crate::world::world_loader::rolling_grid::RollingGrid;

    impl<T: Default> RollingGrid<T, IVec3> {
        pub fn new_default(width: usize, position: IVec3) -> Self {
            Self::new(width, position, &mut (), |_, _| T::default())
        }

        pub fn reposition_default(&mut self, new_position: IVec3) {
            self.reposition(new_position, &mut (), |_, _| T::default(), |_, _, _| ());
        }
    }

    #[test]
    fn test() {
        let mut grid: RollingGrid<bool, IVec3> = RollingGrid::new_default(3, IVec3::ZERO);
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
        let mut grid: RollingGrid<bool, IVec3> = RollingGrid::new_default(1, IVec3::ZERO);
        grid.reposition(
            ivec3(1, 0, 0),
            &mut (&mut load, &mut unload),
            |(load, _unload), vec| {
                load.insert(vec);
                false
            },
            |(_load, unload), vec, _| {
                unload.insert(vec);
            },
        );
        assert_eq!(load, HashSet::from([ivec3(1, 0, 0)]));
        assert_eq!(unload, HashSet::from([ivec3(0, 0, 0)]));

        load.clear();
        unload.clear();

        grid.reposition(
            ivec3(10, -20, 30),
            &mut (&mut load, &mut unload),
            |(load, _unload), vec| load.insert(vec),
            |(_load, unload), vec, _| {
                unload.insert(vec);
            },
        );
        assert_eq!(load, HashSet::from([ivec3(10, -20, 30)]));
        assert_eq!(unload, HashSet::from([ivec3(1, 0, 0)]));

        load.clear();
        unload.clear();

        let mut grid: RollingGrid<bool, IVec3> = RollingGrid::new_default(3, IVec3::ZERO);
        // the grid should only retain at x == y == z == 1
        grid.reposition(
            ivec3(2, 2, 2),
            &mut (&mut load, &mut unload),
            |(load, _unload), vec| {
                load.insert(vec);
                false
            },
            |(_load, unload), vec, _| {
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
            &mut (&mut load, &mut unload),
            |(load, _unload), vec| {
                load.insert(vec);
                false
            },
            |(_load, unload), vec, _| {
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
