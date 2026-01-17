use std::{
    array,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

use itertools::Itertools;

#[derive(Clone)]
pub struct NdArray<const N: usize, T> {
    array: Box<[T]>,
    bounds: [usize; N],
}

impl<const N: usize, T> NdArray<N, T> {
    pub fn from_fn(bounds: [usize; N], init: impl Fn([usize; N]) -> T) -> Self {
        let mut slice = Box::new_uninit_slice(bounds.iter().product());

        for (linear_index, index) in bounds
            .map(|i| 0..i)
            .into_iter()
            .multi_cartesian_product()
            .enumerate()
        {
            slice[linear_index] = MaybeUninit::new(init(index.try_into().unwrap()));
        }

        Self {
            array: unsafe { slice.assume_init() },
            bounds,
        }
    }

    fn array_index(&self, index: [usize; N]) -> usize {
        let mut result = 0;
        for (i, bound) in index.iter().copied().zip_eq(self.bounds.iter().copied()) {
            assert!(i < bound);
            result *= bound;
            result += i;
        }

        result
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.array.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = T> {
        self.array.into_iter()
    }

    pub fn iter_enumerated(&self) -> impl Iterator<Item = ([usize; N], &T)> {
        self.iter()
            .enumerate()
            .map(|(index, value)| (Self::reconstruct_array_index(self.bounds, index), value))
    }

    pub fn into_iter_enumerated(self) -> impl Iterator<Item = ([usize; N], T)> {
        let Self { array, bounds } = self;
        array
            .into_iter()
            .enumerate()
            .map(move |(index, value)| (Self::reconstruct_array_index(bounds, index), value))
    }

    fn reconstruct_array_index(bounds: [usize; N], linear_index: usize) -> [usize; N] {
        assert!(linear_index < bounds.iter().product());
        let mut remainder = linear_index;
        let mut index = [0; N];

        for (dimension, bounds) in bounds.iter().copied().enumerate().rev() {
            index[dimension] = remainder % bounds;
            remainder /= bounds;
        }

        index
    }
}

impl<const N: usize, T: Default> NdArray<N, T> {
    pub fn default(bounds: [usize; N]) -> Self {
        let size = bounds.into_iter().product();
        let mut vec = Vec::with_capacity(bounds.into_iter().product());
        for _ in 0..size {
            vec.push(T::default());
        }

        Self {
            array: vec.into_boxed_slice(),
            bounds,
        }
    }
}

impl<const N: usize, T, I> Index<I> for NdArray<N, T>
where
    I: Into<[usize; N]>,
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.array[self.array_index(index.into())]
    }
}

impl<const N: usize, T, I> IndexMut<I> for NdArray<N, T>
where
    I: Into<[usize; N]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.array[self.array_index(index.into())]
    }
}

#[derive(Clone)]
pub struct ShiftedNdArray<const N: usize, T> {
    array: NdArray<N, T>,
    shift: [isize; N],
}

impl<const N: usize, T> ShiftedNdArray<N, T> {
    fn from_fn(bounds: [usize; N], shift: [isize; N], init: impl Fn([isize; N]) -> T) -> Self {
        let array = NdArray::from_fn(bounds, |index| {
            init(array::from_fn(|i| index[i] as isize + shift[i]))
        });

        Self { array, shift }
    }

    fn array_index(&self, index: [isize; N]) -> usize {
        let mut result = 0;
        for (i, (bound, shift)) in index.iter().copied().zip_eq(
            self.array
                .bounds
                .iter()
                .copied()
                .zip_eq(self.shift.iter().copied()),
        ) {
            let unshifted: usize = (i - shift).try_into().unwrap();
            assert!(unshifted < bound);
            result *= bound;
            result += unshifted;
        }

        result
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.array.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = T> {
        self.array.into_iter()
    }

    pub fn iter_enumerated(&self) -> impl Iterator<Item = ([isize; N], &T)> {
        self.array
            .iter_enumerated()
            .map(|(index, val)| (array::from_fn(|i| index[i] as isize + self.shift[i]), val))
    }

    pub fn into_iter_enumerated(self) -> impl Iterator<Item = ([isize; N], T)> {
        let Self { array, shift } = self;
        array
            .into_iter_enumerated()
            .map(move |(index, val)| (array::from_fn(|i| index[i] as isize + shift[i]), val))
    }
}

impl<const N: usize, T> ShiftedNdArray<N, T>
where
    T: Default,
{
    pub fn default(bounds: [usize; N], shift: [isize; N]) -> Self {
        Self {
            shift,
            array: NdArray::default(bounds),
        }
    }
}

impl<const N: usize, T, I> Index<I> for ShiftedNdArray<N, T>
where
    I: Into<[isize; N]>,
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.array.array[self.array_index(index.into())]
    }
}

impl<const N: usize, T, I> IndexMut<I> for ShiftedNdArray<N, T>
where
    I: Into<[isize; N]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.array.array[self.array_index(index.into())]
    }
}

#[derive(Clone)]
pub struct HyperCubeArray<const N: usize, const SIZE: usize, T> {
    array: NdArray<N, T>,
}

impl<const N: usize, const SIZE: usize, T> HyperCubeArray<N, SIZE, T> {
    #[expect(dead_code)]
    pub fn from_fn(init: impl Fn([usize; N]) -> T) -> Self {
        Self {
            array: NdArray::from_fn([SIZE; N], init),
        }
    }

    #[expect(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.array.iter()
    }

    #[expect(dead_code)]
    pub fn into_iter(self) -> impl Iterator<Item = T> {
        self.array.into_iter()
    }

    #[expect(dead_code)]
    pub fn iter_enumerated(&self) -> impl Iterator<Item = ([usize; N], &T)> {
        self.array.iter_enumerated()
    }

    #[expect(dead_code)]
    pub fn into_iter_enumerated(self) -> impl Iterator<Item = ([usize; N], T)> {
        self.array.into_iter_enumerated()
    }
}

impl<const N: usize, const SIZE: usize, T> HyperCubeArray<N, SIZE, T>
where
    T: Default,
{
    pub fn default() -> Self {
        Self {
            array: NdArray::default([SIZE; N]),
        }
    }
}

impl<const N: usize, const SIZE: usize, T, I> Index<I> for HyperCubeArray<N, SIZE, T>
where
    I: Into<[usize; N]>,
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        self.array.index(index.into())
    }
}

impl<const N: usize, const SIZE: usize, T, I> IndexMut<I> for HyperCubeArray<N, SIZE, T>
where
    I: Into<[usize; N]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.array.index_mut(index.into())
    }
}

#[derive(Clone)]
pub struct ShiftedHyperCubeArray<const N: usize, const SIZE: usize, T> {
    array: ShiftedNdArray<N, T>,
}

impl<const N: usize, const SIZE: usize, T> ShiftedHyperCubeArray<N, SIZE, T> {
    pub fn from_fn(shift: [isize; N], init: impl Fn([isize; N]) -> T) -> Self {
        Self {
            array: ShiftedNdArray::from_fn([SIZE; N], shift, init),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.array.iter()
    }

    #[expect(dead_code)]
    pub fn into_iter(self) -> impl Iterator<Item = T> {
        self.array.into_iter()
    }

    #[expect(dead_code)]
    pub fn iter_enumerated(&self) -> impl Iterator<Item = ([isize; N], &T)> {
        self.array.iter_enumerated()
    }

    pub fn into_iter_enumerated(self) -> impl Iterator<Item = ([isize; N], T)> {
        self.array.into_iter_enumerated()
    }
}

impl<const N: usize, const SIZE: usize, T> ShiftedHyperCubeArray<N, SIZE, T>
where
    T: Default,
{
    pub fn default(shift: [isize; N]) -> Self {
        Self {
            array: ShiftedNdArray::default([SIZE; N], shift),
        }
    }
}

impl<const N: usize, const SIZE: usize, T, I> Index<I> for ShiftedHyperCubeArray<N, SIZE, T>
where
    I: Into<[isize; N]>,
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        self.array.index(index.into())
    }
}

impl<const N: usize, const SIZE: usize, T, I> IndexMut<I> for ShiftedHyperCubeArray<N, SIZE, T>
where
    I: Into<[isize; N]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.array.index_mut(index.into())
    }
}

#[cfg(test)]
mod tests {
    use crate::math::nd_array::{NdArray, ShiftedNdArray};

    #[test]
    fn test_from_fn() {
        let array = NdArray::from_fn([2, 3, 4], |pos| pos);

        for x in 0..2 {
            for y in 0..3 {
                for z in 0..4 {
                    assert_eq!(array[[x, y, z]], [x, y, z]);
                }
            }
        }
    }

    #[test]
    fn test_shifted() {
        let array = ShiftedNdArray::from_fn([2, 3, 4], [-1, -1, -1], |pos| pos);

        for x in -1..1 {
            for y in -1..2 {
                for z in -1..3 {
                    assert_eq!(array[[x, y, z]], [x, y, z]);
                }
            }
        }
    }

    #[test]
    fn test_iter_enumerated() {
        let array = NdArray::from_fn([2, 3, 4], |pos| pos);

        for (index, value) in array.iter_enumerated() {
            assert_eq!(index, *value);
        }
    }

    #[test]
    fn test_iter_enumerated_shifted() {
        let array = ShiftedNdArray::from_fn([2, 3, 4], [-1, -1, -1], |pos| pos);

        for (index, value) in array.iter_enumerated() {
            assert_eq!(index, *value);
        }
    }
}
