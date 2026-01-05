use std::ops::{Index, IndexMut};

use itertools::Itertools;

pub struct NdArray<const N: usize, T> {
    bounds: [usize; N],
    array: Box<[T]>,
}

impl<const N: usize, T> NdArray<N, T> {
    fn array_index(&self, index: [usize; N]) -> usize {
        let mut result = 0;
        for (i, bounds) in index.iter().copied().zip_eq(self.bounds.iter().copied()) {
            result *= bounds;
            result += i;
        }

        result
    }
}

impl<const N: usize, T: Default> NdArray<N, T> {
    pub fn default(bounds: [usize; N]) -> Self {
        let size = bounds.iter().product();
        let mut vec = Vec::with_capacity(size);

        for _ in 0..size {
            vec.push(Default::default());
        }

        Self {
            bounds,
            array: vec.into_boxed_slice(),
        }
    }
}

impl<const N: usize, T, I: Into<[usize; N]>> Index<I> for NdArray<N, T> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.array[self.array_index(index.into())]
    }
}

impl<const N: usize, T, I: Into<[usize; N]>> IndexMut<I> for NdArray<N, T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.array[self.array_index(index.into())]
    }
}

pub struct HyperCubeArray<const N: usize, const SIZE: usize, T> {
    array: NdArray<N, T>,
}

impl<const N: usize, const SIZE: usize, T: Default> HyperCubeArray<N, SIZE, T> {
    pub fn default() -> Self {
        Self {
            array: NdArray::default([SIZE; N]),
        }
    }
}

impl<const N: usize, const SIZE: usize, T, I: Into<[usize; N]>> Index<I>
    for HyperCubeArray<N, SIZE, T>
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        self.array.index(index.into())
    }
}

impl<const N: usize, const SIZE: usize, T, I: Into<[usize; N]>> IndexMut<I>
    for HyperCubeArray<N, SIZE, T>
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.array.index_mut(index.into())
    }
}
