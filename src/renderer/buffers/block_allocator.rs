use std::{
    marker::PhantomData,
    rc::{Rc, Weak},
};

use crate::renderer::buffers::{AsBytes, WriteBuffer};

pub type RcBlockHandle = Rc<u64>;

/// Allocator for fixed-size blocks of the generic type.
pub struct BlockAllocator<T: AsBytes> {
    blocks_allocated: Box<[bool]>,
    phantom: PhantomData<T>,
}

impl<T: AsBytes> BlockAllocator<T> {
    pub fn new(block_count: u64) -> Self {
        Self {
            blocks_allocated: vec![false; block_count as usize].into_boxed_slice(),
            phantom: PhantomData,
        }
    }

    pub fn allocate_block(&mut self, target: &mut impl WriteBuffer, data: &T, block: u64) {
        target.write(block * std::mem::size_of::<T>() as u64, data.get_bytes());

        self.blocks_allocated[block as usize] = true;
    }

    #[allow(dead_code)]
    pub fn deallocate_block(&mut self, block: u64) {
        self.blocks_allocated[block as usize] = false;
    }

    #[allow(dead_code)]
    pub fn first_free_block(&self, offset: u64) -> Option<u64> {
        self.blocks_allocated
            .iter()
            .skip(offset as usize)
            .position(|allocated| !allocated)
            .map(|index| index as u64)
    }
}

pub struct RcBlockAllocator<T: AsBytes> {
    block_allocator: BlockAllocator<T>,
    blocks: Box<[Weak<u64>]>,
}

impl<T: AsBytes> RcBlockAllocator<T> {
    pub fn new(block_count: u64) -> Self {
        Self {
            block_allocator: BlockAllocator::new(block_count),
            blocks: vec![Weak::new(); block_count as usize].into_boxed_slice(),
        }
    }

    fn first_free_block(&self) -> Option<u64> {
        self.blocks
            .iter()
            .position(|block| block.upgrade().is_none())
            .map(|index| index as u64)
    }

    #[must_use]
    pub fn allocate_first_free_block(
        &mut self,
        target: &mut impl WriteBuffer,
        data: &T,
    ) -> RcBlockHandle {
        let index = self.first_free_block().expect("No free block available");
        let handle = Rc::new(index);
        self.blocks[index as usize] = Rc::downgrade(&handle);

        self.block_allocator.allocate_block(target, data, index);

        handle
    }
}

#[cfg(test)]
mod tests {
    use crate::renderer::buffers::tests::TestMemoryTarget;

    use super::*;

    impl AsBytes for u8 {
        fn get_bytes(&self) -> &[u8] {
            std::slice::from_ref(self)
        }
    }

    #[test]
    fn test_rc() {
        let mut alloc = RcBlockAllocator::new(1);
        let mut mem = TestMemoryTarget { memory: [0; 1] };

        let handle = alloc.allocate_first_free_block(&mut mem, &0);
        let handle_clone = Rc::clone(&handle);

        assert!(alloc.blocks[0].upgrade().is_some_and(|value| *value == 0));
        drop(handle);
        assert!(alloc.blocks[0].upgrade().is_some_and(|value| *value == 0));
        drop(handle_clone);
        assert!(alloc.blocks[0].upgrade().is_none());
    }
}
