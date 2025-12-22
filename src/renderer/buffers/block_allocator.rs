use std::marker::PhantomData;

use crate::renderer::buffers::{AllocationError, AsBytes, WriteBuffer};

#[derive(Clone, Copy, Debug)]
pub struct BlockHandle<T>(pub u64, pub PhantomData<T>);

#[derive(Clone, Copy, Debug)]
pub struct CountedBlockHandle<T>(pub u64, PhantomData<T>);

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

    /// Write to block, returns error if the block is already allocated.
    pub fn allocate_block(
        &mut self,
        target: &mut impl WriteBuffer,
        block: BlockHandle<T>,
        data: &T,
    ) -> Result<(), AllocationError> {
        let marker = self
            .blocks_allocated
            .get_mut(block.0 as usize)
            .ok_or(AllocationError::InvalidHandle)
            .and_then(|value| {
                if !*value {
                    Ok(value)
                } else {
                    Err(AllocationError::MemoryNotFree)
                }
            })?;

        *marker = true;

        self.overwrite_block(target, block, data)
    }

    /// Write to block, does not check whether block is allocated.
    pub fn overwrite_block(
        &mut self,
        target: &mut impl WriteBuffer,
        block: BlockHandle<T>,
        data: &T,
    ) -> Result<(), AllocationError> {
        if self.blocks_allocated.len() <= block.0 as usize {
            return Err(AllocationError::InvalidHandle);
        }
        target.write(block.0 * std::mem::size_of::<T>() as u64, data.get_bytes());
        Ok(())
    }

    /// Deallocate block. Returns error if block is not currently allocated.
    pub fn deallocate_block(&mut self, block: BlockHandle<T>) -> Result<(), AllocationError> {
        let previously_allocated = std::mem::replace(
            self.blocks_allocated
                .get_mut(block.0 as usize)
                .ok_or(AllocationError::InvalidHandle)?,
            false,
        );

        if !previously_allocated {
            Err(AllocationError::IllegalFree)
        } else {
            Ok(())
        }
    }

    pub fn first_free_block(&self) -> Result<BlockHandle<T>, AllocationError> {
        let index = self
            .blocks_allocated
            .iter()
            .position(|allocated| !allocated)
            .ok_or(AllocationError::NoFreeSegmentAvailable)?;
        Ok(BlockHandle(index as u64, PhantomData))
    }
}

pub struct CountedBlockAllocator<T: AsBytes> {
    // Each integer counts the current usage of each block. A zero marks a free block.
    blocks: Box<[u32]>,
    phantom: PhantomData<T>,
}

impl<T: AsBytes> CountedBlockAllocator<T> {
    pub fn new(block_count: u64) -> Self {
        Self {
            blocks: vec![0; block_count as usize].into_boxed_slice(),
            phantom: PhantomData,
        }
    }

    fn first_free_block(&self) -> Option<usize> {
        self.blocks.iter().position(|block| *block == 0)
    }

    pub fn allocate_first_free_block(
        &mut self,
        target: &mut impl WriteBuffer,
        data: &T,
    ) -> Result<CountedBlockHandle<T>, AllocationError> {
        let index = self
            .first_free_block()
            .ok_or(AllocationError::NoFreeSegmentAvailable)?;

        self.blocks[index] = 1;
        target.write((index * std::mem::size_of::<T>()) as u64, data.get_bytes());

        Ok(CountedBlockHandle(index as u64, PhantomData))
    }

    /// Increment the counter, returning the new counter value or Err, if the handle is invalid.
    pub fn increment_counter(
        &mut self,
        handle: CountedBlockHandle<T>,
    ) -> Result<u32, AllocationError> {
        if self.blocks[handle.0 as usize] == 0 {
            return Err(AllocationError::InvalidHandle);
        }

        let count = &mut self.blocks[handle.0 as usize];
        *count += 1;
        Ok(*count)
    }

    /// Decrement the counter, returing the new counter value or None if the block is freed due to the counter reaching 0. Returns Err if the handle is invalid.
    pub fn decrement_counter(
        &mut self,
        handle: CountedBlockHandle<T>,
    ) -> Result<Option<u32>, AllocationError> {
        if self.blocks[handle.0 as usize] == 0 {
            return Err(AllocationError::InvalidHandle);
        }

        let count = &mut self.blocks[handle.0 as usize];
        *count -= 1;

        Ok(if *count > 0 { Some(*count) } else { None })
    }
}

#[cfg(test)]
mod tests {
    use crate::renderer::buffers::tests::TestMemoryTarget;

    use super::*;

    #[test]
    fn test_rc() -> Result<(), anyhow::Error> {
        let mut alloc = CountedBlockAllocator::new(1);
        let mut mem = TestMemoryTarget { memory: [0; 4] };

        let handle = alloc.allocate_first_free_block(&mut mem, &0)?;

        assert_eq!(alloc.blocks[0], 1);
        alloc.increment_counter(handle)?;
        assert_eq!(alloc.blocks[0], 2);
        alloc.decrement_counter(handle)?;
        assert_eq!(alloc.blocks[0], 1);
        alloc.decrement_counter(handle)?;
        assert_eq!(alloc.blocks[0], 0);
        alloc
            .decrement_counter(handle)
            .expect_err("Should fail because handle is no longer pointing to valid allocation");

        Ok(())
    }
}
