use crate::renderer::buffers::{self, AllocationError, CopyFromBuffer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentHandle {
    pub offset: u64,
    pub size: u64,
}

impl SegmentHandle {
    fn end(&self) -> u64 {
        self.offset + self.size
    }
}

/// Pool/arena allocator that allocates variable-sized objects and manages free segments.
pub struct PoolAllocator {
    size: u64,
    occupied_segments: Vec<SegmentHandle>,
    free_segments: Vec<SegmentHandle>,
}

impl PoolAllocator {
    pub fn new(size: u64) -> Self {
        Self {
            size,
            occupied_segments: Vec::new(),
            free_segments: vec![SegmentHandle { offset: 0, size }],
        }
    }

    pub fn reserve_segment(
        &mut self,
        size: u64,
        alignment: u64,
    ) -> Result<SegmentHandle, AllocationError> {
        // Find the smallest segment that is sufficiently large to store the contents of the `source` buffer plus optional alignment bytes.
        // `alignment_bytes` is the number of bytes added to the segment offset for correct data alignment
        let Some((index, free_segment, alignment_bytes)) = self
            .free_segments
            .iter()
            .enumerate()
            .filter_map(|(index, segment)| {
                let aligned_offset = buffers::align_up(segment.offset, alignment);
                // number of bytes added to the segment offset for correct data alignment
                let alignment_bytes = aligned_offset - segment.offset;

                if segment.size - alignment_bytes >= size {
                    Some((index, segment, alignment_bytes))
                } else {
                    None
                }
            })
            .min_by_key(|(_, segment, _)| segment.size)
        else {
            return Err(AllocationError::NoFreeSegmentAvailable);
        };

        let new_occupied_segment = SegmentHandle {
            offset: free_segment.offset + alignment_bytes,
            size,
        };
        self.occupied_segments.push(new_occupied_segment);

        // Create free segment after the new occupied segment, if the original free segment was not fully used
        if alignment_bytes + size < free_segment.size {
            self.free_segments.push(SegmentHandle {
                offset: free_segment.offset + alignment_bytes + size,
                size: free_segment.size - alignment_bytes - size,
            });
        }

        // If alignment was neccessary, store alignment bytes as new segment.
        // Otherwise remove the segment.
        if alignment_bytes == 0 {
            self.free_segments.swap_remove(index);
        } else {
            self.free_segments[index].size = alignment_bytes;
        }

        Ok(new_occupied_segment)
    }

    /// Copy from source to target. `segment.size` bytes are copied, beginning from 0 at the source, and `segment.offset` for the target.
    pub fn insert_into_segment<T>(
        &mut self,
        source: &T,
        target: &mut impl CopyFromBuffer<T>,
        segment: SegmentHandle,
    ) {
        target.copy_from_buffer(source, 0, segment.offset, segment.size);
    }

    pub fn deallocate(&mut self, handle: SegmentHandle) -> Result<(), AllocationError> {
        let index = self
            .occupied_segments
            .iter()
            .position(|segment| *segment == handle)
            .ok_or(AllocationError::InvalidHandle)?;

        self.occupied_segments.swap_remove(index);

        let mut new_free_segment = handle;

        // Find segment immediately before the deallocated segment, merge if present
        if let Some((index_before, segment_before)) = self
            .free_segments
            .iter()
            .enumerate()
            .find(|&(_, segment)| segment.end() == new_free_segment.offset)
        {
            new_free_segment.offset -= segment_before.size;
            new_free_segment.size += segment_before.size;
            self.free_segments.swap_remove(index_before);
        }

        // Find segment immediately after the deallocated segment, merge if present
        if let Some((index_after, segment_after)) = self
            .free_segments
            .iter()
            .enumerate()
            .find(|&(_, segment)| segment.offset == new_free_segment.end())
        {
            new_free_segment.size += segment_after.size;
            self.free_segments.swap_remove(index_after);
        }

        self.free_segments.push(new_free_segment);

        Ok(())
    }

    pub fn grow(&mut self, new_size: u64) {
        if let Some(free_segment) = self
            .free_segments
            .iter_mut()
            .find(|segment| segment.end() == self.size)
        {
            free_segment.size += new_size - self.size;
        } else {
            self.free_segments.push(SegmentHandle {
                offset: self.size,
                size: new_size - self.size,
            });
        }

        self.size = new_size;
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

#[cfg(test)]
mod tests {

    use crate::{renderer::buffers::tests::TestMemoryTarget, tests::cmp_vec_unordered};

    use super::*;

    fn init() -> (TestMemoryTarget<16>, PoolAllocator) {
        (TestMemoryTarget { memory: [0; 16] }, PoolAllocator::new(16))
    }

    #[test]
    fn test_allocate_from_buffer() -> Result<(), anyhow::Error> {
        let (mut mem, mut pool) = init();

        let handle = pool.reserve_segment(16, 1)?;
        pool.insert_into_segment(&[0xFF; 16], &mut mem, handle);
        assert_eq!(mem.memory, [0xFF; 16]);
        pool.deallocate(handle)?;

        let handle = pool.reserve_segment(1, 1)?;
        pool.insert_into_segment(&[0x00; 16], &mut mem, handle);
        assert_eq!(mem.memory[0], 0x00);
        // Data previously deallocated isn't cleared, just marked as empty
        assert_eq!(mem.memory[1..16], [0xFF; 15]);

        Ok(())
    }

    #[test]
    fn test_reallocation() -> Result<(), anyhow::Error> {
        let (mut mem, mut pool) = init();

        let (h1, h2, h3, h4) = (
            pool.reserve_segment(4, 1)?,
            pool.reserve_segment(4, 1)?,
            pool.reserve_segment(4, 1)?,
            pool.reserve_segment(4, 1)?,
        );

        pool.insert_into_segment(&[0x01; 16], &mut mem, h1);
        pool.insert_into_segment(&[0x02; 16], &mut mem, h2);
        pool.insert_into_segment(&[0x03; 16], &mut mem, h3);
        pool.insert_into_segment(&[0x04; 16], &mut mem, h4);

        pool.deallocate(h2)?;
        pool.deallocate(h3)?;

        let handle = pool.reserve_segment(8, 1)?;
        pool.insert_into_segment(&[0xFF; 16], &mut mem, handle);
        assert_eq!(
            mem.memory,
            [
                [0x01; 4].as_slice(),
                [0xFF; 8].as_slice(),
                [0x04; 4].as_slice()
            ]
            .concat()
            .as_slice()
        );

        Ok(())
    }

    #[test]
    fn test_alignment() -> Result<(), anyhow::Error> {
        let (mut mem, mut pool) = init();

        let (h1, h2) = (pool.reserve_segment(1, 1)?, pool.reserve_segment(8, 4)?);
        pool.insert_into_segment(&[0xEE; 16], &mut mem, h1);
        pool.insert_into_segment(&[0xFF; 16], &mut mem, h2);

        assert_eq!(
            mem.memory,
            [
                [0xEE; 1].as_slice(),
                [0x00; 3].as_slice(),
                [0xFF; 8].as_slice(),
                [0x00; 4].as_slice()
            ]
            .concat()
            .as_slice()
        );

        Ok(())
    }

    #[test]
    fn test_state() -> Result<(), anyhow::Error> {
        let (mut mem, mut pool) = init();

        let (h1, h2) = (pool.reserve_segment(1, 1)?, pool.reserve_segment(8, 4)?);
        pool.insert_into_segment(&[0xEE; 16], &mut mem, h1);
        pool.insert_into_segment(&[0xFF; 16], &mut mem, h2);

        cmp_vec_unordered(
            &pool.occupied_segments,
            &vec![
                SegmentHandle { offset: 0, size: 1 },
                SegmentHandle { offset: 4, size: 8 },
            ],
        )
        .expect("Occupied segments incorrect");

        cmp_vec_unordered(
            &pool.free_segments,
            &vec![
                SegmentHandle { offset: 1, size: 3 },
                SegmentHandle {
                    offset: 12,
                    size: 4,
                },
            ],
        )
        .expect("Free segments incorrect");

        Ok(())
    }

    #[test]
    fn test_deallocate() -> Result<(), anyhow::Error> {
        let (mut mem, mut pool) = init();

        let (h1, h2) = (pool.reserve_segment(1, 1)?, pool.reserve_segment(8, 4)?);
        pool.insert_into_segment(&[0xEE; 16], &mut mem, h1);
        pool.insert_into_segment(&[0xFF; 16], &mut mem, h2);

        pool.deallocate(h1)?;

        cmp_vec_unordered(
            &pool.occupied_segments,
            &vec![SegmentHandle { offset: 4, size: 8 }],
        )
        .expect("Occupied segments incorrect");

        cmp_vec_unordered(
            &pool.free_segments,
            &vec![
                SegmentHandle { offset: 0, size: 4 },
                SegmentHandle {
                    offset: 12,
                    size: 4,
                },
            ],
        )
        .expect("Free segments incorrect");

        pool.deallocate(h2)?;

        assert_eq!(pool.occupied_segments.len(), 0);

        cmp_vec_unordered(
            &pool.free_segments,
            &vec![SegmentHandle {
                offset: 0,
                size: 16,
            }],
        )
        .expect("Free segments incorrect");

        Ok(())
    }

    #[test]
    fn test_reserve_segment_complete() -> Result<(), anyhow::Error> {
        let (_, mut pool) = init();
        pool.reserve_segment(17, 1)
            .expect_err("Should not be able to create segment larger than the buffer");

        pool.reserve_segment(1, 1)?;
        pool.reserve_segment(16, 1).expect_err(
            "Should not be able to create segment larger than the remaining free buffer space",
        );

        Ok(())
    }
}
