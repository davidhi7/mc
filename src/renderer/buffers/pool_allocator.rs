use crate::renderer::buffers::{self, MemoryTarget};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentHandle {
    pub offset: u64,
    pub size: u64,
}

pub struct PoolAllocator {
    occupied_segments: Vec<SegmentHandle>,
    free_segments: Vec<SegmentHandle>,
}

impl PoolAllocator {
    pub fn new(size: u64) -> Self {
        Self {
            occupied_segments: Vec::new(),
            free_segments: vec![SegmentHandle { offset: 0, size }],
        }
    }

    pub fn allocate_from_buffer<T>(
        &mut self,
        target: &mut impl MemoryTarget<T>,
        source: &T,
        copy_size: u64,
        alignment: u64,
    ) -> SegmentHandle {
        // Find the smallest segment that is sufficiently large to store the contents of the `source` buffer
        // `alignment_offset` is the number of bytes added to the segment offset for correct data alignment
        let (index, free_segment, alignment_offset) = self
            .free_segments
            .iter()
            .enumerate()
            .filter_map(|(index, segment)| {
                let aligned_offset = buffers::align_up(segment.offset, alignment as u64);
                let alignment_bytes = aligned_offset - segment.offset;

                if segment.size - alignment_bytes >= copy_size {
                    Some((index, segment, alignment_bytes))
                } else {
                    None
                }
            })
            .min_by_key(|(_index, segment, _alignment)| segment.size)
            .expect("No empty segment of sufficient size found");

        target.copy_from_buffer(source, 0, free_segment.offset + alignment_offset, copy_size);

        let new_occupied_segment = SegmentHandle {
            offset: free_segment.offset + alignment_offset,
            size: copy_size,
        };
        self.occupied_segments.push(new_occupied_segment);

        // Create free segment following the new occupied segment, if the original free segment was not fully used
        if alignment_offset + copy_size < free_segment.size {
            self.free_segments.push(SegmentHandle {
                offset: free_segment.offset + alignment_offset + copy_size,
                size: free_segment.size - alignment_offset - copy_size,
            });
        }

        // Keep free segment before the occupied segment that covers the alignment bytes
        if alignment_offset == 0 {
            self.free_segments.swap_remove(index);
        } else {
            self.free_segments[index].size = alignment_offset;
        }

        new_occupied_segment
    }

    pub fn deallocate(&mut self, handle: &SegmentHandle) {
        let index = self
            .occupied_segments
            .iter()
            .position(|segment| segment == handle)
            .expect("Invalid handle provided");

        self.occupied_segments.swap_remove(index);

        let mut new_free_segment = *handle;

        if let Some((index_before, segment_before)) = self
            .free_segments
            .iter()
            .enumerate()
            .filter(|&(_index, segment)| segment.size + segment.offset == new_free_segment.offset)
            .next()
        {
            new_free_segment.offset -= segment_before.size;
            new_free_segment.size += segment_before.size;
            self.free_segments.swap_remove(index_before);
        }

        if let Some((index_after, segment_after)) = self
            .free_segments
            .iter()
            .enumerate()
            .filter(|&(_index, segment)| {
                segment.offset == new_free_segment.size + new_free_segment.offset
            })
            .next()
        {
            new_free_segment.size += segment_after.size;
            self.free_segments.swap_remove(index_after);
        }

        self.free_segments.push(new_free_segment);
    }
}

#[cfg(test)]
mod tests {

    use crate::{renderer::buffers::tests::TestMemoryTarget, tests::bad_cmp_vec_unordered};

    use super::*;

    fn init() -> (TestMemoryTarget<16>, PoolAllocator) {
        (TestMemoryTarget { memory: [0; 16] }, PoolAllocator::new(16))
    }

    #[test]
    fn test_allocate_from_buffer() {
        let (mut mem, mut pool) = init();

        let handle = pool.allocate_from_buffer(&mut mem, &&[0xFF; 16], 16, 1);

        assert_eq!(mem.memory, [0xFF; 16]);

        pool.deallocate(&handle);
        pool.allocate_from_buffer(&mut mem, &[0x00; 16], 1, 1);
        assert_eq!(mem.memory[0], 0x00);
        // Data previously deallocated isn't cleared, just marked as empty
        assert_eq!(mem.memory[1..16], [0xFF; 15]);
    }

    #[test]
    fn test_reallocation() {
        let (mut mem, mut pool) = init();

        pool.allocate_from_buffer(&mut mem, &&[0x01; 16], 4, 1);
        let handle_2 = pool.allocate_from_buffer(&mut mem, &&[0x02; 16], 4, 1);
        let handle_3 = pool.allocate_from_buffer(&mut mem, &&[0x03; 16], 4, 1);
        pool.allocate_from_buffer(&mut mem, &&[0x04; 16], 4, 1);

        pool.deallocate(&handle_2);
        pool.deallocate(&handle_3);

        pool.allocate_from_buffer(&mut mem, &[0xFF; 16], 8, 1);
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
    }

    #[test]
    fn test_alignment() {
        let (mut mem, mut pool) = init();

        pool.allocate_from_buffer(&mut mem, &[0xEE; 16], 1, 1);
        pool.allocate_from_buffer(&mut mem, &[0xFF; 16], 8, 4);

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
    }

    #[test]
    fn test_state() -> Result<(), String> {
        let (mut mem, mut pool) = init();

        pool.allocate_from_buffer(&mut mem, &[0xEE; 16], 1, 1);
        pool.allocate_from_buffer(&mut mem, &[0xFF; 16], 8, 4);

        bad_cmp_vec_unordered(
            &pool.occupied_segments,
            &vec![
                SegmentHandle { offset: 0, size: 1 },
                SegmentHandle { offset: 4, size: 8 },
            ],
        )?;

        bad_cmp_vec_unordered(
            &pool.free_segments,
            &vec![
                SegmentHandle { offset: 1, size: 3 },
                SegmentHandle {
                    offset: 12,
                    size: 4,
                },
            ],
        )?;

        Ok(())
    }

    #[test]
    fn test_deallocate() -> Result<(), String> {
        let (mut mem, mut pool) = init();

        let handle_1 = pool.allocate_from_buffer(&mut mem, &[0xEE; 16], 1, 1);
        let handle_2 = pool.allocate_from_buffer(&mut mem, &[0xFF; 16], 8, 4);

        pool.deallocate(&handle_1);

        bad_cmp_vec_unordered(
            &pool.occupied_segments,
            &vec![SegmentHandle { offset: 4, size: 8 }],
        )?;

        bad_cmp_vec_unordered(
            &pool.free_segments,
            &vec![
                SegmentHandle { offset: 0, size: 4 },
                SegmentHandle {
                    offset: 12,
                    size: 4,
                },
            ],
        )?;

        pool.deallocate(&handle_2);

        assert_eq!(pool.occupied_segments.len(), 0);

        bad_cmp_vec_unordered(
            &pool.free_segments,
            &vec![SegmentHandle {
                offset: 0,
                size: 16,
            }],
        )?;

        Ok(())
    }

    #[test]
    fn test_panic() {
        let (mut mem, mut pool) = init();
        assert!(std::panic::catch_unwind(move || {
            pool.allocate_from_buffer(&mut mem, &[0xFF; 16], 17, 1);
        })
        .is_err());

        let (mut mem, mut pool) = init();
        pool.allocate_from_buffer(&mut mem, &[0xFF; 16], 1, 1);
        assert!(std::panic::catch_unwind(move || {
            pool.allocate_from_buffer(&mut mem, &[0xFF; 16], 16, 1);
        })
        .is_err());
    }
}
