use wgpu::{
    util::{DrawIndexedIndirectArgs, DrawIndirectArgs},
    Buffer, CommandEncoder, Queue,
};

pub mod block_allocator;
pub mod pool_allocator;

pub fn align_up(number: u64, alignment: u64) -> u64 {
    let delta = number % alignment;
    if delta == 0 {
        number
    } else {
        number + (alignment - delta)
    }
}

pub trait AsBytes {
    fn get_bytes(&self) -> &[u8];
}

impl AsBytes for DrawIndirectArgs {
    fn get_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl AsBytes for DrawIndexedIndirectArgs {
    fn get_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}

pub trait MemoryTarget<T> {
    fn write(&mut self, offset: u64, data: &[u8]);
    fn copy_from_buffer(
        &mut self,
        source: &T,
        source_offset: u64,
        destination_offset: u64,
        copy_size: u64,
    );
}

pub struct BufferMemoryTarget<'a> {
    buffer: &'a Buffer,
    queue: &'a Queue,
    command_encoder: &'a mut CommandEncoder,
}

impl<'a> BufferMemoryTarget<'a> {
    pub fn new(
        buffer: &'a Buffer,
        queue: &'a Queue,
        command_encoder: &'a mut CommandEncoder,
    ) -> Self {
        Self {
            buffer,
            queue,
            command_encoder,
        }
    }
}

impl<'a> MemoryTarget<Buffer> for BufferMemoryTarget<'a> {
    fn write(&mut self, offset: u64, data: &[u8]) {
        self.queue.write_buffer(&self.buffer, offset, data);
    }

    fn copy_from_buffer(
        &mut self,
        source: &Buffer,
        source_offset: u64,
        destination_offset: u64,
        copy_size: u64,
    ) {
        self.command_encoder.copy_buffer_to_buffer(
            source,
            source_offset,
            &self.buffer,
            destination_offset,
            copy_size,
        );
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub struct TestMemoryTarget<const N: usize> {
        pub memory: [u8; N],
    }

    impl<const N: usize> MemoryTarget<[u8; N]> for TestMemoryTarget<N> {
        fn write(&mut self, offset: u64, data: &[u8]) {
            self.memory[(offset as usize)..(offset as usize + data.len())].copy_from_slice(data);
        }

        fn copy_from_buffer(
            &mut self,
            source: &[u8; N],
            source_offset: u64,
            destination_offset: u64,
            copy_size: u64,
        ) {
            self.memory[(destination_offset as usize)..((destination_offset + copy_size) as usize)]
                .copy_from_slice(
                    &source[(source_offset as usize)..(source_offset + copy_size) as usize],
                );
        }
    }

    #[test]
    fn test_zero_alignment() {
        for alignment in [1, 2, 4, 8] {
            assert_eq!(0, align_up(0, alignment));
            assert_eq!(alignment, align_up(alignment, alignment));
            assert_eq!(5 * alignment, align_up(5 * alignment, alignment));
        }
    }

    #[test]
    fn test_nonzero_alignment() {
        assert_eq!(8, align_up(4, 8));
        assert_eq!(16, align_up(9, 8));
    }
}
