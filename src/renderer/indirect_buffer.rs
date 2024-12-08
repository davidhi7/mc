use std::{collections::BTreeMap, marker::PhantomData};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, BufferAddress, BufferDescriptor, BufferUsages, Device, Queue, VertexBufferLayout,
};

pub struct MultiDrawIndirectBuffer<T: Pod> {
    pub vertex_buffer: Buffer,
    pub indirect_buffer: Buffer,
    occupied_regions: BTreeMap<u64, u64>,
    contiguous_regions: BTreeMap<u64, u64>,
    phantom: PhantomData<T>,
}

impl<T: Pod> MultiDrawIndirectBuffer<T> {
    pub fn new(
        device: &Device,
        queue: &Queue,
        label: &str,
        initial_data: Vec<&[T]>,
        data_stride: BufferAddress,
    ) -> Self {
        // let ib = device.create_buffer(&BufferDescriptor {
        //     label: Some(&("indirect buffer ".to_owned() + label)),
        //     usage: BufferUsages::INDIRECT,
        //     size: initial_data.len() as u64 * 16,
        //     mapped_at_creation: true,
        // });
        // ib.slice(..).get_mapped_range_mut()

        let mut draw_indirect_args = Vec::new();
        let mut instance_count = 0;

        for region in initial_data.iter() {
            draw_indirect_args.push(DrawIndirectArgs {
                vertex_count: 4,
                instance_count: region.len() as u32,
                first_vertex: 0,
                first_instance: instance_count,
            });

            instance_count += region.len() as u32;
        }

        println!("{:?}", draw_indirect_args.as_slice());

        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("merged buffer ".to_owned() + label)),
            size: instance_count as u64 * data_stride,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(initial_data[0]));
        queue.write_buffer(
            &vertex_buffer,
            data_stride * initial_data[0].len() as u64,
            bytemuck::cast_slice(initial_data[1]),
        );

        let indirect_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&("indirect buffer ".to_owned() + label)),
            contents: bytemuck::cast_slice(draw_indirect_args.as_slice()),
            usage: BufferUsages::INDIRECT,
        });

        Self {
            vertex_buffer,
            indirect_buffer,
            contiguous_regions: BTreeMap::new(),
            occupied_regions: BTreeMap::new(),
            phantom: PhantomData,
        }
    }
}

pub struct BufferRegion {
    region_location: u64,
    region_size: u64,
    indirect_buffer_index: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DrawIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}
