use std::{collections::BTreeMap, marker::PhantomData, num::NonZero};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, Device,
    ShaderStages,
};

pub struct MultiDrawIndirectBuffer<Vertex: Pod, Uniform: Pod> {
    pub indirect_buffer: Buffer,
    pub vertex_buffer: Buffer,
    pub uniform_buffer: Buffer,
    pub uniform_bind_group_layout: BindGroupLayout,
    pub uniform_bind_group: BindGroup,
    pub batches_count: u64,
    occupied_regions: BTreeMap<u64, u64>,
    contiguous_regions: BTreeMap<u64, u64>,
    phantom_v: PhantomData<Vertex>,
    phantom_u: PhantomData<Uniform>,
}

const DRAW_ARGS_SIZE: usize = std::mem::size_of::<DrawIndirectArgs>();

impl<Vertex: Pod, Uniform: Pod> MultiDrawIndirectBuffer<Vertex, Uniform> {
    pub fn new(
        device: &Device,
        label: &str,
        initial_batches: Vec<(&[Vertex], Uniform)>,
        batches_count: u64,
    ) -> Self {
        if batches_count < initial_batches.len() as u64 {
            panic!(
                "`batches_count` {} smaller than `initial_batches` length {}",
                batches_count,
                initial_batches.len()
            )
        };

        let vertex_stride = std::mem::size_of::<Vertex>();
        let uniform_stride = std::mem::size_of::<Uniform>();

        let max_batch_size = initial_batches
            .iter()
            .map(|batch| batch.0.len() as u64)
            .max()
            .expect("`initial_batches` is empty");

        // Estimated buffer size is batches_count * max_batches * 1.5
        let vertex_buffer_size_heuristics =
            vertex_stride as u64 * batches_count * (max_batch_size + max_batch_size >> 2);

        let indirect_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("indirect buffer ".to_owned() + label)),
            usage: BufferUsages::INDIRECT,
            size: batches_count * DRAW_ARGS_SIZE as u64,
            mapped_at_creation: true,
        });
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("vertex buffer ".to_owned() + label)),
            size: vertex_buffer_size_heuristics,
            usage: BufferUsages::VERTEX,
            mapped_at_creation: true,
        });
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("chunk uniform buffer ".to_owned() + label)),
            size: batches_count * uniform_stride as u64,
            usage: BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });

        let mut indirect_buffer_view = indirect_buffer.slice(..).get_mapped_range_mut();
        let mut vertex_buffer_view = vertex_buffer.slice(..).get_mapped_range_mut();
        let mut uniform_buffer_view = uniform_buffer.slice(..).get_mapped_range_mut();

        let mut stored_batches = 0;
        let mut instance_count = 0u32;

        for (vertex_slice, uniform) in initial_batches.iter() {
            let indirect_buffer_range =
                (stored_batches * DRAW_ARGS_SIZE)..((stored_batches + 1) * DRAW_ARGS_SIZE);

            let vertex_buffer_range = (instance_count as usize * vertex_stride)
                ..((instance_count as usize + vertex_slice.len()) * vertex_stride);

            let uniform_buffer_range = (stored_batches as usize * uniform_stride)
                ..((stored_batches as usize + 1) * uniform_stride);

            let draw_args = DrawIndirectArgs {
                vertex_count: 4,
                instance_count: vertex_slice.len() as u32,
                first_vertex: 0,
                first_instance: instance_count,
            };

            indirect_buffer_view[indirect_buffer_range]
                .copy_from_slice(bytemuck::bytes_of(&draw_args));
            vertex_buffer_view[vertex_buffer_range]
                .copy_from_slice(bytemuck::cast_slice(*vertex_slice));
            uniform_buffer_view[uniform_buffer_range].copy_from_slice(bytemuck::bytes_of(uniform));

            stored_batches += 1;
            instance_count += vertex_slice.len() as u32;
        }

        drop(indirect_buffer_view);
        drop(vertex_buffer_view);
        drop(uniform_buffer_view);
        indirect_buffer.unmap();
        vertex_buffer.unmap();
        uniform_buffer.unmap();

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("uniform bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("uniform bind group layout"),
            layout: &uniform_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            indirect_buffer,
            vertex_buffer,
            uniform_buffer,
            uniform_bind_group_layout,
            uniform_bind_group,
            batches_count,
            contiguous_regions: BTreeMap::new(),
            occupied_regions: BTreeMap::new(),
            phantom_v: PhantomData,
            phantom_u: PhantomData,
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
