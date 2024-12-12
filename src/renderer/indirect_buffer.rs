use core::panic;
use std::{collections::BTreeMap, marker::PhantomData};

use std::fmt::Debug;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, Device, Queue,
    ShaderStages,
};

pub struct MultiDrawIndirectBuffer<Vertex: Pod, Uniform: Pod + Debug> {
    pub indirect_buffer: Buffer,
    pub vertex_buffer: Buffer,
    pub uniform_buffer: Buffer,
    pub uniform_bind_group_layout: BindGroupLayout,
    pub uniform_bind_group: BindGroup,
    pub occupied_regions: BTreeMap<u64, BufferRegion<Uniform>>,
    empty_regions: BTreeMap<u64, u64>,
    max_occupied_regions_count: usize,
    phantom_v: PhantomData<Vertex>,
    phantom_u: PhantomData<Uniform>,
}

const DRAW_ARGS_SIZE: usize = std::mem::size_of::<DrawIndirectArgs>();

impl<Vertex: Pod, Uniform: Pod + Debug> MultiDrawIndirectBuffer<Vertex, Uniform> {
    pub fn new(
        device: &Device,
        label: &str,
        initial_batches: Vec<(&[Vertex], Uniform)>,
        batches_count: u64,
    ) -> Self {
        let mut occupied_regions = BTreeMap::new();
        let mut empty_regions = BTreeMap::new();
        if batches_count < initial_batches.len() as u64 {
            panic!(
                "`batches_count` {} smaller than `initial_batches` length {}",
                batches_count,
                initial_batches.len()
            )
        };

        let vertex_stride = std::mem::size_of::<Vertex>();
        let uniform_stride = std::mem::size_of::<Uniform>();

        let max_batch_size: u64 = initial_batches
            .iter()
            .map(|batch| batch.0.len() as u64)
            .max()
            .expect("`initial_batches` is empty");

        // Estimated buffer size is batches_count * max_batches * 1.5
        let vertex_buffer_size_heuristics =
            vertex_stride as u64 * batches_count * (max_batch_size + max_batch_size >> 2);

        let indirect_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("indirect buffer ".to_owned() + label)),
            usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
            size: batches_count * DRAW_ARGS_SIZE as u64,
            mapped_at_creation: true,
        });
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("vertex buffer ".to_owned() + label)),
            size: vertex_buffer_size_heuristics,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("chunk uniform buffer ".to_owned() + label)),
            size: batches_count * uniform_stride as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        let mut indirect_buffer_view = indirect_buffer.slice(..).get_mapped_range_mut();
        let mut vertex_buffer_view = vertex_buffer.slice(..).get_mapped_range_mut();
        let mut uniform_buffer_view = uniform_buffer.slice(..).get_mapped_range_mut();

        let mut stored_batches = 0;
        let mut stored_instances = 0;

        for (vertex_slice, uniform) in initial_batches.iter() {
            let indirect_buffer_range =
                (stored_batches * DRAW_ARGS_SIZE)..((stored_batches + 1) * DRAW_ARGS_SIZE);

            let vertex_buffer_range = (stored_instances * vertex_stride)
                ..((stored_instances + vertex_slice.len()) * vertex_stride);

            let uniform_buffer_range =
                (stored_batches as usize * uniform_stride)..((stored_batches + 1) * uniform_stride);

            let draw_args = DrawIndirectArgs {
                vertex_count: 4,
                instance_count: vertex_slice.len() as u32,
                first_vertex: 4 * stored_batches as u32,
                first_instance: stored_instances as u32,
            };

            indirect_buffer_view[indirect_buffer_range]
                .copy_from_slice(bytemuck::bytes_of(&draw_args));
            vertex_buffer_view[vertex_buffer_range]
                .copy_from_slice(bytemuck::cast_slice(*vertex_slice));
            uniform_buffer_view[uniform_buffer_range].copy_from_slice(bytemuck::bytes_of(uniform));

            occupied_regions.insert(
                stored_instances as u64,
                BufferRegion {
                    vb_location: stored_instances as u64,
                    vb_size: vertex_slice.len() as u64,
                    ib_location: stored_batches as u64,
                    uniform: *uniform,
                },
            );

            stored_batches += 1;
            stored_instances += vertex_slice.len() as usize;
        }

        empty_regions.insert(
            stored_instances as u64,
            vertex_buffer_size_heuristics - stored_instances as u64,
        );

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
                        ty: BufferBindingType::Storage { read_only: true },
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
            occupied_regions,
            empty_regions,
            phantom_v: PhantomData,
            phantom_u: PhantomData,
            max_occupied_regions_count: batches_count as usize,
        }
    }

    pub fn drop_region(&mut self, queue: &Queue, target_region: &BufferRegion<Uniform>) {
        // TODO don't use ib_location as primary ID for this object
        let target_region = self
            .occupied_regions
            .iter()
            .filter(|(loc, region)| region.vb_location == target_region.vb_location)
            .last()
            .unwrap()
            .1;
        let target_region = *target_region;

        // If the region doesn't own the last indirect/uniform buffer slot, fill the slot with another region
        if target_region.ib_location < self.occupied_regions.len() as u64 - 1 {
            let (.., last_region) = self
                .occupied_regions
                .iter()
                .max_by_key(|(.., region)| region.ib_location)
                .expect("`occupied_regions` empty");

            // Move last draw call & uniform to new empty spot, update draw call parameters
            queue.write_buffer(
                &self.indirect_buffer,
                target_region.ib_location * std::mem::size_of::<DrawIndirectArgs>() as u64,
                bytemuck::bytes_of(&DrawIndirectArgs {
                    vertex_count: 4,
                    instance_count: last_region.vb_size as u32,
                    first_vertex: 4 * target_region.ib_location as u32,
                    first_instance: last_region.vb_location as u32,
                }),
            );

            queue.write_buffer(
                &self.uniform_buffer,
                target_region.ib_location * std::mem::size_of::<Uniform>() as u64,
                bytemuck::bytes_of(&last_region.uniform),
            );

            self.occupied_regions
                .entry(last_region.vb_location)
                .and_modify(|entry| entry.ib_location = target_region.ib_location);
        }

        self.occupied_regions
            .remove(&target_region.vb_location)
            .expect("Region `vb_location` key missing in `occupied_regions`");

        // Mark region as empty
        // TODO check whether to enforce this
        let empty_region_after = self
            .empty_regions
            .remove(&(target_region.vb_location + target_region.vb_size));

        let empty_region_before = self
            .empty_regions
            .iter()
            .filter(|(vb_location, size)| **vb_location + **size == target_region.vb_location)
            .last();

        let mut new_region_location = target_region.vb_location;
        let mut new_region_size: u64 = target_region.vb_size + empty_region_after.unwrap_or(0);

        if let Some((location, size)) = empty_region_before {
            new_region_location = *location;
            new_region_size += size;
        }
        self.empty_regions
            .insert(new_region_location, new_region_size);
    }

    pub fn insert_region(
        &mut self,
        queue: &Queue,
        batch: (&[Vertex], Uniform),
    ) -> BufferRegion<Uniform> {
        if self.max_occupied_regions_count < self.occupied_regions.len() + 1 {
            panic!(
                "Not enough indirect buffer space available for {} regions",
                self.occupied_regions.len() + 1
            );
        }

        let (new_region_location, new_region_size) = self
            .empty_regions
            .iter()
            .filter(|(.., size)| **size >= batch.0.len() as u64)
            .min_by_key(|(.., size)| **size)
            .expect(&format!(
                "Not enough vertex buffer space available for region of size {}",
                batch.0.len()
            ));

        let new_region_location = *new_region_location;
        let new_region_size = *new_region_size;

        let ib_location = self.occupied_regions.len() as u64;
        let region: BufferRegion<Uniform> = BufferRegion {
            vb_location: new_region_location,
            vb_size: batch.0.len() as u64,
            ib_location,
            uniform: batch.1,
        };

        self.occupied_regions.insert(new_region_location, region);
        queue.write_buffer(
            &self.indirect_buffer,
            ib_location * std::mem::size_of::<DrawIndirectArgs>() as u64,
            bytemuck::bytes_of(&DrawIndirectArgs {
                vertex_count: 4,
                instance_count: region.vb_size as u32,
                first_vertex: 4 * ib_location as u32,
                first_instance: region.vb_location as u32,
            }),
        );

        queue.write_buffer(
            &self.uniform_buffer,
            ib_location * std::mem::size_of::<Uniform>() as u64,
            bytemuck::bytes_of(&batch.1),
        );
        queue.write_buffer(
            &self.vertex_buffer,
            region.vb_location * std::mem::size_of::<Vertex>() as u64,
            bytemuck::cast_slice(batch.0),
        );

        self.empty_regions.remove(&new_region_location);
        if new_region_size != region.vb_size {
            self.empty_regions.insert(
                new_region_location + region.vb_size,
                new_region_size - region.vb_size,
            );
        }

        region
    }

    pub fn draw_count(&self) -> u32 {
        self.occupied_regions.len() as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferRegion<Uniform: Pod> {
    // TODO unset public
    pub vb_location: u64,
    pub vb_size: u64,
    pub ib_location: u64,
    pub uniform: Uniform,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DrawIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}
