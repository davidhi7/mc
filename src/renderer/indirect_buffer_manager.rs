use core::panic;
use std::collections::HashMap;
use std::hash::Hash;
use std::{collections::BTreeMap, marker::PhantomData};

use std::fmt::Debug;

use bytemuck::Pod;
use wgpu::util::DrawIndirectArgs;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoder, Device, Queue, ShaderStages,
};

pub trait DrawCallBucket: Copy + Eq + Hash {
    /// Size of a single instance, in bytes
    fn instance_size(&self) -> u64;
}

pub struct MultiDrawIndirectBuffer<Uniform: Pod + Debug, Bucket: DrawCallBucket> {
    pub indirect_buffer: Buffer,
    pub vertex_buffer: Buffer,
    uniform_buffer: Buffer,
    pub uniform_bind_group_layout: BindGroupLayout,
    pub uniform_bind_group: BindGroup,
    occupied_regions: BTreeMap<BufferRegion, BufferRegionData<Uniform, Bucket>>,
    empty_regions: BTreeMap<u64, u64>,
    batches_count: usize,
    phantom_uniform: PhantomData<Uniform>,
    phantom_bucket: PhantomData<Bucket>,
}

const DRAW_ARGS_SIZE: usize = std::mem::size_of::<DrawIndirectArgs>();

impl<Uniform: Pod + Debug, Bucket: DrawCallBucket> MultiDrawIndirectBuffer<Uniform, Bucket> {
    pub fn new(
        device: &Device,
        label: &str,
        buckets: &[Bucket],
        batches_count: usize,
        max_batch_size_map: &HashMap<Bucket, u64>,
    ) -> Self {
        let mut empty_regions = BTreeMap::new();

        let mut vertex_buffer_size_bytes = 0;
        for bucket in buckets {
            vertex_buffer_size_bytes += batches_count as u64
                * bucket.instance_size()
                * *max_batch_size_map
                    .get(bucket)
                    .expect("Bucket not valid key in `average_batch_size_map`");
        }

        let indirect_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("indirect buffer ".to_owned() + label)),
            size: (batches_count * DRAW_ARGS_SIZE) as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("vertex buffer ".to_owned() + label)),
            size: vertex_buffer_size_bytes,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&("chunk uniform buffer ".to_owned() + label)),
            size: (batches_count * std::mem::size_of::<Uniform>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        empty_regions.insert(0, vertex_buffer_size_bytes);

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("uniform bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
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
            occupied_regions: BTreeMap::new(),
            empty_regions,
            batches_count,
            phantom_uniform: PhantomData,
            phantom_bucket: PhantomData,
        }
    }

    pub fn drop_region(&mut self, queue: &Queue, target_region: &BufferRegion) {
        // TODO is clone neccessary here?
        let region_data = self
            .occupied_regions
            .get(target_region)
            .expect("TODO")
            .clone();

        // If the region doesn't own the last indirect/uniform buffer slot, fill the slot with another region
        if region_data.indirect_buffer_slot < self.occupied_regions.len() as u64 - 1 {
            // Find BufferRegionData instance with highest indirect buffer slot
            let (.., last_region) = self
                .occupied_regions
                .iter()
                .max_by_key(|(.., region)| region.indirect_buffer_slot)
                .expect("`occupied_regions` empty");

            // Move last draw call & uniform to new empty spot, write to buffers
            // TODO alignment
            queue.write_buffer(
                &self.indirect_buffer,
                region_data.indirect_buffer_slot * std::mem::size_of::<DrawIndirectArgs>() as u64,
                &DrawIndirectArgs {
                    vertex_count: 4,
                    instance_count: (last_region.region.vb_size
                        / last_region.bucket.instance_size())
                        as u32,
                    first_vertex: 4 * region_data.indirect_buffer_slot as u32,
                    first_instance: (last_region.region.vb_location
                        / last_region.bucket.instance_size())
                        as u32,
                }
                .as_bytes(),
            );

            queue.write_buffer(
                &self.uniform_buffer,
                region_data.indirect_buffer_slot * std::mem::size_of::<Uniform>() as u64,
                bytemuck::bytes_of(&last_region.uniform),
            );

            self.occupied_regions
                .entry(last_region.region)
                .and_modify(|entry| entry.indirect_buffer_slot = region_data.indirect_buffer_slot);
        }

        self.occupied_regions
            .remove(target_region)
            .expect("Region `target_region` key missing in `occupied_regions`");

        // Mark region as empty
        // TODO check whether to enforce this
        let following_empty_region = self
            .empty_regions
            .remove(&(&region_data.region.vb_location + &region_data.region.vb_size));

        let empty_region_before = self
            .empty_regions
            .iter()
            .filter(|(vb_location, vb_size)| {
                **vb_location + **vb_size == region_data.region.vb_location
            })
            .last();

        let mut new_region_location = region_data.region.vb_location;
        let mut new_region_size: u64 =
            region_data.region.vb_size + following_empty_region.unwrap_or(0);

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
        command_encoder: &mut CommandEncoder,
        bucket: Bucket,
        batch_vb: &Buffer,
        batch_instance_count: u64,
        uniform: Uniform,
    ) -> BufferRegion {
        if self.batches_count < self.occupied_regions.len() + 1 {
            panic!(
                "Not enough indirect buffer space available for {} regions",
                self.occupied_regions.len() + 1
            );
        }

        let (new_region_offset, new_region_size) = self
            .empty_regions
            .iter()
            .filter(|(.., size)| **size >= batch_instance_count * bucket.instance_size())
            .min_by_key(|(.., size)| **size)
            .expect(&format!(
                "Not enough vertex buffer space available for region of size {}",
                batch_instance_count * bucket.instance_size()
            ));

        let new_region_offset = *new_region_offset;
        let new_region_size = *new_region_size;

        let indirect_buffer_slot = self.occupied_regions.len() as u64;
        let region_data = BufferRegionData {
            region: BufferRegion {
                vb_location: new_region_offset,
                vb_size: batch_instance_count * bucket.instance_size(),
            },
            indirect_buffer_slot,
            uniform,
            bucket,
        };

        // TODO handle alignment!
        self.occupied_regions
            .insert(region_data.region, region_data);
        queue.write_buffer(
            &self.indirect_buffer,
            indirect_buffer_slot * std::mem::size_of::<DrawIndirectArgs>() as u64,
            &DrawIndirectArgs {
                vertex_count: 4,
                instance_count: batch_instance_count as u32,
                first_vertex: 4 * indirect_buffer_slot as u32,
                first_instance: region_data.region.vb_location as u32
                    / bucket.instance_size() as u32,
            }
            .as_bytes(),
        );
        queue.write_buffer(
            &self.uniform_buffer,
            indirect_buffer_slot * std::mem::size_of::<Uniform>() as u64,
            bytemuck::bytes_of(&uniform),
        );
        command_encoder.copy_buffer_to_buffer(
            batch_vb,
            0,
            &self.vertex_buffer,
            new_region_offset,
            batch_instance_count * bucket.instance_size(),
        );

        self.empty_regions.remove(&new_region_offset);
        if new_region_size > region_data.region.vb_size {
            self.empty_regions.insert(
                new_region_offset + region_data.region.vb_size,
                new_region_size - region_data.region.vb_size,
            );
        }

        region_data.region
    }

    pub fn draw_count(&self) -> u32 {
        self.occupied_regions.len() as u32
    }
}

/// Struct describing a segment in the vertex/instance buffer
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferRegion {
    /// Offset of the vertex/instance buffer segment, in bytes
    vb_location: u64,
    /// Size of the vertex/instance buffer segment, in bytes
    vb_size: u64,
}

impl PartialOrd for BufferRegion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.vb_location.partial_cmp(&other.vb_location) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.vb_size.partial_cmp(&other.vb_size)
    }
}

impl Ord for BufferRegion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Struct describing a segment in the vertex/instance buffer as well as indirect- and uniform buffer
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferRegionData<Uniform: Pod, Bucket: DrawCallBucket> {
    /// Data describing associated vertex/instance buffer segment
    region: BufferRegion,
    /// Slot of the corresponding draw call and uniform entry in the indirect/uniform buffers
    indirect_buffer_slot: u64,
    /// Uniform data
    uniform: Uniform,
    /// Bucket type
    bucket: Bucket,
}
