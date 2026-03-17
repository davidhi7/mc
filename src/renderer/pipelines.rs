use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages, Device, Queue,
    ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::{camera::ViewProjectionMatrix, renderer::pipelines::shadow_mapping::NUM_CASCADES};

pub mod block_outlines;
pub mod debug_crosshair;
pub mod frustum_culling;
pub mod shadow_mapping;
pub mod terrain;

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct Globals {
    view_proj: ViewProjectionMatrix,
    light_view_projections: [ViewProjectionMatrix; NUM_CASCADES],
    light_direction: Vec4,
}

/// Binding for ubiquitous data, currently only the view projection matrix.
pub struct GlobalsBinding {
    globals_buffer: Buffer,
    pub layout: BindGroupLayout,
    pub binding: BindGroup,
}

impl GlobalsBinding {
    pub fn new(device: &Device) -> Self {
        let globals_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("globals buffer"),
            contents: bytemuck::bytes_of(&Globals {
                view_proj: ViewProjectionMatrix::default(),
                light_view_projections: [ViewProjectionMatrix::default(); NUM_CASCADES],
                light_direction: Vec4::ZERO,
            }),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("globals layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let binding = device.create_bind_group(&BindGroupDescriptor {
            label: Some("globals binding"),
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        Self {
            globals_buffer,
            layout,
            binding,
        }
    }

    pub fn update(
        &self,
        queue: &Queue,
        camera_view_projection: ViewProjectionMatrix,
        light_view_projections: [ViewProjectionMatrix; NUM_CASCADES],
        light_direction: Vec3,
    ) {
        queue.write_buffer(
            &self.globals_buffer,
            0,
            bytemuck::bytes_of(&Globals {
                view_proj: camera_view_projection,
                light_view_projections,
                light_direction: light_direction.extend(0.0),
            }),
        );
    }
}
