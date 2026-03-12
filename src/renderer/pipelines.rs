use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, vec3};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages, Device, Queue,
    ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::camera::{Perspective, View, ViewProjectionMatrix};

pub mod block_outlines;
pub mod debug_crosshair;
pub mod frustum_culling;
pub mod shadow_mapping;
pub mod terrain;

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct Globals {
    view_proj: ViewProjectionMatrix,
    light_view_proj: ViewProjectionMatrix,
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
                light_view_proj: ViewProjectionMatrix::default(),
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

    pub fn update(&self, queue: &Queue, view: View, perspective: Perspective) {
        queue.write_buffer(
            &self.globals_buffer,
            0,
            bytemuck::bytes_of(&Globals {
                view_proj: ViewProjectionMatrix::new(view, perspective),
                light_view_proj: ViewProjectionMatrix::from_matrix(
                    Mat4::orthographic_lh(-100.0, 100.0, -100.0, 100.0, 0.0, 1000.0)
                        * Mat4::look_to_lh(
                            vec3(16.0, 100.0, 16.0),
                            vec3(0.5, -1.0, -0.2).normalize(),
                            vec3(0.1, 1.0, 0.0).normalize(),
                        ),
                ),
            }),
        );
    }
}
