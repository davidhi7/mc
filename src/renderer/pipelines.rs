use bytemuck::{Pod, Zeroable};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages, Device, Queue,
    ShaderStages,
};

use crate::world::camera::CameraController;

pub mod frustum_culling;
pub mod terrain;
pub mod ui;

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
struct Globals {
    view_proj: [[f32; 4]; 4],
}

pub struct GlobalsBinding {
    globals_buffer: Buffer,
    pub layout: BindGroupLayout,
    pub binding: BindGroup,
}

impl GlobalsBinding {
    pub fn new(device: &Device, camera: &CameraController) -> Self {
        let globals_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("globals buffer"),
            contents: bytemuck::bytes_of(&Globals {
                view_proj: camera.get_view_projection_matrix().to_cols_array_2d(),
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

    pub fn update(&self, queue: &Queue, camera: &CameraController) {
        queue.write_buffer(
            &self.globals_buffer,
            0,
            bytemuck::bytes_of(&Globals {
                view_proj: camera.get_view_projection_matrix().to_cols_array_2d(),
            }),
        );
    }
}
