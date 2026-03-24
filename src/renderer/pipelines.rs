use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages, Device, Queue,
    ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::{
    camera::{PerspectiveProj, ViewProjectionMatrix},
    renderer::pipelines::shadow_mapping::NUM_CASCADES,
};

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
    light_direction: Vec3,
    _padding: u32,
    /// Represented as vecXf, which mandates that NUM_CASCADES is within 2..=4
    cascades_far_distances: [f32; NUM_CASCADES],
}

/// Binding for ubiquitous data, currently only the view projection matrix.
pub struct GlobalsBinding {
    state: Globals,
    globals_buffer: Buffer,
    pub layout: BindGroupLayout,
    pub binding: BindGroup,
}

impl GlobalsBinding {
    pub fn new(device: &Device, camera_projection: PerspectiveProj) -> Self {
        let state = Globals {
            view_proj: ViewProjectionMatrix::default(),
            light_view_projections: [ViewProjectionMatrix::default(); NUM_CASCADES],
            light_direction: Vec3::ZERO,
            _padding: 0,
            cascades_far_distances: shadow_mapping::compute_frustum_slice_z_far(camera_projection),
        };
        let globals_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("globals buffer"),
            contents: bytemuck::bytes_of(&state),
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
            state,
            globals_buffer,
            layout,
            binding,
        }
    }

    pub fn update(
        &mut self,
        queue: &Queue,
        camera_view_projection: ViewProjectionMatrix,
        light_view_projections: [ViewProjectionMatrix; NUM_CASCADES],
        light_direction: Vec3,
    ) {
        let new_state = Globals {
            view_proj: camera_view_projection,
            light_view_projections,
            light_direction,
            ..self.state
        };
        self.state = new_state;
        queue.write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&self.state));
    }
}
