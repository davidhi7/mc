use std::mem;

use bytemuck::{Pod, Zeroable};
use glam::{swizzles::*, vec2, vec3, Vec2, Vec3, Vec4};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, Buffer, BufferAddress, BufferUsages, Device, ShaderStages,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode,
};

use crate::world::blocks::Direction;

pub const QUAD_VERTEX_COUNT: u32 = 4;

#[derive(Debug, Clone, Copy)]
struct Vertex {
    pub position: Vec3,
    pub tex_coordinates: Vec2,
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct AlignedVertex {
    position: Vec4,
    tex_coordinates: Vec4,
}
impl From<Vertex> for AlignedVertex {
    fn from(value: Vertex) -> Self {
        Self {
            position: value.position.extend(0.0),
            tex_coordinates: value.tex_coordinates.extend(0.0).extend(0.0),
        }
    }
}

// Quad facing in negative Z direction
const QUAD_VERTICES: &[Vertex] = &[
    Vertex {
        position: vec3(0.0, 0.0, 0.0),
        tex_coordinates: vec2(0.0, 1.0),
    },
    Vertex {
        position: vec3(0.0, 1.0, 0.0),
        tex_coordinates: vec2(0.0, 0.0),
    },
    Vertex {
        position: vec3(1.0, 0.0, 0.0),
        tex_coordinates: vec2(1.0, 1.0),
    },
    Vertex {
        position: vec3(1.0, 1.0, 0.0),
        tex_coordinates: vec2(1.0, 0.0),
    },
];

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct QuadInstance {
    /// Bits starting from the LSB:
    /// * `0-5`: x coordinate inside the cunk
    /// * `5-10`: y coordinate inside the cunk
    /// * `10-15`: z coordinate inside the cunk
    /// * `15-23`: texture id
    /// * `23-26`: direction (`crate::world::blocks::Direction`)
    pub attributes: u32,
    /// Bits starting from the LSB:
    /// * `0-2`: AO factor for first vertex
    /// * `2-4`: AO factor for second vertex
    /// * `4-6`: AO factor for third vertex
    /// * `6-8`: AO factor for forth vertex
    pub ao_attributes: u32,
}
impl QuadInstance {
    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: 2 * mem::size_of::<u32>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &[
                VertexAttribute {
                    offset: 0 as BufferAddress,
                    shader_location: 0,
                    format: VertexFormat::Uint32,
                },
                VertexAttribute {
                    offset: mem::size_of::<u32>() as BufferAddress,
                    shader_location: 1,
                    format: VertexFormat::Uint32,
                },
            ],
        }
    }
}

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct TransparentQuadInstance {
    /// Bits starting from the LSB:
    /// * `0-5`: x coordinate inside the cunk
    /// * `5-10`: y coordinate inside the cunk
    /// * `10-15`: z coordinate inside the cunk
    /// * `15-23`: texture id
    /// * `23-26`: direction (`crate::world::blocks::Direction`)
    pub attributes: u32,
}
impl TransparentQuadInstance {
    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: mem::size_of::<u32>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &[VertexAttribute {
                offset: 0 as BufferAddress,
                shader_location: 0,
                format: VertexFormat::Uint32,
            }],
        }
    }
}

pub fn create_vertex_buffer(device: &Device) -> Buffer {
    let mut quad_variants: Vec<Vertex> = Vec::with_capacity(4 * 2 * 6);

    let flipped_quad_vertices = QUAD_VERTICES
        .iter()
        .map(|vertex| flip_quad_vertex(vertex.to_owned()));

    for direction in Direction::into_iter() {
        quad_variants.extend(
            QUAD_VERTICES
                .iter()
                .map(|vertex| swizzle_vertex(direction, vertex.to_owned())),
        );

        quad_variants.extend(
            flipped_quad_vertices
                .clone()
                .map(|vertex| swizzle_vertex(direction, vertex)),
        );
    }

    let aligned_quad_variants: Vec<AlignedVertex> =
        quad_variants.into_iter().map(AlignedVertex::from).collect();

    device.create_buffer_init(&BufferInitDescriptor {
        label: Some("quad vertices uniform buffer"),
        contents: bytemuck::cast_slice(aligned_quad_variants.as_slice()),
        usage: BufferUsages::UNIFORM,
    })
}

fn swizzle_vertex(direction: Direction, vertex: Vertex) -> Vertex {
    let mut v = vertex.clone();
    match direction {
        Direction::NegX => {
            // -X
            v.position = vec3(0.0, v.position.x, v.position.y);
            v.tex_coordinates = v.tex_coordinates.yx();
        }
        Direction::X => {
            // +X
            v.position = vec3(1.0, v.position.y, v.position.x);
        }
        Direction::NegY => {
            // -Y
            v.position = vec3(v.position.y, 0.0, v.position.x);
        }
        Direction::Y => {
            // +Y
            v.position = vec3(v.position.x, 1.0, v.position.y);
        }
        Direction::NegZ => {
            // NegZ is the default direction of the quad, so do nothing
        }
        Direction::Z => {
            // +Z
            v.position = vec3(v.position.y, v.position.x, 1.0);
            v.tex_coordinates = v.tex_coordinates.yx();
        }
    };
    v
}

fn flip_quad_vertex(vertex: Vertex) -> Vertex {
    let mut v = vertex.clone();

    // Effectively rotate the line separating thetwo triangles that form a quad
    // Relevant for AO interpolation in some cases
    v.position.x = (1.0 - v.position.x).abs();
    v.position = v.position.yxz();

    v.tex_coordinates.x = (1.0 - v.tex_coordinates.x).abs();
    v.tex_coordinates = v.tex_coordinates.yx();

    v
}

pub fn get_bind_group(device: &Device) -> (BindGroupLayout, BindGroup) {
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("quad vertices bind group layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("quad vertices bind group"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: create_vertex_buffer(device).as_entire_binding(),
        }],
    });

    (bind_group_layout, bind_group)
}
