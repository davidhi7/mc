use std::mem;

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3, Vec4};
use wgpu::{
    BlendState, Buffer, BufferAddress, BufferDescriptor, BufferUsages, ColorTargetState,
    ColorWrites, CompareFunction, DepthBiasState, DepthStencilState, Device, FragmentState,
    FrontFace, MultisampleState, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology,
    Queue, RenderPass, RenderPipeline, RenderPipelineDescriptor, StencilState, TextureFormat,
    VertexAttribute, VertexBufferLayout, VertexState, VertexStepMode,
};

use crate::{renderer::pipelines::GlobalsBinding, shaders};

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
#[allow(dead_code)]
struct AlignedPureVertex {
    position: Vec4,
}

impl AlignedPureVertex {
    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: mem::size_of::<Vec4>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[VertexAttribute {
                offset: 0 as BufferAddress,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x4,
            }],
        }
    }
}

// Simple square with side length of 1.005 and center at (0.5, 0.5, 0.5)
// TODO prettier outlines aware of adjacent blocks
const BLOCK_OUTLINES: [Vec3; 24] = [
    // bottom face (z = -0.0025)
    Vec3::new(-0.0025, -0.0025, -0.0025),
    Vec3::new(1.0025, -0.0025, -0.0025),
    Vec3::new(1.0025, -0.0025, -0.0025),
    Vec3::new(1.0025, 1.0025, -0.0025),
    Vec3::new(1.0025, 1.0025, -0.0025),
    Vec3::new(-0.0025, 1.0025, -0.0025),
    Vec3::new(-0.0025, 1.0025, -0.0025),
    Vec3::new(-0.0025, -0.0025, -0.0025),
    // top face (z = 1.0025)
    Vec3::new(-0.0025, -0.0025, 1.0025),
    Vec3::new(1.0025, -0.0025, 1.0025),
    Vec3::new(1.0025, -0.0025, 1.0025),
    Vec3::new(1.0025, 1.0025, 1.0025),
    Vec3::new(1.0025, 1.0025, 1.0025),
    Vec3::new(-0.0025, 1.0025, 1.0025),
    Vec3::new(-0.0025, 1.0025, 1.0025),
    Vec3::new(-0.0025, -0.0025, 1.0025),
    // vertical edges
    Vec3::new(-0.0025, -0.0025, -0.0025),
    Vec3::new(-0.0025, -0.0025, 1.0025),
    Vec3::new(1.0025, -0.0025, -0.0025),
    Vec3::new(1.0025, -0.0025, 1.0025),
    Vec3::new(1.0025, 1.0025, -0.0025),
    Vec3::new(1.0025, 1.0025, 1.0025),
    Vec3::new(-0.0025, 1.0025, -0.0025),
    Vec3::new(-0.0025, 1.0025, 1.0025),
];

pub struct BlockOutlinePipeline {
    pipeline: RenderPipeline,
    vertex_buffer: Buffer,
    block: Option<IVec3>,
}

impl BlockOutlinePipeline {
    pub fn new(
        device: &Device,
        globals_binding: &GlobalsBinding,
        surface_format: TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(shaders::SHADER_BLOCK_OUTLINES);

        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("block outline vertex buffer"),
            size: 24 * AlignedPureVertex::desc().array_stride,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("block outline render pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("block outline render pipeline layout"),
                bind_group_layouts: &[&globals_binding.layout],
                push_constant_ranges: &[],
            })),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[AlignedPureVertex::desc()],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: FrontFace::Cw,
                cull_mode: None,
                unclipped_depth: false,
                // TODO ?
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            vertex_buffer,
            block: None,
        }
    }

    pub fn set_outlined_block(&mut self, queue: &Queue, block: Option<IVec3>) {
        match block {
            Some(block) => {
                let vertices = BLOCK_OUTLINES.map(|offset| AlignedPureVertex {
                    position: (block.as_vec3() + offset).extend(0.0),
                });

                queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
                self.block = Some(block);
            }
            None => self.block = None,
        }
    }

    pub fn render(&self, render_pass: &mut RenderPass, globals: &GlobalsBinding) {
        if self.block.is_none() {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_bind_group(0, &globals.binding, &[]);
        render_pass.draw(0..24, 0..1);
    }
}
