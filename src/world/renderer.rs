use std::{
    any::Any,
    borrow::{Borrow, BorrowMut},
    f32::consts::PI,
    rc::Rc,
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    core::instance,
    naga::Handle,
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, Buffer, Device, Queue, RenderPass, RenderPipeline, SurfaceConfiguration,
};

use crate::{
    camera::CameraController,
    texture,
    world::{blocks::Direction, chunk::Chunk, World, CHUNK_DIMENSIONS, VERTICAL_CHUNK_COUNT},
};

const CHUNK_RENDER_DISTANCE: i32 = 4;

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct Vertex {
    pub position: [f32; 3],
    pub tex_coordinates: [f32; 2],
}
impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

const CUBE_FACE_VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.5, -0.5, 0.0],
        tex_coordinates: [0.0, 1.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.0],
        tex_coordinates: [0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        tex_coordinates: [1.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.0],
        tex_coordinates: [1.0, 0.0],
    },
];

#[derive(Debug, Copy, Clone)]
pub struct CubeFaceInstance {
    pub position: glam::Vec3,
    pub direction: Direction,
    pub tex_index: u32,
}

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct RawCubeFaceInstance {
    pub model_matrix: [[f32; 4]; 4],
    pub tex_index: u32,
    pub direction: u32,
}
impl RawCubeFaceInstance {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: (mem::size_of::<[[f32; 4]; 4]>() + 2 * mem::size_of::<u32>())
                as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: (mem::size_of::<[f32; 16]>() + mem::size_of::<u32>())
                        as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }

    pub fn from_cube_face(instance: CubeFaceInstance) -> Self {
        // TODO why do angles need to be inverted?
        let world_translation = glam::Mat4::from_translation(instance.position);
        let mat = world_translation
            * match instance.direction {
                Direction::X => {
                    glam::Mat4::from_translation(glam::vec3(1.0, 0.5, 0.5))
                        * glam::Mat4::from_rotation_y(-PI / 2.0)
                }
                Direction::NegX => {
                    glam::Mat4::from_translation(glam::vec3(0.0, 0.5, 0.5))
                        * glam::Mat4::from_rotation_y(PI / 2.0)
                }

                Direction::Y => {
                    glam::Mat4::from_translation(glam::vec3(0.5, 1.0, 0.5))
                        * glam::Mat4::from_rotation_x(PI / 2.0)
                }
                Direction::NegY => {
                    glam::Mat4::from_translation(glam::vec3(0.5, 0.0, 0.5))
                        * glam::Mat4::from_rotation_y(PI)
                        * glam::Mat4::from_rotation_x(-PI / 2.0)
                }

                Direction::Z => {
                    glam::Mat4::from_translation(glam::vec3(0.5, 0.5, 1.0))
                // TODO why is this needed?
                    * glam::Mat4::from_rotation_y(PI)
                }
                Direction::NegZ => {
                    glam::Mat4::from_translation(glam::vec3(0.5, 0.5, 0.0))
                    // TODO why is this not needed?
                    // * glam::Mat4::from_rotation_y(PI)
                }
            };
        Self {
            model_matrix: mat.to_cols_array_2d(),
            direction: instance.direction as u32,
            tex_index: instance.tex_index,
        }
    }
}

pub struct WorldRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    vertex_buffer: Buffer,
    instance_buffer: Buffer,
    pub camera_controller: CameraController,
    camera_uniform: Buffer,
    camera_bind_group: BindGroup,
    texture_bind_group: BindGroup,
    render_pipeline: RenderPipeline,
    buffer_capacity: usize,
    previous_camera_u: Option<i32>,
    previous_camera_w: Option<i32>,

    loading_thread_handle: Vec<JoinHandle<Vec<RawCubeFaceInstance>>>,
}

impl WorldRenderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface_config: &SurfaceConfiguration,
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube face vertex buffer"),
            contents: bytemuck::cast_slice(CUBE_FACE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // TODO use sensible default size, research `mapped_at_creation`
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cube face instance buffer"),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: 0,
            mapped_at_creation: false,
        });

        let camera_controller = CameraController::new(
            glam::Vec3::NEG_X,
            0.0,
            0.0,
            45.0,
            surface_config.width as f32 / surface_config.height as f32,
            0.1,
            1000.0,
            0.05,
            0.001,
        );

        let camera_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera uniform buffer"),
            contents: bytemuck::cast_slice(&[camera_controller.get_view_projection_matrix()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera bind group layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform.as_entire_binding(),
            }],
            label: Some("camera bind group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shader.wgsl").into()),
        });

        let (texture_bind_group_layout, texture_bind_group) =
            texture::load_textures(&device, &queue).unwrap();

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("world render pipeline layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("world render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), RawCubeFaceInstance::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        WorldRenderer {
            device,
            queue,
            vertex_buffer,
            instance_buffer,
            camera_controller,
            camera_uniform,
            camera_bind_group,
            texture_bind_group,
            render_pipeline,
            buffer_capacity: 0,
            previous_camera_u: None,
            previous_camera_w: None,

            loading_thread_handle: Vec::new(),
        }
    }

    pub fn update(&mut self, world: Arc<Mutex<World>>) {
        self.queue.write_buffer(
            &self.camera_uniform,
            0,
            bytemuck::cast_slice(&[self.camera_controller.get_view_projection_matrix()]),
        );

        if let Some(handle) = self.loading_thread_handle.pop() {
            if handle.is_finished() {
                let instances = handle.join().unwrap();
                if instances.len() > self.buffer_capacity {
                    self.instance_buffer.destroy();
                    self.instance_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
                        label: Some("cube face instance buffer"),
                        contents: bytemuck::cast_slice(instances.as_slice()),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
                    self.buffer_capacity = instances.len();
                } else {
                    self.queue.write_buffer(
                        &self.instance_buffer,
                        0,
                        bytemuck::cast_slice(instances.as_slice()),
                    );
                }
            } else {
                self.loading_thread_handle.push(handle);
            }
        }

        let camera_u = self.camera_controller.get_position().x as i32 / CHUNK_DIMENSIONS;
        let camera_w = self.camera_controller.get_position().z as i32 / CHUNK_DIMENSIONS;

        if self.previous_camera_u.is_some_and(|u| u == camera_u)
            && self.previous_camera_w.is_some_and(|w| w == camera_w)
        {
            return;
        }

        self.previous_camera_u = Some(camera_u);
        self.previous_camera_w = Some(camera_w);

        let device_clone = Arc::clone(&self.device);
        // let instance_buffer_clone = self.instance_buffer;

        let handle = thread::spawn(move || {
            let chunk_range_u =
                camera_u - CHUNK_RENDER_DISTANCE..camera_u + CHUNK_RENDER_DISTANCE + 1;
            let chunk_range_w =
                camera_w - CHUNK_RENDER_DISTANCE..camera_w + CHUNK_RENDER_DISTANCE + 1;

            let mut world_handle = world.lock().unwrap();

            for u in chunk_range_u.clone() {
                for w in chunk_range_w.clone() {
                    // TODO error handling
                    if let None = world_handle.chunk_columns.get(&(u, w)) {
                        world_handle.create_chunks(u, w);
                    }
                }
            }

            let mut instances: Vec<&RawCubeFaceInstance> = Vec::new();
            for u in chunk_range_u.clone() {
                for w in chunk_range_w.clone() {
                    let instances_ = (0..VERTICAL_CHUNK_COUNT)
                        .flat_map(|v| world_handle.meshed_chunks.get(&(u, v as u32, w)).unwrap())
                        .collect::<Vec<&RawCubeFaceInstance>>();

                    instances.extend(instances_);
                }
            }

            let instances: Vec<RawCubeFaceInstance> = instances
                .iter()
                .map(|instance| (*instance).to_owned())
                .collect();

            return instances;
        });

        self.loading_thread_handle.push(handle);

        // let chunk_range_u = camera_u - CHUNK_RENDER_DISTANCE..camera_u + CHUNK_RENDER_DISTANCE + 1;
        // let chunk_range_w = camera_w - CHUNK_RENDER_DISTANCE..camera_w + CHUNK_RENDER_DISTANCE + 1;

        // let handle = thread::spawn(|| {
        //     let mut instances: Vec<&RawCubeFaceInstance> = Vec::new();

        //     world.chunk_columns.get(&(0, 0));
        // });

        // for u in chunk_range_u.clone() {
        //     for w in chunk_range_w.clone() {
        //         if let None = world.chunk_columns.get(&(u, w)) {
        //             world.create_chunks(u, w);
        //         }
        //     }
        // }

        // let instances: Vec<&RawCubeFaceInstance> = Vec::new();
        // for u in chunk_range_u {
        //     for w in chunk_range_w.clone() {
        //         instances.extend(
        //             (0..VERTICAL_CHUNK_COUNT)
        //                 .flat_map(|v| world.meshed_chunks.get(&(u, v as u32, w)).unwrap())
        //                 .collect::<Vec<&RawCubeFaceInstance>>(),
        //         );
        //     }
        // }

        // let instances: Vec<RawCubeFaceInstance> = instances
        //     .iter()
        //     .map(|instance| *instance.to_owned())
        //     .collect();

        // if instances.len() > self.buffer_capacity {
        //     self.instance_buffer.destroy();
        //     self.instance_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
        //         label: Some("cube face instance buffer"),
        //         contents: bytemuck::cast_slice(instances.as_slice()),
        //         usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        //     });
        //     self.buffer_capacity = instances.len();
        // } else {
        //     self.queue.write_buffer(
        //         &self.instance_buffer,
        //         0,
        //         bytemuck::cast_slice(instances.as_slice()),
        //     );
        // }
    }

    pub fn render<'a: 'b, 'b>(&'a self, render_pass: &mut RenderPass<'b>) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

        render_pass.draw(
            0..CUBE_FACE_VERTICES.len() as u32,
            0..self.buffer_capacity as u32,
        );
    }
}
