use std::{collections::HashMap, iter, time::Instant};

use glam::vec3;
use wgpu::{
    Color, CommandEncoderDescriptor, Device, Extent3d, LoadOp, Operations, Queue, RenderPass,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor, StoreOp,
    Surface, SurfaceError, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor,
};
use winit::dpi::PhysicalSize;

use crate::{
    renderer::{
        indirect_buffer_manager::MultiDrawIndirectBuffer,
        pipelines::{
            block_outlines::BlockOutlinePipeline, frustum_culling::FrustumCullingComputePass,
            terrain::TerrainPipeline, ui::UiPipeline, GlobalsBinding,
        },
    },
    texture,
    window::InputState,
    world::{
        camera::{block_ray_caster, player::Player, CameraController},
        chunk::VERTICAL_CHUNK_COUNT,
        world_loader::{ChunkUniform, TerrainBuckets, WorldLoader},
        World,
    },
};

pub mod buffers;
pub mod indirect_buffer_manager;
mod pipelines;

pub mod vertex_buffer;

const CHUNK_RENDER_DISTANCE: u32 = 1;

pub struct Renderer {
    device: Device,
    queue: Queue,
    depth_texture: Texture,
    depth_texture_view: TextureView,
    world_renderer: WorldRenderer,
}

impl Renderer {
    pub fn new(
        device: Device,
        queue: Queue,
        surface_size: PhysicalSize<u32>,
        surface_format: TextureFormat,
    ) -> Self {
        let (depth_texture, depth_texture_view) =
            Renderer::create_depth_texture(&device, surface_size.width, surface_size.height);

        let world_renderer = WorldRenderer::new(
            device.clone(),
            queue.clone(),
            surface_size,
            surface_format,
            World::new(0),
        );

        Self {
            device,
            queue,
            depth_texture,
            depth_texture_view,
            world_renderer,
        }
    }

    fn create_depth_texture(device: &Device, width: u32, height: u32) -> (Texture, TextureView) {
        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("depth texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_texture_view = depth_texture.create_view(&TextureViewDescriptor::default());

        (depth_texture, depth_texture_view)
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        let (depth_texture, depth_texture_view) =
            Renderer::create_depth_texture(&self.device, new_size.width, new_size.height);
        self.depth_texture = depth_texture;
        self.depth_texture_view = depth_texture_view;

        self.world_renderer.update_aspect_ratio(new_size);
    }

    pub fn update(&mut self, state: &InputState) {
        self.world_renderer.update(state);
    }

    pub fn render(
        &self,
        surface: &Surface<'_>,
        surface_format: TextureFormat,
    ) -> Result<(), SurfaceError> {
        let surface_texture = surface.get_current_texture()?;
        let view = surface_texture.texture.create_view(&TextureViewDescriptor {
            format: Some(surface_format.add_srgb_suffix()),
            ..Default::default()
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color {
                            // TODO don't use hardcoded clear color
                            r: 135.0 / 255.0,
                            g: 206.0 / 255.0,
                            b: 235.0 / 255.0,
                            a: 1.0,
                        }),
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            self.world_renderer.render(&mut render_pass);
        }

        self.queue.submit(iter::once(encoder.finish()));
        surface_texture.present();

        Ok(())
    }
}

pub struct WorldRenderer {
    device: Device,
    queue: Queue,
    player: Player,
    globals: GlobalsBinding,
    ui_pipeline: UiPipeline,
    terrain_pipeline: TerrainPipeline,
    world_loader: WorldLoader,
    indirect_draw_buffer: MultiDrawIndirectBuffer<ChunkUniform, TerrainBuckets, 2>,
    frustum_culling_pass: FrustumCullingComputePass,
    block_outline_pipeline: BlockOutlinePipeline,
    last_update: Instant,
}

impl WorldRenderer {
    pub fn new(
        device: Device,
        queue: Queue,
        surface_size: PhysicalSize<u32>,
        surface_format: TextureFormat,
        world: World,
    ) -> Self {
        let camera_controller = CameraController::new(
            vec3(177.0, 33.61, 142.1),
            glam::Vec3::Z,
            glam::Vec3::Y,
            f32::to_radians(90.0),
            surface_size.width as f32 / surface_size.height as f32,
            0.1,
            1000.0,
        );

        let globals = GlobalsBinding::new(&device, &camera_controller);

        let mut world_loader = WorldLoader::new(world, 8, device.clone(), CHUNK_RENDER_DISTANCE);

        // TODO find better values
        let mut batches_map = HashMap::new();
        batches_map.insert(TerrainBuckets::SOLID, 3000);
        batches_map.insert(TerrainBuckets::TRANSPARENT, 1000);

        let chunks_per_bucket = (2 * CHUNK_RENDER_DISTANCE as u64 + 1).pow(2)
            * u64::min(
                CHUNK_RENDER_DISTANCE as u64 * 2 + 1,
                VERTICAL_CHUNK_COUNT as u64,
            );
        let mut ib = MultiDrawIndirectBuffer::new(
            &device,
            "",
            [TerrainBuckets::SOLID, TerrainBuckets::TRANSPARENT],
            chunks_per_bucket,
            &batches_map,
        );

        world_loader.load_chunks(&device, &queue, &mut ib, &camera_controller);

        let terrain_pipeline = TerrainPipeline::new(
            &device,
            &globals,
            &vertex_buffer::create_vertex_buffer(&device),
            &ib.uniform_buffer,
            texture::load_textures(&device, &queue).unwrap(),
            &texture::create_sampler(&device),
            surface_format,
        );

        let ui_pipeline = UiPipeline::new(&device, &globals, surface_format);

        let frustum_culling_pass = FrustumCullingComputePass::new(
            &device,
            &ib.uniform_buffer,
            &ib.indirect_buffer,
            chunks_per_bucket as u32,
            2 * chunks_per_bucket as u32,
        );

        let block_outline_pipeline = BlockOutlinePipeline::new(&device, &globals, surface_format);

        WorldRenderer {
            device,
            queue,
            player: Player::new(camera_controller, 10.0, 0.002),
            globals,
            ui_pipeline,
            terrain_pipeline,
            world_loader,
            indirect_draw_buffer: ib,
            frustum_culling_pass,
            block_outline_pipeline,
            last_update: Instant::now(),
        }
    }

    fn update_aspect_ratio(&mut self, new_size: PhysicalSize<u32>) {
        self.player
            .camera
            .set_aspect_ratio(new_size.width as f32 / new_size.height as f32);
    }

    pub fn update(&mut self, input_state: &InputState) {
        let now = Instant::now();
        self.player.handle_input(
            &input_state.pressed_keys,
            input_state.mouse_movement,
            now.duration_since(self.last_update).as_secs_f32(),
            |coordinates| {
                self.world_loader
                    .world
                    .get_block(coordinates)
                    .is_some_and(|block| block.is_solid())
            },
        );

        self.last_update = now;

        self.globals.update(&self.queue, &self.player.camera);

        self.world_loader.load_chunks(
            &self.device,
            &self.queue,
            &mut self.indirect_draw_buffer,
            &self.player.camera,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        self.frustum_culling_pass
            .run(&self.queue, &mut encoder, &self.player.camera);

        let focused_block =
            block_ray_caster::find_looked_at_blocks(&self.player.camera, &self.world_loader.world);

        self.block_outline_pipeline.set_outlined_block(
            &self.queue,
            focused_block.solid_block.map(|block| block.coords),
        );

        self.queue.submit(iter::once(encoder.finish()));
    }

    pub fn render<'a: 'b, 'b>(&'a self, render_pass: &mut RenderPass<'b>) {
        if self.indirect_draw_buffer.draw_count(TerrainBuckets::SOLID) > 0 {
            self.terrain_pipeline.render_terrain(
                render_pass,
                &self.globals,
                &self.indirect_draw_buffer.vertex_buffer,
                &self.indirect_draw_buffer.indirect_buffer,
                self.indirect_draw_buffer
                    .indirect_buffer_offset_bytes(TerrainBuckets::SOLID, 0),
                self.indirect_draw_buffer.draw_count(TerrainBuckets::SOLID) as u32,
            );
        }

        if self
            .indirect_draw_buffer
            .draw_count(TerrainBuckets::TRANSPARENT)
            > 0
        {
            self.terrain_pipeline.render_water(
                render_pass,
                &self.globals,
                &self.indirect_draw_buffer.vertex_buffer,
                &self.indirect_draw_buffer.indirect_buffer,
                self.indirect_draw_buffer
                    .indirect_buffer_offset_bytes(TerrainBuckets::TRANSPARENT, 0),
                self.indirect_draw_buffer
                    .draw_count(TerrainBuckets::TRANSPARENT) as u32,
            );
        }

        self.block_outline_pipeline
            .render(render_pass, &self.globals);
        self.ui_pipeline.render(render_pass, &self.globals);
    }
}
