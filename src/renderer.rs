use std::time::Duration;
use web_time::Instant;

use glam::{IVec3, Vec3, vec3};
use smallvec::SmallVec;
use wgpu::{
    Color, CommandEncoder, CommandEncoderDescriptor, Device, Extent3d, LoadOp, Operations, Queue,
    RenderPass, RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    StoreOp, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};
use winit::{dpi::PhysicalSize, event::MouseButton, keyboard::KeyCode};

use crate::{
    camera::{
        Perspective, View,
        block_ray_caster::BlockHitInfo,
        player::{self, PlayerState},
    },
    input::InputState,
    renderer::{
        indirect_buffer_manager::IndirectBufferManager,
        pipelines::{
            GlobalsBinding, block_outlines::BlockOutlinePipeline,
            debug_crosshair::CrosshairPipeline, frustum_culling::FrustumCullingComputePass,
            terrain::TerrainPipeline,
        },
    },
    texture,
    ui::{AddToGui, GuiModule},
    world::{
        World,
        blocks::{Block, BlockPhysicsType},
        chunk::VERTICAL_CHUNK_COUNT,
        world_loader::{TerrainType, WorldLoader},
    },
};

pub mod buffers;
pub mod indirect_buffer_manager;
mod pipelines;

pub mod vertex_buffer;

const CHUNK_RENDER_DISTANCE: u32 = 4;

pub struct SceneState {
    device: Device,
    queue: Queue,
    depth_texture: Texture,
    depth_texture_view: TextureView,
    world_renderer: WorldRenderer,
    world_loader: WorldLoader,
    player: PlayerState,
    update_loop: FixedTimestepLoop,
    indirect_buffer_manager: IndirectBufferManager<TerrainType>,
}

impl SceneState {
    pub fn new(
        device: Device,
        queue: Queue,
        surface_size: PhysicalSize<u32>,
        surface_format: TextureFormat,
        texture_array: TextureView,
    ) -> Self {
        let player = PlayerState::new(
            Perspective {
                fov_y_rad: f32::to_radians(90.0),
                aspect_ratio: surface_size.width as f32 / surface_size.height as f32,
                z_near: 0.1,
                z_far: 1000.0,
            },
            vec3(177.0, 128., 142.1),
            Vec3::Z,
        );

        let world = World::new();
        let world_loader = WorldLoader::new(world, player.eye(), CHUNK_RENDER_DISTANCE);

        let chunks_per_bucket = (2 * CHUNK_RENDER_DISTANCE as u64 + 1).pow(2)
            * u64::min(
                CHUNK_RENDER_DISTANCE as u64 * 2 + 1,
                VERTICAL_CHUNK_COUNT as u64,
            );
        let indirect_buffer_manager =
            IndirectBufferManager::new(&device, "terrain".into(), chunks_per_bucket);

        let (depth_texture, depth_texture_view) =
            SceneState::create_depth_texture(&device, surface_size.width, surface_size.height);

        let world_renderer = WorldRenderer::new(
            device.clone(),
            queue.clone(),
            surface_format,
            texture_array,
            &indirect_buffer_manager,
            &player,
        );

        // todo reorder struct fields
        Self {
            device,
            queue,
            depth_texture,
            depth_texture_view,
            world_renderer,
            player,
            world_loader,
            update_loop: FixedTimestepLoop::new(Duration::from_secs_f32(player::TPS.recip())),
            indirect_buffer_manager,
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
            SceneState::create_depth_texture(&self.device, new_size.width, new_size.height);
        self.depth_texture = depth_texture;
        self.depth_texture_view = depth_texture_view;

        self.player
            .set_aspect_ratio(new_size.width as f32 / new_size.height as f32);
    }

    pub fn update(&mut self, input_state: &mut InputState) {
        self.player.update_rotation(input_state);
        let lag_s = self
            .update_loop
            .tick(|TickInformation { timestep_s, time_s }| {
                self.player.update_position(
                    input_state,
                    timestep_s,
                    time_s,
                    self.world_loader.world(),
                );
            });
        self.player
            .update_looked_at_blocks(self.world_loader.world());

        let mut updated_blocks = SmallVec::new();
        if let Some(BlockHitInfo {
            coords: looked_at_block_coords,
            face: Some(direction),
            ..
        }) = self.player.looked_at_blocks().solid_block
        {
            let left_mouse_pressed = input_state.pull_is_pressed(MouseButton::Left);
            let right_mouse_pressed = input_state.pull_is_pressed(MouseButton::Right);

            if left_mouse_pressed || right_mouse_pressed {
                let (pos, block) = if left_mouse_pressed {
                    // if both pressed, mining blocks has a higher priority
                    (looked_at_block_coords, Block::Air)
                } else {
                    // right mouse pressed
                    (
                        looked_at_block_coords + direction.get_unit_ivec(),
                        Block::LeavesOak,
                    )
                };

                if !self.player.intersects_block(pos)
                    || block.physics_type() != BlockPhysicsType::Solid
                {
                    updated_blocks.push((pos, block));
                }
            }
        }

        let extrapolated_view = self
            .player
            .extrapolate_view(lag_s, self.world_loader.world());

        self.world_renderer.update(
            extrapolated_view,
            self.player.perspective(),
            self.player
                .looked_at_blocks()
                .solid_block
                .map(|block| block.coords),
        );

        if input_state.pull_is_pressed(KeyCode::KeyR) {
            self.world_loader
                .reload_world(&mut self.indirect_buffer_manager);
        } else {
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("update command encoder"),
                });
            self.world_loader.load_chunks(
                &self.device,
                &self.queue,
                &mut encoder,
                &mut self.indirect_buffer_manager,
                self.player.eye(),
                updated_blocks,
            );
            self.queue.submit([encoder.finish()]);
        }
    }

    pub fn render(&self, encoder: &mut CommandEncoder, surface_view: &TextureView) {
        self.world_renderer.render(
            encoder,
            surface_view,
            &self.depth_texture_view,
            &self.indirect_buffer_manager,
        );
    }
}

impl AddToGui for BlockHitInfo {
    fn add_to_ui(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.monospace(format!("{:?}", self.block));
            ui.label("at");
            ui.monospace(format!("{:?}", self.coords));
        });
    }
}

impl GuiModule for SceneState {
    fn title(&self) -> &str {
        "Player state"
    }

    fn add_contents(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let Vec3 { x, y, z } = self.player.eye();
            ui.label("eye:");
            ui.monospace(format!("{x:.2} / {y:.2} / {z:.2}"));
        });
        ui.horizontal(|ui| {
            let Vec3 { x, y, z } = self.player.direction();
            ui.label("direction:");
            ui.monospace(format!("{x:+.2} / {y:+.2} / {z:+.2}"));
            ui.label("facing");
            ui.monospace(format!("{:?}", self.player.cardinal_direction()));
        });
        ui.horizontal(|ui| {
            ui.label("focused block:");
            match self.player.looked_at_blocks().solid_block {
                Some(info) => {
                    info.add_to_ui(ui);
                }
                None => {
                    ui.monospace("None");
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("focused liquid:");
            match self.player.looked_at_blocks().liquid_block {
                Some(info) => {
                    info.add_to_ui(ui);
                }
                None => {
                    ui.monospace("None");
                }
            }
        });
    }
}

pub struct WorldRenderer {
    queue: Queue,
    globals: GlobalsBinding,
    crosshair_pipeline: CrosshairPipeline,
    terrain_pipeline: TerrainPipeline,
    frustum_culling_pass: FrustumCullingComputePass,
    block_outline_pipeline: BlockOutlinePipeline,
}

impl WorldRenderer {
    pub fn new(
        device: Device,
        queue: Queue,
        surface_format: TextureFormat,
        texture_array: TextureView,
        indirect_buffer_manager: &IndirectBufferManager<TerrainType>,
        player: &PlayerState,
    ) -> Self {
        let globals = GlobalsBinding::new(&device);

        let terrain_pipeline = TerrainPipeline::new(
            &device,
            &globals,
            &vertex_buffer::create_vertex_buffer(&device),
            indirect_buffer_manager.uniform_buffer(),
            texture_array,
            &texture::create_sampler(&device),
            surface_format,
        );

        let crosshair_pipeline = CrosshairPipeline::new(&device, &globals, surface_format);

        let frustum_culling_pass = FrustumCullingComputePass::new(
            &device,
            indirect_buffer_manager.uniform_buffer(),
            indirect_buffer_manager.indirect_buffer(),
            indirect_buffer_manager.chunks_per_bucket() as u32,
            2 * indirect_buffer_manager.chunks_per_bucket() as u32,
            player.view(),
            player.perspective(),
        );

        let block_outline_pipeline = BlockOutlinePipeline::new(&device, &globals, surface_format);

        WorldRenderer {
            queue,
            globals,
            crosshair_pipeline,
            terrain_pipeline,
            frustum_culling_pass,
            block_outline_pipeline,
        }
    }

    pub fn update(
        &mut self,
        extrapolated_view: View,
        perspective: Perspective,
        focused_block: Option<IVec3>,
    ) {
        self.globals
            .update(&self.queue, extrapolated_view, perspective);
        self.frustum_culling_pass
            .update_camera(&self.queue, extrapolated_view, perspective);

        self.block_outline_pipeline
            .set_outlined_block(&self.queue, focused_block);
    }

    pub fn render(
        &self,
        encoder: &mut CommandEncoder,
        surface_view: &TextureView,
        depth_texture_view: &TextureView,
        indirect_buffer_manager: &IndirectBufferManager<TerrainType>,
    ) {
        self.frustum_culling_pass.run(encoder);

        let mut render_pass: RenderPass<'_> = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("scene render pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: surface_view,
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
                view: depth_texture_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if indirect_buffer_manager.draw_count(TerrainType::Solid) > 0 {
            self.terrain_pipeline.render_terrain(
                &mut render_pass,
                &self.globals,
                indirect_buffer_manager.vertex_buffer(),
                indirect_buffer_manager.indirect_buffer(),
                indirect_buffer_manager.indirect_buffer_offset(TerrainType::Solid),
                indirect_buffer_manager.draw_count(TerrainType::Solid) as u32,
            );
        }

        if indirect_buffer_manager.draw_count(TerrainType::Transparent) > 0 {
            self.terrain_pipeline.render_water(
                &mut render_pass,
                &self.globals,
                indirect_buffer_manager.vertex_buffer(),
                indirect_buffer_manager.indirect_buffer(),
                indirect_buffer_manager.indirect_buffer_offset(TerrainType::Transparent),
                indirect_buffer_manager.draw_count(TerrainType::Transparent) as u32,
            );
        }

        self.block_outline_pipeline
            .render(&mut render_pass, &self.globals);
        self.crosshair_pipeline
            .render(&mut render_pass, &self.globals);
    }
}

struct FixedTimestepLoop {
    timestep_s: f32,
    accumulator_s: f32,
    last_tick: Instant,
    start_time: Instant,
}

struct TickInformation {
    timestep_s: f32,
    time_s: f32,
}

impl FixedTimestepLoop {
    fn new(timestep: Duration) -> Self {
        FixedTimestepLoop {
            timestep_s: timestep.as_secs_f32(),
            accumulator_s: 0.0,
            last_tick: Instant::now(),
            start_time: Instant::now(),
        }
    }

    fn tick(&mut self, mut tick: impl FnMut(TickInformation)) -> f32 {
        let current_time = Instant::now();
        self.accumulator_s += current_time.duration_since(self.last_tick).as_secs_f32();
        self.last_tick = current_time;

        while self.accumulator_s >= self.timestep_s {
            tick(TickInformation {
                timestep_s: self.timestep_s,
                time_s: self.start_time.elapsed().as_secs_f32(),
            });
            self.accumulator_s -= self.timestep_s;
        }

        self.accumulator_s
    }
}
