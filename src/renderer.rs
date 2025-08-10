use std::{collections::HashMap, iter, sync::Arc};

use glam::vec3;
use wgpu::{CommandEncoderDescriptor, Device, Queue, RenderPass, SurfaceConfiguration};

use crate::{
    renderer::{
        indirect_buffer_manager::MultiDrawIndirectBuffer,
        pipelines::{
            frustum_culling::FrustumCullingComputePass, terrain::TerrainPipeline, ui::UiPipeline,
            GlobalsBinding,
        },
    },
    texture,
    world::{
        camera::CameraController,
        chunk::VERTICAL_CHUNK_COUNT,
        world_loader::{ChunkUniform, TerrainBuckets, WorldLoader},
        World,
    },
};

pub mod buffers;
pub mod indirect_buffer_manager;
mod pipelines;

pub mod vertex_buffer;

const CHUNK_RENDER_DISTANCE: u32 = 8;

pub struct WorldRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pub camera_controller: CameraController,
    globals: GlobalsBinding,
    ui_pipeline: UiPipeline,
    terrain_pipeline: TerrainPipeline,
    pub world_loader: WorldLoader,
    indirect_draw_buffer: MultiDrawIndirectBuffer<ChunkUniform, TerrainBuckets, 2>,
    frustum_culling_pass: FrustumCullingComputePass,
}

impl WorldRenderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface_config: &SurfaceConfiguration,
        world: World,
    ) -> Self {
        let camera_controller = CameraController::new(
            vec3(177.0, 33.61, 142.1),
            glam::Vec3::Z,
            glam::Vec3::Y,
            f32::to_radians(90.0),
            surface_config.width as f32 / surface_config.height as f32,
            0.1,
            1000.0,
            10.0,
            0.002,
        );

        let globals = GlobalsBinding::new(&device, &camera_controller);

        let mut world_loader =
            WorldLoader::new(world, 8, Arc::clone(&device), CHUNK_RENDER_DISTANCE);

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
            surface_config.format,
        );

        let ui_pipeline = UiPipeline::new(&device, &globals, surface_config.format);

        let frustum_culling_pass = FrustumCullingComputePass::new(
            &device,
            &ib.uniform_buffer,
            &ib.indirect_buffer,
            chunks_per_bucket as u32,
            2 * chunks_per_bucket as u32,
        );

        WorldRenderer {
            device,
            queue,
            camera_controller,
            globals,
            ui_pipeline,
            terrain_pipeline,
            world_loader,
            indirect_draw_buffer: ib,
            frustum_culling_pass,
        }
    }

    pub fn update(&mut self) {
        self.globals.update(&self.queue, &self.camera_controller);

        self.world_loader.load_chunks(
            &self.device,
            &self.queue,
            &mut self.indirect_draw_buffer,
            &self.camera_controller,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        self.frustum_culling_pass
            .run(&self.queue, &mut encoder, &self.camera_controller);

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

        self.ui_pipeline.render(render_pass, &self.globals);
    }
}
