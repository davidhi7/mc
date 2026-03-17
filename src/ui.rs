use egui::{Context, Shadow, TextureId, Ui, ViewportId, Visuals};
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use egui_winit::State;
use wgpu::{
    CommandBuffer, CommandEncoder, Device, LoadOp, Operations, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, TextureFormat, TextureView,
};
use winit::{dpi::PhysicalSize, event::WindowEvent, window::Window};

pub trait AddToGui {
    fn add_to_ui(&self, ui: &mut Ui);
}

pub trait GuiModule {
    fn title(&self) -> &str;
    fn add_contents(&mut self, ui: &mut Ui);
}

pub struct EguiState {
    context: Context,
    winit_state: State,
    renderer: Renderer,
    device: Device,
    queue: Queue,
}

impl EguiState {
    pub fn new(
        window: &Window,
        device: &Device,
        queue: &Queue,
        surface_view_format: TextureFormat,
    ) -> Self {
        let context = Context::default();
        context.set_visuals(Visuals {
            window_shadow: Shadow::NONE,
            ..Default::default()
        });

        let winit_state = State::new(
            context.clone(),
            ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );

        let renderer = Renderer::new(device, surface_view_format, RendererOptions::default());

        Self {
            context,
            winit_state,
            renderer,
            device: device.clone(),
            queue: queue.clone(),
        }
    }

    pub fn on_window_event(
        &mut self,
        window: &Window,
        event: &WindowEvent,
    ) -> egui_winit::EventResponse {
        self.winit_state.on_window_event(window, event)
    }

    pub fn wants_pointer_input(&self) -> bool {
        self.context.wants_pointer_input()
    }

    pub fn register_native_texture(&mut self, texture_view: &TextureView) -> TextureId {
        self.renderer
            .register_native_texture(&self.device, texture_view, wgpu::FilterMode::Nearest)
    }

    pub fn render(
        &mut self,
        window: &Window,
        encoder: &mut CommandEncoder,
        target_view: &TextureView,
        surface_size: PhysicalSize<u32>,
        mut modules: Vec<&mut dyn GuiModule>,
    ) -> Vec<CommandBuffer> {
        let raw_input = self.winit_state.take_egui_input(window);

        let full_output = self.context.run(raw_input, |ctx| {
            egui::Window::new("Title").title_bar(false).show(ctx, |ui| {
                for module in modules.iter_mut() {
                    ui.heading(module.title());
                    module.add_contents(ui);
                }
            });
        });

        self.winit_state
            .handle_platform_output(window, full_output.platform_output);

        let pixels_per_point = egui_winit::pixels_per_point(&self.context, window);
        let paint_jobs = self
            .context
            .tessellate(full_output.shapes, pixels_per_point);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [surface_size.width, surface_size.height],
            pixels_per_point,
        };

        let user_cmd_buffers = self.renderer.update_buffers(
            &self.device,
            &self.queue,
            encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        {
            let render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("egui render pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            self.renderer.render(
                &mut render_pass.forget_lifetime(),
                &paint_jobs,
                &screen_descriptor,
            );
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }

        user_cmd_buffers
    }
}
