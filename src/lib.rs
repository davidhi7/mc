#![feature(try_blocks)]

mod camera;
mod frametime_metrics;
pub mod input;
mod logging;
mod math;
mod renderer;
pub(crate) mod shaders;
#[cfg(test)]
pub(crate) mod tests;
mod texture;
pub mod thread_pool;
pub mod ui;
#[cfg(target_arch = "wasm32")]
mod wasm_fetch;
mod world;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use std::{iter, sync::Arc};
use web_time::Instant;

use wgpu::{
    CompositeAlphaMode, Device, DeviceDescriptor, ExperimentalFeatures, Features, Instance,
    InstanceDescriptor, Limits, MemoryHints, PowerPreference, PresentMode, Queue,
    RequestAdapterOptions, Surface, SurfaceConfiguration, SurfaceError, TextureFormat,
    TextureUsages, TextureViewDescriptor, Trace, wgt::CommandEncoderDescriptor,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use crate::{
    frametime_metrics::FrameTimeMetrics, input::InputState, renderer::Renderer, ui::EguiState,
};

struct Graphics {
    window: Arc<Window>,
    device: Device,
    queue: Queue,
    surface: Surface<'static>,
    // Format for surface cannot be sRGB in WebGPU
    surface_format: TextureFormat,
    // So add sRGB when creating texture views
    surface_view_format: TextureFormat,
    input_state: InputState,
    frametimes: FrameTimeMetrics,
    renderer: Renderer,
    egui_state: EguiState,
    surface_size: PhysicalSize<u32>,
}

impl Graphics {
    async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let instance = Instance::new(&InstanceDescriptor::default());
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let adapter_info = adapter.get_info();
        log::info!("Backend: {}", adapter_info.backend);
        log::info!(
            "Device:\n{}\n{} {}",
            adapter_info.name,
            adapter_info.driver,
            adapter_info.driver_info
        );

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                required_limits: Limits {
                    #[cfg(not(target_arch = "wasm32"))]
                    max_buffer_size: (1 << 31) - 1,
                    #[cfg(target_arch = "wasm32")]
                    // Max allowed value on firefox?
                    max_buffer_size: (1 << 30) - 1,
                    // relevant for texture binding array: must be greater than or equal the count of textures
                    max_binding_array_elements_per_shader_stage: 127,
                    ..Default::default()
                },
                required_features: Features::INDIRECT_FIRST_INSTANCE,
                memory_hints: MemoryHints::Performance,
                trace: Trace::Off,
                experimental_features: ExperimentalFeatures::disabled(),
            })
            .await?;

        let surface = instance.create_surface(Arc::clone(&window))?;
        let caps = surface.get_capabilities(&adapter);

        // Find linear RGB format and set view format to sRGB later
        // https://gpuweb.github.io/gpuweb/#dom-gpucanvasconfiguration-viewformats
        let surface_format = caps
            .formats
            .iter()
            .find(|format| !format.is_srgb() && format.has_color_aspect())
            .expect("Should find at least one linear RGB surface format")
            .to_owned();
        let surface_view_format = surface_format.add_srgb_suffix();

        log::debug!("Available surface formats: {:?}", caps.formats);
        log::debug!("Used surface format: {:?}", surface_format);
        log::debug!("Used surface view format: {:?}", surface_view_format);

        let size = window.inner_size();

        let renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            size,
            surface_view_format,
            texture::load_textures(&device, &queue).await.unwrap(),
        );

        let frametimes = FrameTimeMetrics::new(1000);

        let mut egui_state = EguiState::new(&window, &device, surface_view_format);
        egui_state.add_gui_module(frametime_metrics::create_ui_module(&frametimes));
        let surface_size = window.inner_size();

        let state = Graphics {
            window,
            device,
            queue,
            surface,
            surface_format,
            surface_view_format,
            input_state: Default::default(),
            frametimes,
            renderer,
            egui_state,
            surface_size,
        };

        state.configure_surface(size);

        Ok(state)
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.surface_size = new_size;
        self.configure_surface(new_size);
        self.renderer.resize(new_size);
    }

    fn configure_surface(&self, size: PhysicalSize<u32>) {
        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::AutoNoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: CompositeAlphaMode::Auto,
            view_formats: vec![self.surface_view_format],
        };

        self.surface.configure(&self.device, &surface_config);
    }

    fn render(&mut self, event_loop: &ActiveEventLoop) {
        let frametime_start = Instant::now();

        self.renderer.update(&mut self.input_state);

        match self.surface.get_current_texture() {
            Ok(surface_texture) => {
                let mut encoder = self
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor {
                        label: Some("render command encoder"),
                    });

                let surface_view = surface_texture.texture.create_view(&TextureViewDescriptor {
                    format: Some(self.surface_view_format),
                    ..Default::default()
                });

                self.renderer.render(&mut encoder, &surface_view);

                self.egui_state.render(
                    &self.window,
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &surface_view,
                    self.surface_size,
                );

                self.queue.submit(iter::once(encoder.finish()));

                surface_texture.present();
            }
            // Reconfigure the surface if it's lost or outdated
            Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                self.configure_surface(self.window.inner_size());
            }
            // The system is out of memory, we should probably quit
            Err(SurfaceError::OutOfMemory) => {
                log::error!("Out of memory");
                event_loop.exit();
            }

            // This happens when the a frame takes too long to present
            Err(SurfaceError::Timeout) => {
                log::warn!("Surface timeout");
            }

            Err(SurfaceError::Other) => {
                log::warn!("Unknown surface error");
                event_loop.exit();
            }
        }

        self.frametimes.push(frametime_start.elapsed());
        self.frametimes.maybe_update_sample();
    }
}

#[allow(clippy::large_enum_variant)]
enum AppState {
    Ready(Graphics),
    Init(Option<EventLoopProxy<Graphics>>),
}

struct App {
    state: AppState,
}

impl App {
    pub fn new(event_loop: &EventLoop<Graphics>) -> Self {
        Self {
            state: AppState::Init(Some(event_loop.create_proxy())),
        }
    }
}

impl ApplicationHandler<Graphics> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let AppState::Init(proxy) = &mut self.state {
            #[allow(unused_variables)]
            let proxy = proxy.take().unwrap();

            #[allow(unused_mut)]
            let mut window_attributes = Window::default_attributes();

            #[cfg(target_arch = "wasm32")]
            {
                use wasm_bindgen::JsCast;
                use winit::platform::web::WindowAttributesExtWebSys;

                const CANVAS_ID: &str = "canvas";

                let window = wgpu::web_sys::window().unwrap_throw();
                let document = window.document().unwrap_throw();
                let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
                let html_canvas_element = canvas.unchecked_into();
                window_attributes = window_attributes
                    .with_canvas(Some(html_canvas_element))
                    // Browser shortcuts should still work
                    .with_prevent_default(false);
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                window_attributes = window_attributes.with_title("mc");
            }

            // window.set_cursor_visible(false);

            let window = Arc::new(
                event_loop
                    .create_window(window_attributes)
                    .expect("Failed to create window"),
            );

            #[cfg(not(target_arch = "wasm32"))]
            {
                let gfx: Graphics = pollster::block_on(Graphics::new(window)).unwrap();
                self.state = AppState::Ready(gfx);
                log::info!("App ready");
            }

            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_futures::spawn_local(async move {
                let gfx = Graphics::new(window)
                    .await
                    .expect("Failed to create graphics context");
                assert!(proxy.send_event(gfx).is_ok());
            });
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: Graphics) {
        // This is where proxy.send_event() ends up
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(event.window.inner_size());
        }
        self.state = AppState::Ready(event);
        log::info!("App ready");
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let AppState::Ready(gfx) = &mut self.state else {
            log::warn!("Device event but app is not ready");
            return;
        };

        if let DeviceEvent::MouseMotion { delta } = event
            && !gfx.egui_state.wants_pointer_input()
        {
            gfx.input_state.increment_mouse_movement(delta);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let AppState::Ready(gfx) = &mut self.state else {
            log::warn!("Window event but app is not ready");
            return;
        };

        if gfx.egui_state.on_window_event(&gfx.window, &event).consumed {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested, terminating");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                gfx.render(event_loop);
                // Emits a new redraw requested event.
                // needed in web?
                gfx.window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                log::info!("Resized to new size: {size:?}");
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always followed up by redraw request.
                gfx.resize(size);
            }
            WindowEvent::CursorEntered { .. } => {
                gfx.window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_e| gfx.window.set_cursor_grab(CursorGrabMode::Confined))
                    .unwrap();
            }
            WindowEvent::KeyboardInput { event, .. } => match event {
                KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                    ..
                } => {
                    if let Err(err) = gfx.window.set_cursor_grab(CursorGrabMode::None) {
                        log::warn!("Failed to release cursor: {err:?}");
                    };
                }
                _ => gfx.input_state.handle_key_event(event),
            },
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => gfx.input_state.handle_mouse_event(button, button_state),
            _ => (),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn launch(concurrency: usize) {
    logging::setup_logger();
    thread_pool::init_thread_pool(concurrency).expect("Thread pool initialization failed");

    let event_loop = EventLoop::with_user_event().build().unwrap();
    event_loop.set_control_flow({
        if cfg!(target_arch = "wasm32") {
            // In Firefox and Safari, ControlFlow::Poll appears to cause significant slowdown
            ControlFlow::Wait
        } else {
            ControlFlow::Poll
        }
    });
    let app = App::new(&event_loop);

    run(event_loop, app);
}

#[cfg(not(target_arch = "wasm32"))]
fn run(event_loop: EventLoop<Graphics>, mut app: App) {
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(target_arch = "wasm32")]
fn run(event_loop: EventLoop<Graphics>, app: App) {
    use winit::platform::web::EventLoopExtWebSys;

    wasm_bindgen_futures::spawn_local(async move {
        event_loop.spawn_app(app);
    });
}
