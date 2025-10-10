mod frametime_metrics;
pub mod input;

use std::{sync::Arc, time::Instant};

use wgpu::{
    CompositeAlphaMode, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    MemoryHints, PowerPreference, PresentMode, RequestAdapterOptions, Surface,
    SurfaceConfiguration, SurfaceError, TextureFormat, TextureUsages, Trace,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{CursorGrabMode, Window, WindowId},
};

use crate::{
    renderer::Renderer,
    window::{frametime_metrics::FrameTimeMetrics, input::InputState},
};

struct AppState {
    window: Arc<Window>,
    device: Device,
    surface: Surface<'static>,
    surface_format: TextureFormat,
    input_state: InputState,
    frametimes: FrameTimeMetrics,
    renderer: Renderer,
}

impl AppState {
    async fn new(window: Arc<Window>) -> Self {
        let instance = Instance::new(&InstanceDescriptor::default());
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let adapter_info = adapter.get_info();
        println!("Backend: {}", adapter_info.backend);
        println!(
            "Device:\n{}\n{} {}",
            adapter_info.name, adapter_info.driver, adapter_info.driver_info
        );

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                required_limits: Limits {
                    max_buffer_size: u32::MAX as u64 >> 1,
                    // relevant for texture binding array: must be greater than or equal the count of textures
                    max_binding_array_elements_per_shader_stage: 127,
                    ..Default::default()
                },
                required_features: Features::TEXTURE_BINDING_ARRAY
                    | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                    | Features::POLYGON_MODE_LINE
                    | Features::MULTI_DRAW_INDIRECT
                    | Features::INDIRECT_FIRST_INSTANCE,
                memory_hints: MemoryHints::Performance,
                trace: Trace::Off,
            })
            .await
            .unwrap();

        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let cap = surface.get_capabilities(&adapter);
        let surface_format = cap.formats[0];

        let size = window.inner_size();

        let renderer = Renderer::new(device.clone(), queue.clone(), size, surface_format);

        let state = AppState {
            window,
            device,
            surface,
            surface_format,
            input_state: Default::default(),
            frametimes: FrameTimeMetrics::new(1000),
            renderer,
        };

        state.configure_surface(size);

        state
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
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
            view_formats: vec![self.surface_format.add_srgb_suffix()],
        };

        self.surface.configure(&self.device, &surface_config);
    }

    fn render(&mut self, event_loop: &ActiveEventLoop) {
        let frametime_start = Instant::now();

        self.renderer.update(&mut self.input_state);
        match self.renderer.render(&self.surface, self.surface_format) {
            Ok(_) => {}
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

        // Don't handle the same mouse input twice
        // self.input_state.mouse_movement = (0.0, 0.0);
        self.frametimes.push(frametime_start.elapsed());
        self.frametimes.update_sample();
        self.window.set_title(&format!(
            "mc | {}ms",
            self.frametimes.get_sample_frametime()
        ));
    }
}

#[derive(Default)]
struct App {
    state: Option<AppState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("mc"))
                .unwrap(),
        );
        window.set_cursor_visible(false);

        self.state = Some(pollster::block_on(AppState::new(window.clone())));

        window.request_redraw();
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.state
                .as_mut()
                .unwrap()
                .input_state
                .increment_mouse_movement(delta);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = self.state.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested, terminating");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                state.render(event_loop);
                // Emits a new redraw requested event.
                state.window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                log::info!("Resized to new size: {size:?}");
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always followed up by redraw request.
                state.resize(size);
            }
            WindowEvent::CursorEntered { .. } => {
                state
                    .window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_e| state.window.set_cursor_grab(CursorGrabMode::Confined))
                    .unwrap();
            }
            WindowEvent::KeyboardInput { event, .. } => state.input_state.handle_key_event(event),
            _ => (),
        }
    }
}

pub async fn run() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default()).unwrap();
}
