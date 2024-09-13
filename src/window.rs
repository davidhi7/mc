use std::{
    collections::HashSet,
    iter,
    rc::Rc,
    sync::{Arc, Mutex},
};

use wgpu::{Color, CompositeAlphaMode, PresentMode, TextureFormat};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window},
};

use crate::world::{renderer::WorldRenderer, World};

#[derive(Default)]
pub struct App {
    window: Option<Arc<Window>>,
    gfx_state: Option<GfxState>,
    pressed_keys: HashSet<KeyCode>,
    mouse_movement: (f64, f64),
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("wgpu test"))
                .unwrap(),
        );
        window.set_cursor_visible(false);

        self.gfx_state = Some(pollster::block_on(GfxState::new(Arc::clone(&window))));
        self.window = Some(window);
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.mouse_movement = delta;
            }
            _ => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close button pressed, terminating");
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                log::info!("Resized to new size: {physical_size:?}");
                self.gfx_state.as_mut().unwrap().resize(physical_size);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(keycode),
                        repeat: false,
                        state,
                        ..
                    },
                ..
            } => match state {
                ElementState::Pressed => {
                    self.pressed_keys.insert(keycode);
                }
                ElementState::Released => {
                    self.pressed_keys.remove(&keycode);
                }
            },
            WindowEvent::CursorEntered { .. } => {
                let window = self.window.as_ref().unwrap();
                window
                    .set_cursor_grab(CursorGrabMode::Confined)
                    .or_else(|_e| window.set_cursor_grab(CursorGrabMode::Locked))
                    .unwrap();
            }
            WindowEvent::RedrawRequested => {
                let gfx_state = self.gfx_state.as_mut().unwrap();

                gfx_state.update(&self.pressed_keys, self.mouse_movement);
                // Don't handle the same mouse input twice
                self.mouse_movement = (0.0, 0.0);
                match gfx_state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        gfx_state.resize(self.window.as_ref().unwrap().inner_size())
                    }
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Out of memory");
                        event_loop.exit();
                    }

                    // This happens when the a frame takes too long to present
                    Err(wgpu::SurfaceError::Timeout) => {
                        log::warn!("Surface timeout")
                    }
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

struct GfxState {
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_config: wgpu::SurfaceConfiguration,
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,
    clear_color: wgpu::Color,
    world_renderer: WorldRenderer,
    world: Arc<Mutex<World>>,
}

impl GfxState {
    async fn new(window: Arc<Window>) -> GfxState {
        let size: winit::dpi::PhysicalSize<u32> = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_limits: wgpu::Limits::default(),
                    required_features: wgpu::Features::TEXTURE_BINDING_ARRAY | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING | wgpu::Features::POLYGON_MODE_LINE,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Use sRGB surface
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Auto,
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };

        let (depth_texture, depth_texture_view) =
            GfxState::create_depth_texture(&device, surface_config.width, surface_config.height);

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let world = Arc::new(Mutex::new(World::new(0)));

        let mut world_renderer =
            WorldRenderer::new(Arc::clone(&device), Arc::clone(&queue), &surface_config);
        world_renderer.update(Arc::clone(&world));

        let instance = Self {
            surface,
            device,
            queue,
            surface_config,
            depth_texture,
            depth_texture_view,
            clear_color: Color {
                r: 135.0 / 255.0,
                g: 206.0 / 255.0,
                b: 235.0 / 255.0,
                a: 0.0,
            },
            world_renderer,
            world,
        };

        instance
    }

    pub fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: width,
            height: height,
            depth_or_array_layers: 1,
        };
        let depth_texture_descriptor = wgpu::TextureDescriptor {
            label: Some("depth texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let depth_texture = device.create_texture(&depth_texture_descriptor);
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        (depth_texture, depth_texture_view)
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);

            self.world_renderer
                .camera_controller
                .set_aspect_ratio(new_size.width as f32 / new_size.height as f32);

            let (depth_texture, depth_texture_view) = GfxState::create_depth_texture(
                &self.device,
                self.surface_config.width,
                self.surface_config.height,
            );
            self.depth_texture = depth_texture;
            self.depth_texture_view = depth_texture_view;
        }
    }

    fn update(&mut self, pressed_keys: &HashSet<KeyCode>, mouse_movement: (f64, f64)) {
        self.world_renderer
            .camera_controller
            .handle_input(pressed_keys, mouse_movement);

        self.world_renderer.update(Arc::clone(&self.world));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render rass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            self.world_renderer.render(&mut render_pass);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    let event_loop: EventLoop<()> = EventLoop::new().unwrap();
    event_loop.run_app(&mut App::default()).unwrap();
}
