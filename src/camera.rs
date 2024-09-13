use std::{collections::HashSet, f32::consts::PI};

use glam::Vec3;
use winit::keyboard::KeyCode;

struct Perspective {
    fov_y: f32,
    aspect_ratio: f32,
    z_near: f32,
    z_far: f32,
}

impl Perspective {
    fn get_matrix(&self) -> glam::Mat4 {
        glam::Mat4::perspective_lh(self.fov_y, self.aspect_ratio, self.z_near, self.z_far)
    }
}

struct View {
    pub eye: glam::Vec3,
    pub direction: glam::Vec3,
    pub up: glam::Vec3,
}

impl View {
    fn get_matrix(&self) -> glam::Mat4 {
        glam::Mat4::look_to_lh(self.eye, self.direction, self.up)
    }
}

pub struct CameraController {
    view: View,
    perspective: Perspective,
    speed: f32,
    sensitivity: f32,
    yaw: f32,
    pitch: f32,
}

impl CameraController {
    pub fn new(
        eye: glam::Vec3,
        yaw: f32,
        pitch: f32,
        fov_y: f32,
        aspect_ratio: f32,
        z_near: f32,
        z_far: f32,
        speed: f32,
        sensitivity: f32,
    ) -> Self {
        CameraController {
            view: View {
                eye,
                direction: glam::Vec3::X,
                up: glam::vec3(0.0, 1.0, 0.0),
            },
            perspective: Perspective {
                fov_y,
                aspect_ratio,
                z_near,
                z_far,
            },
            speed,
            sensitivity,
            yaw,
            pitch,
        }
    }

    pub fn handle_input(&mut self, pressed_keys: &HashSet<KeyCode>, mouse_movement: (f64, f64)) {
        let (dx, dy) = mouse_movement;

        let new_yaw = self.yaw - (dx as f32) * self.sensitivity;
        let new_pitch = (self.pitch - (dy as f32) * self.sensitivity).clamp(-0.5, 0.5);

        let (yaw_sin, yaw_cos) = ((new_yaw) * PI).sin_cos();
        let (pitch_sin, pitch_cos) = ((new_pitch) * PI).sin_cos();

        let xz_forward = glam::vec3(yaw_cos, 0.0, yaw_sin);
        let xz_right = glam::vec3(-yaw_sin, 0.0, yaw_cos);

        self.view.direction = glam::vec3(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin);
        self.yaw = new_yaw;
        self.pitch = new_pitch;

        self.view.up = xz_right.cross(self.view.direction);

        let mut speed = self.speed;

        if pressed_keys.contains(&KeyCode::ShiftLeft) {
            speed *= 3.0;
        }

        if pressed_keys.contains(&KeyCode::KeyW) {
            self.view.eye += xz_forward * speed;
        }

        if pressed_keys.contains(&KeyCode::KeyS) {
            self.view.eye -= xz_forward * speed;
        }

        if pressed_keys.contains(&KeyCode::KeyD) {
            self.view.eye -= xz_right * speed;
        }

        if pressed_keys.contains(&KeyCode::KeyA) {
            self.view.eye += xz_right * speed;
        }

        if pressed_keys.contains(&KeyCode::Space) {
            self.view.eye += glam::Vec3::Y * speed;
        }

        if pressed_keys.contains(&KeyCode::ControlLeft) {
            self.view.eye -= glam::Vec3::Y * speed;
        }
    }

    pub fn get_view_projection_matrix(&self) -> glam::Mat4 {
        self.perspective.get_matrix() * self.view.get_matrix()
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.perspective.aspect_ratio = aspect_ratio;
    }

    pub fn get_position(&self) -> Vec3 {
        self.view.eye
    }

    #[allow(dead_code)]
    pub fn get_direction(&self) -> Vec3 {
        self.view.direction
    }
}
