use std::{collections::HashSet, f32::consts::PI};

use glam::{vec3, Mat4, Vec3};
use winit::keyboard::KeyCode;

struct Perspective {
    fov_y: f32,
    aspect_ratio: f32,
    z_near: f32,
    z_far: f32,
}

impl Perspective {
    fn get_matrix(&self) -> Mat4 {
        Mat4::perspective_lh(self.fov_y, self.aspect_ratio, self.z_near, self.z_far)
    }
}

struct View {
    eye: Vec3,
    direction: Vec3,
    up: Vec3,
}

impl View {
    fn get_matrix(&self) -> Mat4 {
        Mat4::look_to_lh(self.eye, self.direction, self.up)
    }
}

pub struct CameraController {
    view: View,
    perspective: Perspective,
    /// Camera translation per second
    speed: f32,
    /// Camera rotation per mouse movement step, multiplied by pi
    sensitivity: f32,
    /// Horizontal camera orientation; 0.0 is facing towards X+ / east
    yaw: f32,
    /// vertical camera orientation within [-0.5, 0.5]; 0 is facing forward
    pitch: f32,
}

impl CameraController {
    pub fn new(
        eye: Vec3,
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
                direction: Vec3::X,
                up: vec3(0.0, 1.0, 0.0),
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

    pub fn handle_input(
        &mut self,
        pressed_keys: &HashSet<KeyCode>,
        mouse_movement: (f64, f64),
        delta_s: f32,
    ) {
        let (dx, dy) = mouse_movement;
        let time_adjusted_speed = self.speed * delta_s;
        let mut speed_multiplier = 1.0;

        let mut new_yaw = self.yaw - (dx as f32) * self.sensitivity;
        // Normalize yaw value
        new_yaw %= 2.0;
        if new_yaw < 0.0 {
            new_yaw += 2.0;
        }

        let new_pitch = (self.pitch - (dy as f32) * self.sensitivity).clamp(-0.5, 0.5);

        let (yaw_sin, yaw_cos) = ((new_yaw) * PI).sin_cos();
        let (pitch_sin, pitch_cos) = ((new_pitch) * PI).sin_cos();

        let xz_forward = vec3(yaw_cos, 0.0, yaw_sin);
        let xz_right = vec3(-yaw_sin, 0.0, yaw_cos);

        self.view.direction = vec3(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin);
        self.yaw = new_yaw;
        self.pitch = new_pitch;

        self.view.up = xz_right.cross(self.view.direction);

        if pressed_keys.contains(&KeyCode::ShiftLeft) {
            speed_multiplier = 3.0;
        }

        if pressed_keys.contains(&KeyCode::KeyW) {
            self.view.eye += xz_forward * time_adjusted_speed * speed_multiplier;
        }

        if pressed_keys.contains(&KeyCode::KeyS) {
            self.view.eye -= xz_forward * time_adjusted_speed * speed_multiplier;
        }

        if pressed_keys.contains(&KeyCode::KeyA) {
            self.view.eye += xz_right * time_adjusted_speed * speed_multiplier;
        }

        if pressed_keys.contains(&KeyCode::KeyD) {
            self.view.eye -= xz_right * time_adjusted_speed * speed_multiplier;
        }

        if pressed_keys.contains(&KeyCode::Space) {
            self.view.eye.y += time_adjusted_speed * speed_multiplier;
        }

        if pressed_keys.contains(&KeyCode::ControlLeft) {
            self.view.eye.y -= time_adjusted_speed * speed_multiplier;
        }
    }

    pub fn get_view_projection_matrix(&self) -> Mat4 {
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
