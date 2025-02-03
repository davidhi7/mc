use core::f32;
use std::{collections::HashSet, f32::consts::PI};

use bytemuck::{Pod, Zeroable};
use glam::{vec3, Mat4, Vec3};
use winit::keyboard::KeyCode;

use crate::math::Plane;

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
    right: Vec3,
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
    /// Horizontal camera orientation multiplied with pi. Within [0.0, 2.0). 0.0 is facing towards X+ / east; 0.5 is facing towards Z+ / north
    yaw: f32,
    /// vertical camera orientation coefficient multiplied with pi. Within [-0.5, 0.5]. 0.0 is facing forward; -0.5 is facing downward
    pitch: f32,
}

impl CameraController {
    pub fn new(
        eye: Vec3,
        direction: Vec3,
        up: Vec3,
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
                direction,
                up,
                right: up.cross(direction),
            },
            perspective: Perspective {
                fov_y,
                aspect_ratio,
                z_near,
                z_far,
            },
            speed,
            sensitivity,
            yaw: f32::atan2(direction.x, direction.z),
            pitch: f32::atan(direction.y),
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
        // Effectively rotate xz_forward by 90 degrees
        let xz_right = vec3(yaw_sin, 0.0, -yaw_cos);

        self.view.direction = vec3(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin);
        self.yaw = new_yaw;
        self.pitch = new_pitch;

        self.view.right = xz_right;
        self.view.up = self.view.direction.cross(self.view.right);

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
            self.view.eye -= xz_right * time_adjusted_speed * speed_multiplier;
        }

        if pressed_keys.contains(&KeyCode::KeyD) {
            self.view.eye += xz_right * time_adjusted_speed * speed_multiplier;
        }

        if pressed_keys.contains(&KeyCode::Space) {
            self.view.eye.y += time_adjusted_speed * speed_multiplier;
        }

        if pressed_keys.contains(&KeyCode::ControlLeft) {
            self.view.eye.y -= time_adjusted_speed * speed_multiplier;
        }

        println!("{}", self.yaw);
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

#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct CameraFrustum {
    pub top: Plane,
    pub bottom: Plane,
    pub left: Plane,
    pub right: Plane,
    pub near: Plane,
    pub far: Plane,
}

impl CameraFrustum {
    pub fn from_camera(camera: &CameraController) -> Self {
        let half_v_side = camera.perspective.z_far * f32::tan(camera.perspective.fov_y / 2.0);
        let half_h_side = half_v_side * camera.perspective.aspect_ratio;

        let direction_z_near = camera.view.direction * camera.perspective.z_near;
        let direction_z_far = camera.view.direction * camera.perspective.z_far;

        let origin = camera.view.eye;
        let up = camera.view.up;
        let right = camera.view.right;

        let frustum_zfar_t = (direction_z_far + half_v_side * up).normalize();
        let frustum_zfar_b = (direction_z_far - half_v_side * up).normalize();
        let frustum_zfar_l = (direction_z_far - half_h_side * right).normalize();
        let frustum_zfar_r = (direction_z_far + half_h_side * right).normalize();

        // Since all vector pairs have two perpendicular vectors, no normalization is needed
        let normal_top = frustum_zfar_t.cross(right);
        let normal_bottom = right.cross(frustum_zfar_b);
        let normal_left = frustum_zfar_l.cross(up);
        let normal_right = up.cross(frustum_zfar_r);

        Self {
            top: Plane {
                normal: normal_top,
                distance: origin.dot(normal_top),
            },
            bottom: Plane {
                normal: normal_bottom,
                distance: origin.dot(normal_bottom),
            },
            left: Plane {
                normal: normal_left,
                distance: origin.dot(normal_left),
            },
            right: Plane {
                normal: normal_right,
                distance: origin.dot(normal_right),
            },
            near: Plane {
                normal: -camera.view.direction,
                distance: (origin + direction_z_near).dot(-camera.view.direction),
            },
            far: Plane {
                normal: camera.view.direction,
                distance: (origin + direction_z_far).dot(camera.view.direction),
            },
        }
    }
}
