use core::f32;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

pub mod block_ray_caster;
pub mod player;

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
    /// Horizontal camera orientation when multiplied with pi. Within [0.0, 2.0). 0.0 is facing towards X+ / east; 0.5 is facing towards Z+ / north
    yaw: f32,
    /// vertical camera orientation when multiplied with pi. Within [-0.5, 0.5]. 0.0 is facing forward; -0.5 is facing downward
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
            yaw: f32::atan2(direction.x, direction.z),
            pitch: f32::atan(direction.y),
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

#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct CameraFrustum {
    top: Plane,
    bottom: Plane,
    left: Plane,
    right: Plane,
    near: Plane,
    far: Plane,
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
