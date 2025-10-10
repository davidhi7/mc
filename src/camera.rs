use core::f32;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

pub mod block_ray_caster;
pub mod player;

#[derive(Debug, Clone, Copy)]
pub struct Perspective {
    pub fov_y: f32,
    pub aspect_ratio: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Perspective {
    pub fn get_matrix(&self) -> Mat4 {
        Mat4::perspective_lh(self.fov_y, self.aspect_ratio, self.z_near, self.z_far)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct View {
    pub eye: Vec3,
    pub direction: Vec3,
    pub up: Vec3,
}

impl View {
    pub fn get_matrix(&self) -> Mat4 {
        Mat4::look_to_lh(self.eye, self.direction, self.up)
    }
}

#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct Plane {
    pub normal: Vec3,
    pub distance: f32,
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
    pub fn from_camera(view: &View, perspective: &Perspective) -> Self {
        let half_v_side = perspective.z_far * f32::tan(perspective.fov_y / 2.0);
        let half_h_side = half_v_side * perspective.aspect_ratio;

        let direction_z_near = view.direction * perspective.z_near;
        let direction_z_far = view.direction * perspective.z_far;

        let origin = view.eye;
        let up = view.up;
        let right = view.up.cross(view.direction);

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
                normal: -view.direction,
                distance: (origin + direction_z_near).dot(-view.direction),
            },
            far: Plane {
                normal: view.direction,
                distance: (origin + direction_z_far).dot(view.direction),
            },
        }
    }
}
