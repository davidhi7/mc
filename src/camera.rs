use core::f32;
use std::f32::consts::PI;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, vec3};

pub mod block_ray_caster;
pub mod player;

/// Container for normalized yaw and pitch values. See the two attributes.
#[derive(Debug, Clone, Copy)]
pub struct YawPitch {
    /// Horizontal camera orientation. Within [0.0, 2.0). 0.0 is facing towards X+ / east; 0.5 is facing towards Z+ / north.
    pub yaw_norm: f32,
    /// Vertical camera orientation. Within [-0.5, 0.5]. 0.0 is facing forward; -0.5 is facing downward.
    pub pitch_norm: f32,
}

pub fn yaw_pitch_to_direction(
    YawPitch {
        yaw_norm,
        pitch_norm,
    }: YawPitch,
) -> Vec3 {
    let (yaw_sin, yaw_cos) = (yaw_norm * PI).sin_cos();
    let (pitch_sin, pitch_cos) = (pitch_norm * PI).sin_cos();
    vec3(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin)
}

pub fn direction_to_yaw_pitch(direction: Vec3) -> YawPitch {
    YawPitch {
        // Convert [-1.0, 1.0] to [0.0, 2.0]
        yaw_norm: (f32::atan2(direction.z, direction.x) / PI + 2.0) % 2.0,
        pitch_norm: f32::asin(direction.y) / PI,
    }
}

pub trait ToMatrix {
    fn matrix(&self) -> Mat4;
}

pub trait ToPlanes {
    fn planes(&self, view: View) -> CameraPlanes;
}

#[derive(Debug, Clone, Copy)]
pub enum Projection {
    Perspective(PerspectiveProj),
    Orthographic(OrthographicProj),
}

impl ToMatrix for Projection {
    fn matrix(&self) -> Mat4 {
        match self {
            Projection::Perspective(perspective_proj) => perspective_proj.matrix(),
            Projection::Orthographic(orthographic_proj) => orthographic_proj.matrix(),
        }
    }
}

impl ToPlanes for Projection {
    fn planes(&self, view: View) -> CameraPlanes {
        match self {
            Projection::Perspective(perspective_proj) => perspective_proj.planes(view),
            Projection::Orthographic(orthographic_proj) => orthographic_proj.planes(view),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PerspectiveProj {
    pub fov_y_rad: f32,
    pub aspect_ratio: f32,
    pub z_near: f32,
    pub z_far: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct OrthographicProj {
    pub left: f32,
    pub right: f32,
    pub bottom: f32,
    pub top: f32,
    pub near: f32,
    pub far: f32,
}

impl ToMatrix for PerspectiveProj {
    fn matrix(&self) -> Mat4 {
        Mat4::perspective_lh(self.fov_y_rad, self.aspect_ratio, self.z_near, self.z_far)
    }
}

impl ToPlanes for PerspectiveProj {
    fn planes(&self, view: View) -> CameraPlanes {
        let half_v_side = self.z_far * f32::tan(self.fov_y_rad / 2.0);
        let half_h_side = half_v_side * self.aspect_ratio;

        let direction_z_near = view.direction * self.z_near;
        let direction_z_far = view.direction * self.z_far;

        let origin = view.eye;
        let up = view.up;
        let right = up.cross(view.direction);

        let frustum_zfar_t = (direction_z_far + half_v_side * up).normalize();
        let frustum_zfar_b = (direction_z_far - half_v_side * up).normalize();
        let frustum_zfar_l = (direction_z_far - half_h_side * right).normalize();
        let frustum_zfar_r = (direction_z_far + half_h_side * right).normalize();

        let normal_top = frustum_zfar_t.cross(right).normalize();
        let normal_bottom = right.cross(frustum_zfar_b).normalize();
        let normal_left = frustum_zfar_l.cross(up).normalize();
        let normal_right = up.cross(frustum_zfar_r).normalize();

        CameraPlanes {
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

impl ToMatrix for OrthographicProj {
    fn matrix(&self) -> Mat4 {
        Mat4::orthographic_lh(
            self.left,
            self.right,
            self.bottom,
            self.top,
            self.near,
            self.far,
        )
    }
}

impl ToPlanes for OrthographicProj {
    fn planes(&self, view: View) -> CameraPlanes {
        let origin = view.eye;
        let direction = view.direction;
        let up = view.up;
        let right = up.cross(direction);

        CameraPlanes {
            top: Plane {
                normal: up,
                distance: (origin + up * self.top).dot(up),
            },
            bottom: Plane {
                normal: -up,
                distance: (origin + up * self.bottom).dot(-up),
            },
            left: Plane {
                normal: -right,
                distance: (origin + right * self.left).dot(-right),
            },
            right: Plane {
                normal: right,
                distance: (origin + right * self.right).dot(right),
            },
            near: Plane {
                normal: -direction,
                distance: (origin + direction * self.near).dot(-direction),
            },
            far: Plane {
                normal: direction,
                distance: (origin + direction * self.far).dot(direction),
            },
        }
    }
}

/// `direction` and `up` must be orthogonal.
#[derive(Debug, Clone, Copy)]
pub struct View {
    pub eye: Vec3,
    pub direction: Vec3,
    pub up: Vec3,
    _private: (),
}

impl View {
    pub fn new(eye: Vec3, direction: Vec3, up: Vec3) -> Self {
        assert!(direction.is_normalized(), "`direction` must be normalized");
        assert!(up.is_normalized(), "`up` must be normalized");
        assert!(
            direction.dot(up).abs() <= 1e-5,
            "`direction` and `up` must be orthogonal"
        );
        Self {
            eye,
            direction,
            up,
            _private: (),
        }
    }
}

impl ToMatrix for View {
    fn matrix(&self) -> Mat4 {
        Mat4::look_to_lh(self.eye, self.direction, self.up)
    }
}

#[derive(Debug, Clone, Copy, Default, Zeroable, Pod)]
#[repr(C)]
pub struct ViewProjectionMatrix([[f32; 4]; 4]);

impl ViewProjectionMatrix {
    pub fn new(view: View, projection: Projection) -> Self {
        Self((projection.matrix() * view.matrix()).to_cols_array_2d())
    }
}

#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct Plane {
    pub normal: Vec3,
    pub distance: f32,
}

/// Planes of the camera frustum (for perspective projections) and rect (for orthographic projections).
/// All plane normals should direct towards the outside of the volume.
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct CameraPlanes {
    top: Plane,
    bottom: Plane,
    left: Plane,
    right: Plane,
    near: Plane,
    far: Plane,
}
