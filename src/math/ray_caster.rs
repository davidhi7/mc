use glam::{ivec3, IVec3, Vec3};

use crate::world::blocks::Direction;

#[derive(Clone, Copy, Debug)]
pub struct RaycastHit {
    /// Integer coordinates of the intersected voxel
    pub voxel: IVec3,
    /// Direction of the intersected voxel face. None, if origin is inside this voxel
    /// Example: If ray is directed in negative x direction and hits a voxel, direction is Direction::X
    // pub voxel_face: Option<Direction>,
    /// TODO update
    pub voxel_face: Direction,
    /// Value of t, so that the intersection is at origin + t * direction
    pub t: f32,
}

pub enum RaycastStatus {
    Continue,
    Stop,
}

// Implementation of this: http://www.cse.yorku.ca/~amana/research/grid.pdf
pub fn cast_ray(
    origin: Vec3,
    direction: Vec3,
    distance: f32,
    mut callback: impl FnMut(RaycastHit) -> RaycastStatus,
) {
    let step_x = direction.x.signum() as i32;
    let step_y = direction.y.signum() as i32;
    let step_z = direction.z.signum() as i32;

    let inverse_direction_x = if step_x > 0 {
        Direction::NegX
    } else {
        Direction::X
    };

    let inverse_direction_y = if step_y > 0 {
        Direction::NegY
    } else {
        Direction::Y
    };

    let inverse_direction_z = if step_z > 0 {
        Direction::NegZ
    } else {
        Direction::Z
    };

    let t_dx = 1.0 / direction.x.abs();
    let t_dy = 1.0 / direction.y.abs();
    let t_dz = 1.0 / direction.z.abs();

    let mut t_max_x = get_tmax(origin.x, direction.x);
    let mut t_max_y = get_tmax(origin.y, direction.y);
    let mut t_max_z = get_tmax(origin.z, direction.z);

    let start_voxel = origin.floor();
    let mut x = start_voxel.x as i32;
    let mut y = start_voxel.y as i32;
    let mut z = start_voxel.z as i32;
    // TODO why?
    // if direction.x >= 0.0 && origin.x.fract() == 0.0 {
    //     callback(RaycastHit {
    //         voxel: ivec3(x, y, z),
    //         voxel_face: inverse_direction_x,
    //         t: 0.0,
    //     });
    //     t_max_x = t_dx;
    // }
    // if direction.y >= 0.0 && origin.y.fract() == 0.0 {
    //     callback(RaycastHit {
    //         voxel: ivec3(x, y, z),
    //         voxel_face: inverse_direction_y,
    //         t: 0.0,
    //     });
    //     t_max_y = t_dy;
    // }
    // if direction.z >= 0.0 && origin.z.fract() == 0.0 {
    //     callback(RaycastHit {
    //         voxel: ivec3(x, y, z),
    //         voxel_face: inverse_direction_z,
    //         t: 0.0,
    //     });
    //     t_max_z = t_dz;
    // }

    // TODO prüfen
    // if let RaycastStatus::Stop = callback(RaycastHit {
    //     voxel: ivec3(x, y, z),
    //     voxel_face: None,
    //     t: 0.0,
    // }) {
    //     return;
    // }

    loop {
        let min_t_max = t_max_x.min(t_max_y).min(t_max_z);
        if min_t_max >= distance {
            break;
        }
        let direction;

        if min_t_max == t_max_x {
            // Update x
            x += step_x;
            t_max_x += t_dx;
            direction = inverse_direction_x;
        } else if min_t_max == t_max_y {
            // Update y
            y += step_y;
            t_max_y += t_dy;
            direction = inverse_direction_y;
        } else {
            // Update z
            z += step_z;
            t_max_z += t_dz;
            direction = inverse_direction_z;
        }

        if let RaycastStatus::Stop = callback(RaycastHit {
            voxel: ivec3(x, y, z),
            voxel_face: direction,
            t: min_t_max,
        }) {
            break;
        }
    }
}

/// Minimum positive t so that origin + t * direction is an integer, but always f32::INFINITY if direction is 0
fn get_tmax(origin: f32, direction: f32) -> f32 {
    // Distance between origin_fract and the next smaller integer
    let origin_fract = origin - origin.floor();

    if direction == 0.0 {
        return f32::INFINITY;
    }

    if origin_fract == 0.0 {
        return 0.0;
    }

    // If direction is positive, we need the distance to the next integer, not last integer
    let distance_to_next_voxel = if direction.signum() > 0.0 {
        1.0 - origin_fract
    } else {
        origin_fract
    };

    // Compute value of t, so that origin + t * direction is the nearest integer value
    distance_to_next_voxel / direction.abs()
}

#[cfg(test)]
mod tests {

    use googletest::{assert_that, prelude::approx_eq};

    use super::*;

    #[test]
    fn test_get_tmax_zero() {
        assert_that!(get_tmax(0.0, 1.0), approx_eq(0.0));
        assert_that!(get_tmax(1.0, 1.0), approx_eq(0.0));
        assert_that!(get_tmax(-1.0, 1.0), approx_eq(0.0));
    }

    /// positive position
    #[test]
    fn test_get_tmax_positive() {
        assert_that!(get_tmax(0.5, 1.0), approx_eq(0.5));
        assert_that!(get_tmax(0.3, 1.0), approx_eq(0.7));
        assert_that!(get_tmax(0.3, -1.0), approx_eq(0.3));
        assert_that!(get_tmax(0.25, 0.25), approx_eq(3.0));
    }

    /// negative position
    #[test]
    fn test_get_tmax_negative() {
        assert_that!(get_tmax(-0.5, 1.0), approx_eq(0.5));
        assert_that!(get_tmax(-0.3, 1.0), approx_eq(0.3));
        assert_that!(get_tmax(-0.3, -1.0), approx_eq(0.7));
        assert_that!(get_tmax(-0.75, 0.25), approx_eq(3.0));
    }

    /// extreme cases
    #[test]
    fn test_get_tmax_extremes() {
        assert!(!get_tmax(1.1, 0.0).is_finite());
    }
}
