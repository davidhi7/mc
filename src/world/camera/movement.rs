use glam::{ivec3, vec3, IVec3, Vec3};

use crate::{
    math::ray_caster::{self, RaycastHit},
    world::World,
};

const EYE_HEIGHT: f32 = 1.6;
const HITBOX_EXTENDS: Vec3 = vec3(0.6, 1.8, 0.6);

struct IPlane {
    normal: IVec3,
    distance: i32,
}

impl IPlane {
    fn contains_point(&self, point: IVec3) -> bool {
        self.normal.dot(point) == self.distance
    }
}

fn get_hitbox_sample_points(eye: &Vec3) -> Vec<Vec3> {
    let hitbox_bottom_center = *eye - vec3(0.0, EYE_HEIGHT, 0.0);
    let hitbox_top_center = *eye + vec3(0.0, HITBOX_EXTENDS.y - EYE_HEIGHT, 0.0);

    vec![
        hitbox_bottom_center + vec3(-HITBOX_EXTENDS.x / 2.0, 0.0, -HITBOX_EXTENDS.z / 2.0),
        hitbox_bottom_center + vec3(-HITBOX_EXTENDS.x / 2.0, 0.0, HITBOX_EXTENDS.z / 2.0),
        hitbox_bottom_center + vec3(HITBOX_EXTENDS.x / 2.0, 0.0, -HITBOX_EXTENDS.z / 2.0),
        hitbox_bottom_center + vec3(HITBOX_EXTENDS.x / 2.0, 0.0, HITBOX_EXTENDS.z / 2.0),
        hitbox_top_center + vec3(-HITBOX_EXTENDS.x / 2.0, 0.0, -HITBOX_EXTENDS.z / 2.0),
        hitbox_top_center + vec3(-HITBOX_EXTENDS.x / 2.0, 0.0, HITBOX_EXTENDS.z / 2.0),
        hitbox_top_center + vec3(HITBOX_EXTENDS.x / 2.0, 0.0, -HITBOX_EXTENDS.z / 2.0),
        hitbox_top_center + vec3(HITBOX_EXTENDS.x / 2.0, 0.0, HITBOX_EXTENDS.z / 2.0),
    ]
}

pub(super) fn handle_collisions(
    world: &World,
    mut eye: Vec3,
    direction: Vec3,
    distance: f32,
) -> Vec3 {
    debug_assert!(direction.is_normalized());

    let first_hit = raycast_hitbox(world, eye, direction, distance, vec![]);
    if first_hit.is_none() {
        return eye + distance * direction;
    }

    let RaycastHit {
        voxel: _,
        voxel_face,
        t,
    } = first_hit.unwrap();

    eye += t * direction;
    let eye_i32 = ivec3(
        eye.x.floor() as i32,
        eye.y.floor() as i32,
        eye.z.floor() as i32,
    );

    let tmp = voxel_face.get_unit_vector().cross(direction);
    if tmp.length() == 0.0 {
        // TODO fix, don't allow collision
        return eye;
    }

    // TODO validate these values
    let first_slide_direction = tmp.cross(voxel_face.get_unit_vector()).normalize();
    let first_slide_distance = first_slide_direction.dot(direction) * (distance - t);

    let second_hit = raycast_hitbox(
        world,
        eye,
        first_slide_direction,
        first_slide_distance,
        vec![IPlane {
            normal: voxel_face.get_unit_vector_i32(),
            distance: eye_i32.dot(voxel_face.get_unit_vector_i32()),
        }],
    );
    if second_hit.is_none() {
        return eye + first_slide_direction * first_slide_distance;
    }

    let RaycastHit {
        voxel: _,
        voxel_face: second_voxel_face,
        t: second_t,
    } = second_hit.unwrap();

    eye += first_slide_direction * second_t;
    let eye_i32 = ivec3(
        eye.x.floor() as i32,
        eye.y.floor() as i32,
        eye.z.floor() as i32,
    );

    // third segment
    // TODO validate these values
    let mut slide_direction = voxel_face
        .get_unit_vector()
        .cross(second_voxel_face.get_unit_vector());
    let mut slide_length = (first_slide_distance - second_t) * slide_direction.dot(direction);
    if slide_length < 0.0 {
        // TODO ?
        slide_direction *= -1.0;
        slide_length *= -1.0;
    }

    let third_hit = raycast_hitbox(
        world,
        eye,
        slide_direction,
        slide_length,
        vec![
            IPlane {
                normal: voxel_face.get_unit_vector_i32(),
                distance: eye_i32.dot(voxel_face.get_unit_vector_i32()),
            },
            IPlane {
                normal: second_voxel_face.get_unit_vector_i32(),
                distance: eye_i32.dot(second_voxel_face.get_unit_vector_i32()),
            },
        ],
    );
    println!("{:?} {:?}", voxel_face, second_voxel_face);
    if third_hit.is_none() {
        return eye + slide_direction * slide_length;
    }
    let RaycastHit {
        voxel: _,
        voxel_face: _,
        t: third_t,
    } = third_hit.unwrap();

    eye + third_t * slide_direction
}

fn raycast_hitbox(
    world: &World,
    eye: Vec3,
    direction: Vec3,
    distance: f32,
    obligatory_planes: Vec<IPlane>,
) -> Option<RaycastHit> {
    // First hit of a solid block (measured by t value) of any hitbox sample point
    let mut first_hit = None;

    for mut sample_point in get_hitbox_sample_points(&eye) {
        if (sample_point.x.fract() < 0.01) && sample_point.x > eye.x {
            sample_point.x -= 0.02;
        }

        if (sample_point.y.fract() < 0.01) && sample_point.y > eye.y {
            sample_point.y -= 0.02;
        }

        if (sample_point.z.fract() < 0.01) && sample_point.z > eye.z {
            sample_point.z -= 0.02;
        }

        ray_caster::cast_ray(sample_point, direction, distance, |hit| {
            let block_is_solid = world
                .get_block(hit.voxel)
                .is_some_and(|block| block.is_solid());

            let voxel_on_obligatory_planes = obligatory_planes
                .iter()
                .all(|plane| plane.contains_point(hit.voxel));

            let voxel_face_rule = direction.dot(hit.voxel_face.get_unit_vector()) < 0.0;

            if block_is_solid
                && voxel_on_obligatory_planes
                && first_hit.is_none_or(|first: RaycastHit| first.t > hit.t)
                && voxel_face_rule
            {
                first_hit = Some(hit);
                ray_caster::RaycastStatus::Stop
            } else if first_hit.is_some_and(|first| first.t <= hit.t) {
                ray_caster::RaycastStatus::Stop
            } else {
                ray_caster::RaycastStatus::Continue
            }
        });
    }

    first_hit
}
