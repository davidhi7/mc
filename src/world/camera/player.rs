/*
Loosely based on https://www.mcpk.wiki/wiki/Movement_Formulas.
All acceleration/velocity values are m/s and m/s^2 instead of m/t and m/t^2 which is used by minecraft 1.8 with t=1/20s.

The maximum movement speed in the xz plane is the limit of the sequence `v(t) = v(t-1) * BASE_FRICTION + BASE_ACCEL`, which is `BASE_ACCEL/(1-BASE_FRICTION)`.
The terminal velocity in free fall is the limit of the sequence `v(t) = (v(t-1) + GRAVITY) * VERTICAL_DRAG`, which is `VERTICAL_DRAG * GRAVITY / (1-VERTICAL_DRAG)`.

When initiating a jump, the vertical velocity is set to `JUMP_ACCEL` once.
Also when sprinting during jumping (sprinting meaning the sprint key is pressed and velocity.xz() has a length greater than one), the velocity is incremented by `SPRINT_JUMP_ACCEL` once in the current movement direction projected to xz.
*/
use std::{
    collections::HashSet,
    f32::consts::{FRAC_1_SQRT_2, PI},
};

use glam::{IVec3, Mat3, Mat4, Vec3, ivec3, vec3};
use winit::keyboard::KeyCode;

use crate::{
    math::{Aabb3, Aabb3I},
    world::camera::{Perspective, View},
};

/// Multiplied by mouse dx/dy, then added or subtracted from [`PlayerState::yaw`], [`PlayerState::pitch`]
const CAMERA_SENSITIVITY: f32 = 0.002;

const HITBOX_HEIGHT: f32 = 1.8;
const HITBOX_WIDTH: f32 = 0.6;
const HALF_HITBOX_WIDTH: f32 = HITBOX_WIDTH / 2.0;
const EYE_HEIGHT: f32 = 1.6;

/// Minimum possible velocity value per axis. If actual velocity is less than this, it is set to zero.
const MIN_VELOCITY_THRESHOLD: f32 = 20.0 * 0.005;

// Acceleration by direction in the xz plane, measured in m/s^2 but doesn't take drag/friction into account.
const BASE_ACCEL_GROUND: f32 = 20.0 * 0.1 * 1.0 * 0.98;
const BASE_ACCEL_AIRBORNE: f32 = 20.0 * 0.02 * 1.0;
/// Acceleration in the xz plane in the current movement direction when jumping
const SPRINT_JUMP_ACCEL: f32 = 20.0 * 0.2;

// "Friction", that is the factor of velocity in the xz plane that is conserved after every tick.
const BASE_FRICTION_GROUND: f32 = 0.91 * 0.6;
const BASE_FRICTION_AIRBORNE: f32 = 0.91 * 1.0;

/// The velocity along the y axis that initiates a jump.
const JUMP_ACCEL: f32 = 2.0 * 20.0 * 0.42;
/// Acceleration along the y axis during free fall.
const GRAVITY: f32 = 20.0 * -0.08;
// Vertical drag, that is the factor of velocity that is conserved after every tick
const VERTICAL_DRAG: f32 = 0.98;

/// Vertical velocity when flying in m/s
const FLYING_Y_VELOCITY: f32 = 5.0;

#[derive(Debug, Clone, Copy)]
enum MovementState {
    Walking,
    AirBorne,
    Flying { flying_up: bool, flying_down: bool },
}

pub struct PlayerState {
    /// Camera perspective
    perspective: Perspective,
    /// Horizontal camera orientation when multiplied with pi. Within [0.0, 2.0). 0.0 is facing towards X+ / east; 0.5 is facing towards Z+ / north
    yaw: f32,
    /// vertical camera orientation when multiplied with pi. Within [-0.5, 0.5]. 0.0 is facing forward; -0.5 is facing downward
    pitch: f32,
    movement_state: MovementState,
    physics_state: PlayerPhysicsState,
}

#[derive(Debug, Clone, Copy)]
struct PlayerPhysicsState {
    /// Coordinates of the players eye (horizontal center, height of EYE_HEIGHT)
    eye: Vec3,
    /// Player hitbox aabb. Used to work around floating point errors during collision detection
    aabb: Aabb3,
    /// Velocity in m/s
    velocity: Vec3,
    /// Expected future acceleration in m/s^2.
    /// Only used for prediction of the next player position.
    acceleration: Vec3,
    /// Collision info from last tick. Note that if the player didn't move in the xz plane during the previous tick, all collisions along the x and z axis are false.
    collisions: CollisionResult,
}

impl PlayerState {
    pub fn new(perspective: Perspective, eye: Vec3, direction: Vec3) -> Self {
        Self {
            perspective,
            // TODO check yaw and pitch
            yaw: f32::atan2(direction.x, direction.z),
            pitch: f32::atan(direction.y),
            // movement_state: MovementState::Flying {
            //     flying_up: false,
            //     flying_down: false,
            // },
            movement_state: MovementState::Walking,
            physics_state: PlayerPhysicsState {
                eye,
                aabb: {
                    let min = eye - vec3(HALF_HITBOX_WIDTH, EYE_HEIGHT, HALF_HITBOX_WIDTH);
                    let max = eye
                        + vec3(
                            HALF_HITBOX_WIDTH,
                            HITBOX_HEIGHT - EYE_HEIGHT,
                            HALF_HITBOX_WIDTH,
                        );

                    Aabb3 { min, max }
                },
                velocity: Vec3::ZERO,
                acceleration: Vec3::ZERO,
                collisions: CollisionResult {
                    neg_x_collision: false,
                    pos_x_collision: false,
                    neg_y_collision: false,
                    pos_y_collision: false,
                    neg_z_collision: false,
                    pos_z_collision: false,
                },
            },
        }
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.perspective.aspect_ratio = aspect_ratio;
    }

    pub fn update_rotation(&mut self, (dx, dy): (f64, f64)) {
        let mut new_yaw = self.yaw - (dx as f32) * CAMERA_SENSITIVITY;
        let new_pitch = (self.pitch - (dy as f32) * CAMERA_SENSITIVITY)
            .clamp(-0.5 + f32::EPSILON, 0.5 - f32::EPSILON);

        // Normalize yaw value
        new_yaw %= 2.0;
        if new_yaw < 0.0 {
            new_yaw += 2.0;
        }

        self.yaw = new_yaw;
        self.pitch = new_pitch;
    }

    pub fn update_position(
        &mut self,
        pressed_keys: &HashSet<KeyCode>,
        delta_s: f32,
        time_s: f32,
        check_is_solid: impl Fn(IVec3) -> bool,
    ) {
        let is_sprinting = pressed_keys.contains(&KeyCode::ShiftLeft);

        let base_acceleration = match self.movement_state {
            MovementState::Walking => BASE_ACCEL_GROUND,
            MovementState::Flying { .. } | MovementState::AirBorne => BASE_ACCEL_AIRBORNE,
        } * if is_sprinting { 1.3 } else { 1.0 };

        let base_friction = match self.movement_state {
            MovementState::Walking => BASE_FRICTION_GROUND,
            MovementState::Flying { .. } | MovementState::AirBorne => BASE_FRICTION_AIRBORNE,
        };

        // If we are going to move along both axes, use 1/sqrt(2) as coefficient so the maximum diagonal speed can't exceed the maximum straight speed
        let diagonal_correction = if pressed_keys.contains(&KeyCode::KeyW)
            ^ pressed_keys.contains(&KeyCode::KeyS)
            && pressed_keys.contains(&KeyCode::KeyA) ^ pressed_keys.contains(&KeyCode::KeyD)
        {
            FRAC_1_SQRT_2
        } else {
            1.0
        };

        // Acceleration rotated so that +x is forward and +z left. Note that this acceleration doesn't include friction/drag.
        let mut rotated_acceleration = Vec3::ZERO;

        if pressed_keys.contains(&KeyCode::KeyW) {
            rotated_acceleration.x += diagonal_correction * base_acceleration;
        }

        if pressed_keys.contains(&KeyCode::KeyS) {
            rotated_acceleration.x -= diagonal_correction * base_acceleration;
        }

        if pressed_keys.contains(&KeyCode::KeyA) {
            rotated_acceleration.z += diagonal_correction * base_acceleration;
        }

        if pressed_keys.contains(&KeyCode::KeyD) {
            rotated_acceleration.z -= diagonal_correction * base_acceleration;
        }

        // rotate acceleration so that it uses the world coordinate system
        // TODO why 2 - yaw?
        let mut world_acceleration =
            Mat3::from_rotation_y((2.0 - self.yaw) * PI) * rotated_acceleration;

        match self.movement_state {
            MovementState::Walking => {
                // Do a jump if space is pressed, note that jumping is possible even if the player is not on the ground anymore during the current tick
                if pressed_keys.contains(&KeyCode::Space) {
                    // sprint jump boost
                    if is_sprinting {
                        // TODO better direction measure than latest velocity direction?
                        world_acceleration += (self.physics_state.velocity + world_acceleration)
                            .with_y(0.0)
                            .normalize_or_zero()
                            * SPRINT_JUMP_ACCEL;
                    }
                    assert!(
                        self.physics_state.velocity.y == 0.0,
                        "Jump started while vertical velocity is not zero"
                    );
                    world_acceleration.y = JUMP_ACCEL;
                    self.movement_state = MovementState::AirBorne;
                } else {
                    world_acceleration.y += GRAVITY;
                }
            }
            MovementState::AirBorne => {
                world_acceleration.y += GRAVITY;
            }
            MovementState::Flying {
                ref mut flying_up,
                ref mut flying_down,
            } => {
                *flying_up = pressed_keys.contains(&KeyCode::Space);
                *flying_down = pressed_keys.contains(&KeyCode::ControlLeft);

                // Directly set velocity, since acceleration isn't continuous
                if *flying_up == *flying_down {
                    self.physics_state.velocity.y = 0.0;
                } else if *flying_up {
                    self.physics_state.velocity.y = FLYING_Y_VELOCITY;
                } else {
                    self.physics_state.velocity.y = -FLYING_Y_VELOCITY;
                }
            }
        }

        let velocity_before_accel = self.physics_state.velocity;
        self.physics_state.velocity += world_acceleration;
        // Drag along y axis is applied before sampling the velocity, drag along x/z is sampled after according to mcpk.wiki
        self.physics_state.velocity.y *= VERTICAL_DRAG;

        self.physics_state = resolve_collisions(
            self.physics_state,
            self.physics_state.velocity * delta_s,
            &check_is_solid,
        );
        self.physics_state.velocity.x *= base_friction;
        self.physics_state.velocity.z *= base_friction;

        let CollisionResult {
            neg_x_collision,
            pos_x_collision,
            neg_y_collision,
            pos_y_collision,
            neg_z_collision,
            pos_z_collision,
        } = self.physics_state.collisions;

        if let MovementState::Walking = self.movement_state
            && !neg_y_collision
        {
            self.movement_state = MovementState::AirBorne;
        }
        if let MovementState::AirBorne = self.movement_state
            && neg_y_collision
        {
            self.movement_state = MovementState::Walking;
        }

        self.physics_state.acceleration =
            (self.physics_state.velocity - velocity_before_accel) / delta_s;

        if neg_x_collision || pos_x_collision {
            self.physics_state.acceleration.x = 0.0;
        }
        if neg_y_collision || pos_y_collision {
            self.physics_state.acceleration.y = 0.0;
        }
        if neg_z_collision || pos_z_collision {
            self.physics_state.acceleration.z = 0.0;
        }

        if self.physics_state.velocity.x.abs() < MIN_VELOCITY_THRESHOLD {
            self.physics_state.velocity.x = 0.0;
        }
        if self.physics_state.velocity.y.abs() < MIN_VELOCITY_THRESHOLD {
            self.physics_state.velocity.y = 0.0;
        }
        if self.physics_state.velocity.z.abs() < MIN_VELOCITY_THRESHOLD {
            self.physics_state.velocity.z = 0.0;
        }

        println!("{time_s},{},", self.physics_state.eye.y);
        println!(
            "{},,{}",
            time_s + delta_s * 0.25,
            self.extrapolate_view(delta_s * 0.25, &check_is_solid).eye.y
        );
        println!(
            "{},,{}",
            time_s + delta_s * 0.5,
            self.extrapolate_view(delta_s * 0.5, &check_is_solid).eye.y
        );
        println!(
            "{},,{}",
            time_s + delta_s * 0.75,
            self.extrapolate_view(delta_s * 0.75, &check_is_solid).eye.y
        );
    }

    pub fn eye(&self) -> Vec3 {
        self.physics_state.eye
    }

    pub fn direction(&self) -> Vec3 {
        let (yaw_sin, yaw_cos) = (self.yaw * PI).sin_cos();
        let (pitch_sin, pitch_cos) = (self.pitch * PI).sin_cos();
        vec3(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin)
    }

    pub fn view(&self) -> View {
        View {
            eye: self.physics_state.eye,
            direction: self.direction(),
            up: Vec3::Y,
        }
    }

    pub fn perspective(&self) -> Perspective {
        self.perspective
    }

    pub fn view_projection(&self) -> Mat4 {
        self.perspective.get_matrix() * self.view().get_matrix()
    }

    pub fn extrapolate_view(&self, lag_s: f32, check_is_solid: &impl Fn(IVec3) -> bool) -> View {
        let extrapolated_state = resolve_collisions(
            self.physics_state,
            self.physics_state.velocity * lag_s
                + 0.5 * self.physics_state.acceleration * lag_s.powi(2),
            check_is_solid,
        );

        View {
            eye: extrapolated_state.eye,
            direction: self.direction(),
            up: Vec3::Y,
        }
    }

    pub fn extrapolate_view_projection(
        &self,
        lag_s: f32,
        check_is_solid: &impl Fn(IVec3) -> bool,
    ) -> Mat4 {
        let view = self.extrapolate_view(lag_s, check_is_solid);

        self.perspective.get_matrix() * view.get_matrix()
    }
}

#[derive(Debug, Clone, Copy)]
struct CollisionResult {
    neg_x_collision: bool,
    pos_x_collision: bool,
    neg_y_collision: bool,
    pos_y_collision: bool,
    neg_z_collision: bool,
    pos_z_collision: bool,
}

fn resolve_collisions(
    mut physics_state: PlayerPhysicsState,
    translation: Vec3,
    check_is_solid: &impl Fn(IVec3) -> bool,
) -> PlayerPhysicsState {
    if translation.abs().max_element() >= 1.0 {
        eprintln!("Too large translation on at least one axis");
    }

    let mut neg_x_collision = false;
    let mut pos_x_collision = false;
    let mut neg_y_collision = false;
    let mut pos_y_collision = false;
    let mut neg_z_collision = false;
    let mut pos_z_collision = false;

    let translation_components = if physics_state.collisions.pos_y_collision
        && physics_state.velocity.z < 0.0
        || physics_state.collisions.neg_y_collision && physics_state.velocity.z > 0.0
    {
        // If the player is walking in +z/-z direction and already collided with a block in the same direction during the last tick, then apply the x translation first.
        // This way, a player can walk around corners where the corner block is missing.
        // Considering the following xz projection, the player can walk from block a to b while `#` is a solid block and the space represents air.
        // # | b
        // — + —
        // a |
        [
            vec3(translation.x, 0.0, 0.0),
            vec3(0.0, 0.0, translation.z),
            vec3(0.0, translation.y, 0.0),
        ]
    } else {
        [
            vec3(0.0, 0.0, translation.z),
            vec3(translation.x, 0.0, 0.0),
            vec3(0.0, translation.y, 0.0),
        ]
    };

    // Move by y first so the player cannot slide around a corner where the floor block is missing
    'axes: for translation in translation_components {
        physics_state.eye += translation;
        physics_state.aabb.min += translation;
        physics_state.aabb.max += translation;

        let Aabb3I { min, max } = physics_state.aabb.to_ivec_aabb();

        // Use non-inclusive upper bounds because the upper bound is the lower bound for the block aabb
        for x in min.x..max.x {
            for z in min.z..max.z {
                for y in min.y..max.y {
                    if check_is_solid(ivec3(x, y, z)) {
                        // Check if player intersects with the block
                        if !physics_state.aabb.to_ivec_aabb().intersects(Aabb3I {
                            min: ivec3(x, y, z),
                            max: ivec3(x + 1, y + 1, z + 1),
                        }) {
                            continue;
                        }
                        if translation.x < 0.0 {
                            physics_state.eye.x = (x + 1) as f32 + HALF_HITBOX_WIDTH;
                            physics_state.aabb.min.x = (x + 1) as f32;
                            physics_state.aabb.max.x = (x + 1) as f32 + HITBOX_WIDTH;
                            physics_state.velocity.x = 0.0;
                            neg_x_collision = true;
                        } else if translation.y < 0.0 {
                            physics_state.eye.y = (y + 1) as f32 + EYE_HEIGHT;
                            physics_state.aabb.min.y = (y + 1) as f32;
                            physics_state.aabb.max.y = (y + 1) as f32 + HITBOX_HEIGHT;
                            physics_state.velocity.y = 0.0;
                            neg_y_collision = true;
                        } else if translation.z < 0.0 {
                            physics_state.eye.z = (z + 1) as f32 + HALF_HITBOX_WIDTH;
                            physics_state.aabb.min.z = (z + 1) as f32;
                            physics_state.aabb.max.z = (z + 1) as f32 + HITBOX_WIDTH;
                            physics_state.velocity.z = 0.0;
                            neg_z_collision = true;
                        } else if translation.x > 0.0 {
                            physics_state.eye.x = x as f32 - HALF_HITBOX_WIDTH;
                            physics_state.aabb.min.x = x as f32 - HITBOX_WIDTH;
                            physics_state.aabb.max.x = x as f32;
                            physics_state.velocity.x = 0.0;
                            pos_x_collision = true;
                        } else if translation.y > 0.0 {
                            physics_state.eye.y = y as f32 - (HITBOX_HEIGHT - EYE_HEIGHT);
                            physics_state.aabb.min.y = y as f32 - HITBOX_HEIGHT;
                            physics_state.aabb.max.y = y as f32;
                            physics_state.velocity.y = 0.0;
                            pos_y_collision = true;
                        } else if translation.z > 0.0 {
                            physics_state.eye.z = z as f32 - HALF_HITBOX_WIDTH;
                            physics_state.aabb.min.z = z as f32 - HITBOX_WIDTH;
                            physics_state.aabb.max.z = z as f32;
                            physics_state.velocity.z = 0.0;
                            pos_z_collision = true;
                        }
                        continue 'axes;
                    }
                }
            }
        }
    }

    physics_state.collisions = CollisionResult {
        neg_x_collision,
        pos_x_collision,
        neg_y_collision,
        pos_y_collision,
        neg_z_collision,
        pos_z_collision,
    };
    physics_state
}
