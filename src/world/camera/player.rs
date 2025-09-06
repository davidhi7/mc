use std::{collections::HashSet, f32::consts::PI};

use glam::{ivec3, vec3, IVec3, Vec3};
use winit::keyboard::KeyCode;

use crate::{
    math::{Aabb3, Aabb3I},
    world::camera::CameraController,
};

const HITBOX_HEIGHT: f32 = 1.8;
const EYE_HEIGHT: f32 = 1.6;
const HITBOX_WIDTH: f32 = 0.6;
const HALF_HITBOX_WIDTH: f32 = HITBOX_WIDTH / 2.0;

pub struct Player {
    pub camera: CameraController,
    /// Camera translation per second
    speed: f32,
    /// Camera rotation per mouse movement step, multiplied by pi
    sensitivity: f32,
    /// Player aabb. Stored separatly from the camera eye coordinates for floating point precission during collision detection
    aabb: Aabb3,
}

impl Player {
    pub fn new(camera: CameraController, speed: f32, sensitivity: f32) -> Self {
        let aabb = Player::aabb_from_camera(&camera);

        Self {
            camera,
            speed,
            sensitivity,
            aabb,
        }
    }

    fn aabb_from_camera(camera: &CameraController) -> Aabb3 {
        let min = camera.view.eye - vec3(HALF_HITBOX_WIDTH, EYE_HEIGHT, HALF_HITBOX_WIDTH);
        let max = camera.view.eye
            + vec3(
                HALF_HITBOX_WIDTH,
                HITBOX_HEIGHT - EYE_HEIGHT,
                HALF_HITBOX_WIDTH,
            );

        Aabb3 { min, max }
    }

    pub fn handle_input(
        &mut self,
        pressed_keys: &HashSet<KeyCode>,
        mouse_movement: (f64, f64),
        delta_s: f32,
        check_is_solid: impl Fn(IVec3) -> bool,
    ) {
        let (dx, dy) = mouse_movement;

        let time_adjusted_speed = self.speed * delta_s;

        let mut new_yaw = self.camera.yaw - (dx as f32) * self.sensitivity;
        let new_pitch = (self.camera.pitch - (dy as f32) * self.sensitivity).clamp(-0.5, 0.5);

        // Normalize yaw value
        new_yaw %= 2.0;
        if new_yaw < 0.0 {
            new_yaw += 2.0;
        }

        self.camera.yaw = new_yaw;
        self.camera.pitch = new_pitch;

        let (yaw_sin, yaw_cos) = (new_yaw * PI).sin_cos();
        let (pitch_sin, pitch_cos) = (new_pitch * PI).sin_cos();

        // View direction projected to xz plane
        let xz_forward = vec3(yaw_cos, 0.0, yaw_sin);

        self.camera.view.direction = vec3(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin);
        // Effectively rotate xz_forward by 90 deg around y axis
        self.camera.view.right = vec3(yaw_sin, 0.0, -yaw_cos);
        self.camera.view.up = self.camera.view.direction.cross(self.camera.view.right);

        debug_assert!(xz_forward.is_normalized());
        debug_assert!(self.camera.view.direction.is_normalized());
        debug_assert!(self.camera.view.right.is_normalized());
        debug_assert!(self.camera.view.up.is_normalized());

        let mut position_translation = Vec3::ZERO;

        if pressed_keys.contains(&KeyCode::KeyW) {
            position_translation += xz_forward * time_adjusted_speed;
        }

        if pressed_keys.contains(&KeyCode::KeyS) {
            position_translation -= xz_forward * time_adjusted_speed;
        }

        if pressed_keys.contains(&KeyCode::KeyA) {
            position_translation -= self.camera.view.right * time_adjusted_speed;
        }

        if pressed_keys.contains(&KeyCode::KeyD) {
            position_translation += self.camera.view.right * time_adjusted_speed;
        }

        if pressed_keys.contains(&KeyCode::ControlLeft) {
            position_translation.y -= time_adjusted_speed;
        }

        if pressed_keys.contains(&KeyCode::Space) {
            position_translation.y += time_adjusted_speed;
        }

        if pressed_keys.contains(&KeyCode::ShiftLeft) {
            position_translation *= 3.0;
        }

        if position_translation.length() == 0.0 {
            return;
        }

        resolve_collisions(self, position_translation, check_is_solid);
    }
}

fn resolve_collisions(player: &mut Player, movement: Vec3, check_is_solid: impl Fn(IVec3) -> bool) {
    'axis: for axis in [
        vec3(movement.x, 0.0, 0.0),
        vec3(0.0, movement.y, 0.0),
        vec3(0.0, 0.0, movement.z),
    ] {
        player.camera.view.eye += axis;
        player.aabb.min += axis;
        player.aabb.max += axis;

        let Aabb3I { min, max } = player.aabb.to_ivec_aabb();

        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    if check_is_solid(ivec3(x, y, z)) {
                        // Check if player intersects with the block
                        if !player.aabb.to_ivec_aabb().intersects(Aabb3I {
                            min: ivec3(x, y, z),
                            max: ivec3(x + 1, y + 1, z + 1),
                        }) {
                            continue;
                        }
                        if axis.x < 0.0 {
                            player.camera.view.eye.x = (x + 1) as f32 + HALF_HITBOX_WIDTH;
                            player.aabb.min.x = (x + 1) as f32;
                            player.aabb.max.x = (x + 1) as f32 + HITBOX_WIDTH;
                        } else if axis.y < 0.0 {
                            player.camera.view.eye.y = (y + 1) as f32 + EYE_HEIGHT;
                            player.aabb.min.y = (y + 1) as f32;
                            player.aabb.max.y = (y + 1) as f32 + HITBOX_HEIGHT;
                        } else if axis.z < 0.0 {
                            player.camera.view.eye.z = (z + 1) as f32 + HALF_HITBOX_WIDTH;
                            player.aabb.min.z = (z + 1) as f32;
                            player.aabb.max.z = (z + 1) as f32 + HITBOX_WIDTH;
                        } else if axis.x > 0.0 {
                            player.camera.view.eye.x = x as f32 - HALF_HITBOX_WIDTH;
                            player.aabb.min.x = x as f32 - HITBOX_WIDTH;
                            player.aabb.max.x = x as f32;
                        } else if axis.y > 0.0 {
                            player.camera.view.eye.y = y as f32 - (HITBOX_HEIGHT - EYE_HEIGHT);
                            player.aabb.min.y = y as f32 - HITBOX_HEIGHT;
                            player.aabb.max.y = y as f32;
                        } else if axis.z > 0.0 {
                            player.camera.view.eye.z = z as f32 - HALF_HITBOX_WIDTH;
                            player.aabb.min.z = z as f32 - HITBOX_WIDTH;
                            player.aabb.max.z = z as f32;
                        }
                        continue 'axis;
                    }
                }
            }
        }
    }
}
