/*
Loosely based on https://www.mcpk.wiki/wiki/Movement_Formulas.
All following acceleration/velocity constants are measured in m/tick and m/tick^2, where the duration of a tick is the reciproke of `TPS`

The velocity in the xz plane is controlled by the sequence `v(t) = v(t-1) * friction + acceleration`, the limit and maximum velocity is `acceleration/(1-friction)`.
The terminal velocity in free fall is the limit of the sequence `v(t) = (v(t-1) + gravity) * vertical_drag`, which is `vertical_drag * gravity / (1-vertical_drag)`.

When initiating a jump, the vertical velocity is set to `JUMP_VELOCITY`.
Also when sprinting during jumping (sprinting meaning the sprint key is pressed and velocity.xz() has a length greater than one),
the velocity is incremented by `SPRINT_JUMP_ACCEL` facing in the current acceleration direction once in the current movement direction projected to xz.
*/
use std::f32::consts::{FRAC_1_SQRT_2, PI};

use glam::{IVec3, Mat3, Vec3, ivec3, vec3};
use lazy_static::lazy_static;
use winit::keyboard::KeyCode;

use crate::{
    camera::{
        CardinalDirection, PerspectiveProj, View, YawPitch,
        block_ray_caster::{self, BlockHitInfo, LookedAtBlocks},
        direction_to_yaw_pitch, yaw_pitch_to_direction,
    },
    input::InputState,
    math::{Aabb3, Aabb3I},
    ui::{AddToGui, GuiModule},
    world::{LookupBlock, blocks::BlockPhysicsType},
};

/// TPS that is used for updating game physics.
/// Note that changing the TPS will have slight effects on the precise behaviour of movement.
/// Most notably, the jump height decreases and converges to 1.0 as the TPS increases. (With a TPS of 20, the jump height is 1.2522 as it is in Minecraft 1.9+)
pub const TPS: f32 = 40.0;

/// Movement constants taken directly from Minecraft (running at 20TPS)
mod mc_constants {
    pub(super) const NEGLIGIBLE_VELOCITY_THRESHOLD: f32 = 0.005;

    pub(super) const BASE_FRICTION_GROUND: f32 = 0.91 * 0.6;
    pub(super) const BASE_FRICTION_AIRBORNE: f32 = 0.91 * 1.0;

    pub(super) const BASE_ACCEL_GROUND: f32 = 0.1 * 1.0 * 0.98;
    pub(super) const BASE_ACCEL_AIRBORNE: f32 = 0.02 * 1.0;
    pub(super) const BASE_ACCEL_FLYING: f32 = 0.049;

    pub(super) const GRAVITY: f32 = -0.08;
    pub(super) const VERTICAL_DRAG: f32 = 0.98;

    pub(super) const JUMP_VELOCITY: f32 = 0.42;
    pub(super) const SPRINT_JUMP_ACCEL: f32 = 0.2;
}

lazy_static! {
    /// Minimum possible velocity value per axis on the xz plane. If actual velocity is less than this, it is set to zero. m/tick.
    static ref NEGLIGIBLE_VELOCITY_THRESHOLD: f32 = mc_constants::NEGLIGIBLE_VELOCITY_THRESHOLD * 20.0 / TPS;

    /// "Friction", that is the fraction of the x/z velocity conserved after each tick. Applied when walking/sprinting on the ground.
    static ref BASE_FRICTION_GROUND: f32 = mc_constants::BASE_FRICTION_GROUND.powf(20f32 / TPS);
    /// "Friction", that is the fraction of the x/z velocity conserved after each tick. Applied when jumping or falling.
    static ref BASE_FRICTION_AIRBORNE: f32 = mc_constants::BASE_FRICTION_AIRBORNE.powf(20f32 / TPS);
    // Vertical "drag", that is the fraction of the y velocity conserved after each tick.
    static ref VERTICAL_DRAG: f32 = mc_constants::VERTICAL_DRAG.powf(20f32 / TPS);

    /// Acceleration in the xz plane when walking/sprinting on the ground. m/tick^2.
    static ref BASE_ACCEL_GROUND: f32 = mc_constants::BASE_ACCEL_GROUND * 20f32 / TPS * (1.0 - *BASE_FRICTION_GROUND) / (1.0 - mc_constants::BASE_FRICTION_GROUND);
    /// Acceleration in the xz plane when jumping or falling. m/tick^2.
    static ref BASE_ACCEL_AIRBORNE: f32 = mc_constants::BASE_ACCEL_AIRBORNE * 20f32 / TPS * (1.0 - *BASE_FRICTION_AIRBORNE) / (1.0 - mc_constants::BASE_FRICTION_AIRBORNE);
    /// Acceleration in the xz plane when flying. m/tick^2.
    static ref BASE_ACCEL_FLYING: f32 = mc_constants::BASE_ACCEL_FLYING * 20f32 / TPS * (1.0 - *BASE_FRICTION_AIRBORNE) / (1.0 - mc_constants::BASE_FRICTION_AIRBORNE);

    /// Acceleration along the y axis during free fall. m/tick^2.
    static ref GRAVITY: f32 = mc_constants::GRAVITY * 20f32 / TPS * mc_constants::VERTICAL_DRAG / *VERTICAL_DRAG * (1.0 - *VERTICAL_DRAG)
                / (1.0 - mc_constants::VERTICAL_DRAG);

    /// The velocity along the y axis that initiates a jump. m/tick.
    static ref JUMP_VELOCITY: f32 = mc_constants::JUMP_VELOCITY * 20f32 / TPS;
    /// Acceleration in the xz plane in the current acceleration direction when jumping. m/tick^2.
    static ref SPRINT_JUMP_ACCEL: f32 = mc_constants::SPRINT_JUMP_ACCEL * 20f32 / TPS;

    /// Vertical velocity when flying. m/tick.
    static ref FLYING_Y_VELOCITY: f32 = 5.0 / TPS;
}

/// Multiplied by mouse dx and dy, then added or subtracted from [`PlayerState::yaw_norm`], [`PlayerState::pitch_norm`].
const CAMERA_SENSITIVITY: f32 = 0.002;

/// Hitbox height in metres.
const HITBOX_HEIGHT: f32 = 1.8;
/// Hitbox width and depth in metres.
const HITBOX_WIDTH: f32 = 0.6;
/// Half hitbox width and depth in metres.
const HALF_HITBOX_WIDTH: f32 = HITBOX_WIDTH / 2.0;
/// Hitbox eye height in metres.
const EYE_HEIGHT: f32 = 1.6;

/// Acceleration multiplier when sprinting while walking or airborne.
const SPRINTING_MULTIPLIER: f32 = 1.3;
/// Acceleration multiplier when "sprinting" while flying.
const SPRINTING_MULTIPLIER_FLYING: f32 = 2.0;

#[derive(Debug, Clone, Copy)]
enum MovementState {
    Walking,
    AirBorne,
    Flying { flying_up: bool, flying_down: bool },
}

#[derive(Debug)]
pub struct PlayerState {
    /// Camera perspective
    perspective: PerspectiveProj,
    /// yaw and pitch values
    yaw_pitch: YawPitch,
    /// Current movement state.
    movement_state: MovementState,
    /// Current physics related state.
    physics_state: PlayerPhysicsState,
    /// Blocks currently looked-at
    looked_at_blocks: LookedAtBlocks,
}

#[derive(Debug, Clone, Copy)]
struct PlayerPhysicsState {
    /// Coordinates of the players eye (horizontal center, height of EYE_HEIGHT)
    eye: Vec3,
    /// Player hitbox aabb. Used to work around floating point errors during collision detection
    aabb: Aabb3,
    /// Velocity in m/tick
    velocity: Vec3,
    /// Expected future acceleration in m/s^2.
    /// Only used to extrapolate the next player position.
    acceleration: Vec3,
    /// Collision info from last tick. Note that if the player didn't move in the xz plane during the previous tick, all collisions along the x and z axis are false.
    collisions: CollisionResult,
}

impl PlayerState {
    pub fn new(perspective: PerspectiveProj, eye: Vec3, direction: Vec3) -> Self {
        assert_ne!(direction, Vec3::Y);
        assert_ne!(direction, Vec3::NEG_Y);
        assert!(direction.is_normalized());
        Self {
            perspective,
            yaw_pitch: direction_to_yaw_pitch(direction),
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

                    Aabb3::new(min, max)
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
            looked_at_blocks: LookedAtBlocks::default(),
        }
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.perspective.aspect_ratio = aspect_ratio;
    }

    pub fn update_rotation(&mut self, input_state: &mut InputState) {
        let (dx, dy) = input_state.pull_mouse_movement();
        let mut new_yaw = self.yaw_pitch.yaw_norm - (dx as f32) * CAMERA_SENSITIVITY;
        let new_pitch = (self.yaw_pitch.pitch_norm - (dy as f32) * CAMERA_SENSITIVITY)
            .clamp(-0.5 + f32::EPSILON, 0.5 - f32::EPSILON);

        // Normalize yaw value
        new_yaw %= 2.0;
        if new_yaw < 0.0 {
            new_yaw += 2.0;
        }

        self.yaw_pitch.yaw_norm = new_yaw;
        self.yaw_pitch.pitch_norm = new_pitch;
    }

    pub fn update_position(
        &mut self,
        input_state: &mut InputState,
        _delta_s: f32,
        _time_s: f32,
        block_lookup: &impl LookupBlock,
    ) {
        if input_state.pull_key_double_clicked(KeyCode::Space) {
            match self.movement_state {
                MovementState::Walking | MovementState::AirBorne => {
                    self.movement_state = MovementState::Flying {
                        flying_up: false,
                        flying_down: false,
                    }
                }
                MovementState::Flying { .. } => self.movement_state = MovementState::AirBorne,
            }
        }

        let is_sprinting = input_state.is_pressed(KeyCode::ShiftLeft);

        let base_acceleration = match self.movement_state {
            MovementState::Walking => *BASE_ACCEL_GROUND,
            MovementState::AirBorne => *BASE_ACCEL_AIRBORNE,
            MovementState::Flying { .. } => *BASE_ACCEL_FLYING,
        } * match (self.movement_state, is_sprinting) {
            (_, false) => 1.0,
            (MovementState::Walking | MovementState::AirBorne, true) => SPRINTING_MULTIPLIER,
            (MovementState::Flying { .. }, true) => SPRINTING_MULTIPLIER_FLYING,
        };

        let base_friction = match self.movement_state {
            MovementState::Walking => *BASE_FRICTION_GROUND,
            MovementState::Flying { .. } | MovementState::AirBorne => *BASE_FRICTION_AIRBORNE,
        };

        // If we are going to move along both axes, use 1/sqrt(2) as coefficient so the maximum diagonal speed can't exceed the maximum straight speed
        let diagonal_correction = if input_state.is_pressed(KeyCode::KeyW)
            ^ input_state.is_pressed(KeyCode::KeyS)
            && input_state.is_pressed(KeyCode::KeyA) ^ input_state.is_pressed(KeyCode::KeyD)
        {
            FRAC_1_SQRT_2
        } else {
            1.0
        };

        // Acceleration rotated so that +x is forward and +z left. Note that this acceleration doesn't include friction/drag.
        let mut rotated_acceleration = Vec3::ZERO;

        if input_state.is_pressed(KeyCode::KeyW) {
            rotated_acceleration.x += diagonal_correction * base_acceleration;
        }

        if input_state.is_pressed(KeyCode::KeyS) {
            rotated_acceleration.x -= diagonal_correction * base_acceleration;
        }

        if input_state.is_pressed(KeyCode::KeyA) {
            rotated_acceleration.z += diagonal_correction * base_acceleration;
        }

        if input_state.is_pressed(KeyCode::KeyD) {
            rotated_acceleration.z -= diagonal_correction * base_acceleration;
        }

        // rotate acceleration so that it uses the world coordinate system
        // TODO why 2 - yaw?
        let mut world_acceleration =
            Mat3::from_rotation_y((2.0 - self.yaw_pitch.yaw_norm) * PI) * rotated_acceleration;
        let mut jump_initiated = false;

        match self.movement_state {
            MovementState::Walking => {
                // Do a jump if space is pressed, note that jumping is possible even if the player is not on the ground anymore during the current tick
                if input_state.is_single_clicked(KeyCode::Space) {
                    // sprint jump boost
                    if is_sprinting {
                        world_acceleration +=
                            world_acceleration.with_y(0.0).normalize_or_zero() * *SPRINT_JUMP_ACCEL;
                    }
                    assert!(
                        self.physics_state.velocity.y == 0.0,
                        "Jump started while vertical velocity is not zero"
                    );
                    world_acceleration.y = *JUMP_VELOCITY;
                    jump_initiated = true;
                    self.movement_state = MovementState::AirBorne;
                } else {
                    world_acceleration.y += *GRAVITY;
                }
            }
            MovementState::AirBorne => {
                world_acceleration.y += *GRAVITY;
            }
            MovementState::Flying {
                ref mut flying_up,
                ref mut flying_down,
            } => {
                *flying_up = input_state.is_single_clicked(KeyCode::Space);
                *flying_down = input_state.is_single_clicked(KeyCode::ControlLeft);

                // Directly set velocity, since acceleration isn't continuous
                if *flying_up == *flying_down {
                    self.physics_state.velocity.y = 0.0;
                } else if *flying_up {
                    self.physics_state.velocity.y = *FLYING_Y_VELOCITY;
                } else {
                    self.physics_state.velocity.y = -*FLYING_Y_VELOCITY;
                }
            }
        }

        let velocity_before_accel = self.physics_state.velocity;
        self.physics_state.velocity += world_acceleration;
        // Drag along y axis is applied before sampling the velocity, drag along x/z is sampled after according to mcpk.wiki
        if !jump_initiated {
            self.physics_state.velocity.y *= *VERTICAL_DRAG;
        }

        self.physics_state = resolve_collisions(
            self.physics_state,
            self.physics_state.velocity,
            block_lookup,
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
            && !pos_y_collision
        {
            self.movement_state = MovementState::AirBorne;
        }
        if let MovementState::AirBorne = self.movement_state
            && pos_y_collision
        {
            self.movement_state = MovementState::Walking;
        }

        self.physics_state.acceleration = self.physics_state.velocity - velocity_before_accel;

        if neg_x_collision || pos_x_collision {
            self.physics_state.acceleration.x = 0.0;
        }
        if neg_y_collision || pos_y_collision {
            self.physics_state.acceleration.y = 0.0;
        }
        if neg_z_collision || pos_z_collision {
            self.physics_state.acceleration.z = 0.0;
        }

        if self.physics_state.velocity.x.abs() < *NEGLIGIBLE_VELOCITY_THRESHOLD {
            self.physics_state.velocity.x = 0.0;
        }
        if self.physics_state.velocity.y.abs() < *NEGLIGIBLE_VELOCITY_THRESHOLD {
            self.physics_state.velocity.y = 0.0;
        }
        if self.physics_state.velocity.z.abs() < *NEGLIGIBLE_VELOCITY_THRESHOLD {
            self.physics_state.velocity.z = 0.0;
        }
    }

    pub fn update_looked_at_blocks(&mut self, block_lookup: &impl LookupBlock) {
        self.looked_at_blocks =
            block_ray_caster::find_looked_at_blocks(self.eye(), self.direction(), block_lookup);
    }

    pub fn eye(&self) -> Vec3 {
        self.physics_state.eye
    }

    pub fn direction(&self) -> Vec3 {
        yaw_pitch_to_direction(self.yaw_pitch)
    }

    pub fn cardinal_direction(&self) -> CardinalDirection {
        CardinalDirection::from_yaw(self.yaw_pitch.yaw_norm)
    }

    pub fn view(&self) -> View {
        let direction = self.direction();
        debug_assert!(direction.is_normalized());
        View::new(
            self.physics_state.eye,
            direction,
            // Find the vector in the plane between `direction` and Y that is perpendicular to `direction`
            direction.cross(Vec3::Y).normalize().cross(direction),
        )
    }

    pub fn perspective(&self) -> PerspectiveProj {
        self.perspective
    }

    pub fn extrapolate_view(&self, lag_s: f32, block_lookup: &impl LookupBlock) -> View {
        let mut view = self.view();
        let lag_ticks = lag_s / TPS.recip();
        let extrapolated_state = resolve_collisions(
            self.physics_state,
            self.physics_state.velocity * lag_ticks
                + 0.5 * self.physics_state.acceleration * lag_ticks.powi(2),
            block_lookup,
        );

        view.eye = extrapolated_state.eye;
        view
    }

    pub fn looked_at_blocks(&self) -> LookedAtBlocks {
        self.looked_at_blocks
    }

    /// Returns true if the block intersects the player AABB and the block is solid
    pub fn intersects_block(&self, coords: IVec3) -> bool {
        intersects_block(self.physics_state.aabb.to_ivec_aabb(), coords)
    }
}

#[derive(Debug, Clone, Copy)]
struct CollisionResult {
    /// Player collides with block face facing in negative x direction.
    neg_x_collision: bool,
    /// Player collides with block face facing in positive x direction.
    pos_x_collision: bool,
    /// Player collides with block face facing in negative y direction.
    neg_y_collision: bool,
    /// Player collides with block face facing in positive y direction.
    pos_y_collision: bool,
    /// Player collides with block face facing in negative z direction.
    neg_z_collision: bool,
    /// Player collides with block face facing in positive z direction.
    pos_z_collision: bool,
}

fn intersects_block(aabb: Aabb3I, coords: IVec3) -> bool {
    aabb.intersects(Aabb3I::new(coords, coords + IVec3::ONE))
}

fn resolve_collisions(
    mut physics_state: PlayerPhysicsState,
    translation: Vec3,
    block_lookup: &impl LookupBlock,
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

    let translation_components =
        if physics_state.collisions.neg_z_collision || physics_state.collisions.pos_z_collision {
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

        let Aabb3I { min, max, .. } = physics_state.aabb.to_ivec_aabb();

        // Use non-inclusive upper bounds because the upper bound is the lower bound for the block aabb
        for x in min.x..max.x {
            for z in min.z..max.z {
                for y in min.y..max.y {
                    if block_lookup.is_solid(ivec3(x, y, z)) {
                        // Check if player intersects with the block
                        if let Some(block) = block_lookup.lookup_block(ivec3(x, y, z))
                            && block.physics_type() == BlockPhysicsType::Solid
                            && !intersects_block(physics_state.aabb.to_ivec_aabb(), ivec3(x, y, z))
                        {
                            continue;
                        }
                        if translation.x < 0.0 {
                            physics_state.eye.x = (x + 1) as f32 + HALF_HITBOX_WIDTH;
                            physics_state.aabb.min.x = (x + 1) as f32;
                            physics_state.aabb.max.x = (x + 1) as f32 + HITBOX_WIDTH;
                            physics_state.velocity.x = 0.0;
                            pos_x_collision = true;
                        } else if translation.y < 0.0 {
                            physics_state.eye.y = (y + 1) as f32 + EYE_HEIGHT;
                            physics_state.aabb.min.y = (y + 1) as f32;
                            physics_state.aabb.max.y = (y + 1) as f32 + HITBOX_HEIGHT;
                            physics_state.velocity.y = 0.0;
                            pos_y_collision = true;
                        } else if translation.z < 0.0 {
                            physics_state.eye.z = (z + 1) as f32 + HALF_HITBOX_WIDTH;
                            physics_state.aabb.min.z = (z + 1) as f32;
                            physics_state.aabb.max.z = (z + 1) as f32 + HITBOX_WIDTH;
                            physics_state.velocity.z = 0.0;
                            pos_z_collision = true;
                        } else if translation.x > 0.0 {
                            physics_state.eye.x = x as f32 - HALF_HITBOX_WIDTH;
                            physics_state.aabb.min.x = x as f32 - HITBOX_WIDTH;
                            physics_state.aabb.max.x = x as f32;
                            physics_state.velocity.x = 0.0;
                            neg_x_collision = true;
                        } else if translation.y > 0.0 {
                            physics_state.eye.y = y as f32 - (HITBOX_HEIGHT - EYE_HEIGHT);
                            physics_state.aabb.min.y = y as f32 - HITBOX_HEIGHT;
                            physics_state.aabb.max.y = y as f32;
                            physics_state.velocity.y = 0.0;
                            neg_y_collision = true;
                        } else if translation.z > 0.0 {
                            physics_state.eye.z = z as f32 - HALF_HITBOX_WIDTH;
                            physics_state.aabb.min.z = z as f32 - HITBOX_WIDTH;
                            physics_state.aabb.max.z = z as f32;
                            physics_state.velocity.z = 0.0;
                            neg_z_collision = true;
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

impl AddToGui for BlockHitInfo {
    fn add_to_ui(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.monospace(format!("{:?}", self.block));
            ui.label("at");
            ui.monospace(format!("{:?}", self.coords));
        });
    }
}

impl GuiModule for PlayerState {
    fn title(&self) -> Option<&str> {
        Some("Player state")
    }

    fn add_contents(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let Vec3 { x, y, z } = self.eye();
            ui.label("eye:");
            ui.monospace(format!("{x:+.2} {y:+.2} {z:+.2}"));
        });
        ui.horizontal(|ui| {
            let Vec3 { x, y, z } = self.direction();
            ui.label("direction:");
            ui.monospace(format!("{x:+.2} {y:+.2} {z:+.2}"));
        });
        ui.horizontal(|ui| {
            ui.label("focused block:");
            match self.looked_at_blocks().solid_block {
                Some(info) => {
                    info.add_to_ui(ui);
                }
                None => {
                    ui.monospace("None");
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("focused liquid:");
            match self.looked_at_blocks().liquid_block {
                Some(info) => {
                    info.add_to_ui(ui);
                }
                None => {
                    ui.monospace("None");
                }
            }
        });
    }
}
