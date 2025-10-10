use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use winit::{
    event::KeyEvent,
    keyboard::{KeyCode, PhysicalKey},
};

const DOUBLE_CLICK_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Default)]
struct KeyInfo {
    pressed: bool,
    double_clicked: bool,
    last_key_down: Option<Instant>,
}

#[derive(Default)]
pub struct InputState {
    pressed_keys: HashMap<KeyCode, KeyInfo>,
    mouse_movement: (f64, f64),
}

impl InputState {
    pub fn increment_mouse_movement(&mut self, mouse_movement: (f64, f64)) {
        self.mouse_movement.0 += mouse_movement.0;
        self.mouse_movement.1 += mouse_movement.1;
    }

    pub fn pull_mouse_movement(&mut self) -> (f64, f64) {
        std::mem::take(&mut self.mouse_movement)
    }

    pub fn handle_key_event(&mut self, key_event: KeyEvent) {
        let KeyEvent {
            physical_key: PhysicalKey::Code(key_code),
            repeat,
            state,
            ..
        } = key_event
        else {
            eprintln!("Received KeyEvent with unknown KeyCode: {:#?}", key_event);
            return;
        };

        if repeat {
            return;
        }

        let key_info = self.pressed_keys.entry(key_code).or_default();
        if state.is_pressed() {
            key_info.pressed = true;
            key_info.double_clicked = key_info.last_key_down.is_some_and(|last_key_down| {
                Instant::now().duration_since(last_key_down) <= DOUBLE_CLICK_INTERVAL
            });
            key_info.last_key_down = Some(Instant::now());
        } else {
            key_info.pressed = false;
            key_info.double_clicked = false;
        }
    }

    pub fn is_key_pressed(&self, key_code: KeyCode) -> bool {
        self.pressed_keys
            .get(&key_code)
            .is_some_and(|info| info.pressed)
    }

    pub fn pull_key_double_clicked(&mut self, key_code: KeyCode) -> bool {
        return match self.pressed_keys.get_mut(&key_code) {
            Some(key_info) => {
                if key_info.double_clicked {
                    key_info.double_clicked = false;
                    true
                } else {
                    false
                }
            }
            None => false,
        };
    }

    pub fn is_key_single_clicked(&self, key_code: KeyCode) -> bool {
        self.pressed_keys
            .get(&key_code)
            .is_some_and(|info| info.pressed && !info.double_clicked)
    }
}
