use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use winit::{
    event::{ElementState, KeyEvent, MouseButton},
    keyboard::{KeyCode, PhysicalKey},
};

const DOUBLE_CLICK_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Button {
    Mouse(MouseButton),
    Keyboard(KeyCode),
}

impl From<MouseButton> for Button {
    fn from(value: MouseButton) -> Self {
        Button::Mouse(value)
    }
}
impl From<KeyCode> for Button {
    fn from(value: KeyCode) -> Self {
        Button::Keyboard(value)
    }
}

#[derive(Default)]
struct KeyInfo {
    pressed: bool,
    double_clicked: bool,
    last_key_down: Option<Instant>,
}

impl KeyInfo {
    pub fn handle_state_change(&mut self, state: ElementState) {
        if state.is_pressed() {
            self.pressed = true;
            self.double_clicked = self.last_key_down.is_some_and(|last_key_down| {
                Instant::now().duration_since(last_key_down) <= DOUBLE_CLICK_INTERVAL
            });
            self.last_key_down = Some(Instant::now());
        } else {
            self.pressed = false;
            self.double_clicked = false;
        }
    }
}

#[derive(Default)]
pub struct InputState {
    keys: HashMap<Button, KeyInfo>,
    mouse_movement: (f64, f64),
}

impl InputState {
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

        self.keys
            .entry(Button::Keyboard(key_code))
            .or_default()
            .handle_state_change(state);
    }

    pub fn handle_mouse_event(&mut self, button: MouseButton, state: ElementState) {
        self.keys
            .entry(Button::Mouse(button))
            .or_default()
            .handle_state_change(state);
    }

    pub fn increment_mouse_movement(&mut self, mouse_movement: (f64, f64)) {
        self.mouse_movement.0 += mouse_movement.0;
        self.mouse_movement.1 += mouse_movement.1;
    }

    pub fn pull_mouse_movement(&mut self) -> (f64, f64) {
        std::mem::take(&mut self.mouse_movement)
    }

    /// Returns whether the button is currently pressed.
    pub fn is_pressed(&self, key: impl Into<Button>) -> bool {
        self.keys.get(&key.into()).is_some_and(|info| info.pressed)
    }

    /// Returns whether the button is currently pressed. Also mark button as no longer pressed.
    pub fn pull_is_pressed(&mut self, key: impl Into<Button>) -> bool {
        match self.keys.get_mut(&key.into()) {
            Some(key_info) => {
                if key_info.pressed {
                    key_info.pressed = false;
                    key_info.double_clicked = false;
                    true
                } else {
                    false
                }
            }
            None => false,
        }
    }

    /// Returns whether the button has been double clicked. Also reset double clicked state.
    pub fn pull_key_double_clicked(&mut self, key: impl Into<Button>) -> bool {
        match self.keys.get_mut(&key.into()) {
            Some(key_info) => {
                if key_info.double_clicked {
                    key_info.double_clicked = false;
                    true
                } else {
                    false
                }
            }
            None => false,
        }
    }

    /// Returns true if the button is currently pressed but hasn't been double clicked.
    pub fn is_single_clicked(&self, key: impl Into<Button>) -> bool {
        self.keys
            .get(&key.into())
            .is_some_and(|info| info.pressed && !info.double_clicked)
    }
}
