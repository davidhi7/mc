use std::{cell::RefCell, collections::VecDeque, rc::Rc, time::Duration};
use web_time::Instant;

use crate::ui::{CreateGuiModule, GuiModule};

#[derive(Clone)]
pub struct FrameTimeMetrics {
    inner: Rc<RefCell<FrameTimeMetricsInner>>,
}

struct FrameTimeMetricsInner {
    deque: VecDeque<Duration>,
    sampling_interval_ms: u128,
    last_sample_instant: Instant,
    last_sample_frametime_ms: f64,
}

impl FrameTimeMetrics {
    pub fn new(sampling_interval_ms: u128) -> Self {
        let inner = FrameTimeMetricsInner {
            deque: VecDeque::new(),
            sampling_interval_ms,
            last_sample_instant: Instant::now(),
            last_sample_frametime_ms: 0.0,
        };
        FrameTimeMetrics {
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    pub fn push(&self, frametime: Duration) {
        self.inner.borrow_mut().deque.push_back(frametime);
    }

    pub fn maybe_update_sample(&mut self) {
        let mut inner = self.inner.borrow_mut();

        let now = Instant::now();
        if now.duration_since(inner.last_sample_instant).as_millis() >= inner.sampling_interval_ms {
            let frametime_sample_us = inner
                .deque
                .iter()
                .map(|duration: &Duration| duration.as_micros())
                .sum::<u128>()
                / inner.deque.len() as u128;
            inner.last_sample_frametime_ms = frametime_sample_us as f64 / 1000f64;
            inner.deque.clear();
            inner.last_sample_instant = now;
        }
    }
}

impl CreateGuiModule for FrameTimeMetrics {
    fn create_ui_module(&self) -> GuiModule {
        let clone = self.clone();
        GuiModule {
            title: "Frame times".to_string(),
            add_contents: Box::new(move |ui| {
                let frametime = clone.inner.borrow().last_sample_frametime_ms;
                ui.horizontal(|ui| {
                    ui.label("Frame times:");
                    ui.monospace(format!("{:.2} ms", frametime));
                });
            }),
        }
    }
}
