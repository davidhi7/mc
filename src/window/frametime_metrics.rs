use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

pub struct FrameTimeMetrics {
    deque: VecDeque<Duration>,
    sampling_interval_ms: u128,
    last_sample_instant: Instant,
    pub last_sample_frametime_ms: f64,
}

impl FrameTimeMetrics {
    pub fn new(sampling_interval_ms: u128) -> Self {
        FrameTimeMetrics {
            deque: VecDeque::new(),
            sampling_interval_ms,
            last_sample_instant: Instant::now(),
            last_sample_frametime_ms: 0.0,
        }
    }

    pub fn push(&mut self, frametime: Duration) {
        self.deque.push_back(frametime);
    }

    pub fn update_sample(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_sample_instant).as_millis() >= self.sampling_interval_ms {
            let frametime_sample_us = self
                .deque
                .iter()
                .map(|duration: &Duration| duration.as_micros())
                .sum::<u128>()
                / self.deque.len() as u128;
            self.last_sample_frametime_ms = frametime_sample_us as f64 / 1000f64;
            self.deque.clear();
            self.last_sample_instant = now;
        }
    }
}
