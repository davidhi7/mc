use std::fmt::Display;

pub fn setup_logger() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Debug).expect("Couldn't initialize console_log");
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReadableBytes(pub u64);

impl Display for ReadableBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];

        if self.0 < 1024 {
            return write!(f, "{} B", self.0);
        }

        let mut value = self.0 as f64;
        let mut unit = 0;

        while value >= 1024.0 && unit < UNITS.len() - 1 {
            value /= 1024.0;
            unit += 1;
        }

        write!(f, "{:.2} {}", value, UNITS[unit])
    }
}
