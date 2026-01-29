use std::{num::NonZero, thread};

use mc::launch;

fn main() {
    launch(
        thread::available_parallelism()
            .unwrap_or(NonZero::new(2).unwrap())
            .get(),
    );
}
