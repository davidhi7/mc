mod texture;
mod window;
mod world;
mod renderer;

fn main() {
    env_logger::init();
    pollster::block_on(window::run());
}
