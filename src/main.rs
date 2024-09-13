mod camera;
mod texture;
mod window;
mod world;

fn main() {
    env_logger::init();
    pollster::block_on(window::run());
}
