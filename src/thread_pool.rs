use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub static THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();

#[cfg(target_arch = "wasm32")]
pub fn new_wasm_worker(closure: Box<dyn FnOnce() + Send>) -> web_sys::Worker {
    use web_sys::{Worker, WorkerOptions, WorkerType};

    let options = WorkerOptions::new();
    options.set_type(WorkerType::Module);
    let worker = Worker::new_with_options("./dist/worker.js", &options).unwrap();

    // Box<Box<dyn FnOnce()>> because *mut Box<...> is a usize-sized pointer whereass *mut dyn FnOnce() is a fat pointer, complicating this a bit
    let ptr = Box::into_raw(Box::new(Box::new(closure) as Box<dyn FnOnce()>));

    // Share this module, memory instance and pointer to entry point closure to newly created worker
    let array = js_sys::Array::new();
    array.push(&wasm_bindgen::module());
    array.push(&wasm_bindgen::memory());
    array.push(&JsValue::from(ptr as usize));
    worker.post_message(&array).unwrap();

    worker
}

#[cfg(target_arch = "wasm32")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn launch_worker(ptr: usize) {
    let closure = unsafe { Box::from_raw(ptr as *mut Box<dyn FnOnce()>) };
    closure();
}

#[cfg(target_arch = "wasm32")]
fn create_thread_pool(num_threads: usize) -> Result<ThreadPool, ThreadPoolBuildError> {
    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .spawn_handler(|thread| {
            new_wasm_worker(Box::new(|| thread.run()));

            Ok(())
        })
        .build()
}

#[cfg(not(target_arch = "wasm32"))]
fn create_thread_pool(num_threads: usize) -> Result<ThreadPool, ThreadPoolBuildError> {
    ThreadPoolBuilder::new().num_threads(num_threads).build()
}

pub fn init_thread_pool(num_threads: usize) -> Result<(), ThreadPoolBuildError> {
    // We manually set the thread pool here because setting the global thread pool fails in Wasm because it's waiting on the main thread.
    // TODO fix this, here it works: https://github.com/RReverser/wasm-bindgen-rayon
    let pool = create_thread_pool(num_threads)?;

    THREAD_POOL
        .set(pool)
        .expect("Assigning static thread pool failed");
    log::debug!("Thread pool initialized");

    Ok(())
}
