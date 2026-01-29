use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, mpsc};

use crate::thread_pool;

pub type Job<T, Ctx> = Box<dyn FnOnce(Arc<Ctx>) -> T + Send>;

pub trait Executor<T: Send, Ctx> {
    fn dispatch(&mut self, jobs: Box<[Job<T, Ctx>]>, context: Arc<Ctx>);
    fn fetch(&mut self) -> impl Iterator<Item = T>;
}

pub struct ThreadPoolExecutor<T> {
    send_to_self: Sender<T>,
    recv_from_worker: Receiver<T>,
}

impl<T> ThreadPoolExecutor<T> {
    pub fn new() -> Self {
        let (send_to_self, recv_from_worker) = mpsc::channel();
        ThreadPoolExecutor {
            send_to_self,
            recv_from_worker,
        }
    }
}

impl<T: Send + 'static, Ctx: Send + Sync + 'static> Executor<T, Ctx> for ThreadPoolExecutor<T> {
    fn dispatch(&mut self, jobs: Box<[Job<T, Ctx>]>, context: Arc<Ctx>) {
        for job in jobs.into_iter() {
            let closure = {
                let send_to_self = self.send_to_self.clone();
                let context = context.clone();
                move || {
                    let result = job(context);
                    send_to_self.send(result).unwrap();
                }
            };

            thread_pool::THREAD_POOL
                .get()
                .expect("Thread pool should be initialized")
                .spawn_fifo(closure);
        }
    }

    fn fetch(&mut self) -> impl Iterator<Item = T> {
        self.recv_from_worker.try_iter()
    }
}
