import init, * as wasm from "../pkg/mc.js";

// First message contains wasm module and shared memory for initialization, all subsequent messages contain jobs.
self.onmessage = async (message) => {
    let [mod, memory, ptr]: [WebAssembly.Module, WebAssembly.Memory, number] = message.data;
    await init({ module_or_path: mod, memory });

    wasm.launch_worker(ptr);
}
