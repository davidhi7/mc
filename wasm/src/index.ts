import init, * as wasm from "../pkg/mc.js";

init().then(() => {
    wasm.launch(navigator.hardwareConcurrency);
});
