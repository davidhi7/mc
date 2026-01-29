use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    Request, Response,
    js_sys::{ArrayBuffer, Uint8Array},
    window,
};

/// Load bytes from the given path.
pub async fn fetch(path: impl AsRef<str>) -> Result<Vec<u8>, ()> {
    let array_buffer = try {
        let request = Request::new_with_str(path.as_ref())?;

        let window = window().unwrap();
        let response = JsFuture::from(window.fetch_with_request(&request))
            .await?
            .dyn_into::<Response>()?;

        let array_buffer = JsFuture::from(response.array_buffer()?)
            .await?
            .dyn_into::<ArrayBuffer>()?;

        array_buffer
    }
    .map_err(|_| ())?;

    Ok(Uint8Array::new(&array_buffer).to_vec())
}
