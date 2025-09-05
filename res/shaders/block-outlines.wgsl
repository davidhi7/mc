struct Globals {
    view_proj: mat4x4f,
};

@group(0) @binding(0)
var<uniform> globals: Globals;

struct VertexInput {
    @location(0) position: vec3f,
};

@vertex
fn vs_main(vertex: VertexInput) -> @builtin(position) vec4f {
    return globals.view_proj * vec4f(vertex.position, 1.0);
}

@fragment
fn fs_main(@builtin(position) in: vec4f) -> @location(0) vec4<f32> {
    return vec4f(0.0, 0.0, 0.0, 1.0);
}
