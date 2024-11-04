const line_length: f32 = 0.03;

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec3<f32>
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32
) -> VertexOutput {
    // Remove translation component of camera, so reticle is at z=0
    let modified_camera = mat4x4f(camera.view_proj[0], camera.view_proj[1], camera.view_proj[2], vec4f(0.0, 0.0, 0.0, 1.0));

    // If vertex_index is even, this is 0, otherwise 1. Relevant because all even vertices are at (0.0, 0.0, 0.0)
    let uneven_vertex = (vertex_index + 1) & 1;
    // Create unit vector that, depending on the vertex index, has one of x/y/z components set to 1, the other to 0
    // Used as color of the line and vertex clip position after multiplication with uneven_vertex
    let axis = vec3f(vec3u(select(0u, 1u, vertex_index <= 1), (vertex_index >> 1) & 1, (vertex_index >> 2) & 1));

    var out: VertexOutput;
    out.clip_position = modified_camera * vec4f(line_length * f32(uneven_vertex) * axis, 1.0) + vec4f(0.0, 0.0, 0.5, 0.0);
    out.color = axis;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4f(in.color, 1.0);
}
