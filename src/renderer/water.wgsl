struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct Vertex {
    position: vec3<f32>,
    tex_coordinates: vec2<f32>,
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

@group(2) @binding(0)
var<uniform> vertices: array<Vertex, 48>;

@group(3) @binding(0)
var<uniform> chunk: vec3i;

struct InstanceInput {
    @location(0) attributes: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coordinates: vec2<f32>,
    @location(1) @interpolate(flat) tex_index: u32,
    @location(2) @interpolate(flat) direction: u32,
};

@vertex
fn vs_main(
    instance: InstanceInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    let chunk_relative_coords = vec3i(
        i32((instance.attributes >>  0) & 0x1F),
        i32((instance.attributes >>  5) & 0x1F),
        i32((instance.attributes >> 10) & 0x1F),
    );

    let tex_index = (instance.attributes >> 15) & 0xFF;
    let direction = (instance.attributes >> 23) & 0x7;

    let vertex = vertices[2 * direction * 4 + vertex_index];
    let global_position = vec3f(32 * chunk + chunk_relative_coords) + vertex.position;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4f(global_position, 1);
    out.tex_coordinates = vertex.tex_coordinates;
    out.tex_index = tex_index;
    out.direction = direction;
    return out;
}

@group(0) @binding(0)
var t_diffuse: binding_array<texture_2d<f32>>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4f(29.0 / 255.0, 63.0 / 255.0, 117.0 / 255.0, 0.8);
}
