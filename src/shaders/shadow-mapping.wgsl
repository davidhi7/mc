struct Globals {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
};

struct Vertex {
    position: vec3<f32>,
    tex_coordinates: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> globals: Globals;

@group(1) @binding(0)
var<uniform> vertices: array<Vertex, 48>;

@group(1) @binding(1)
var<storage> chunks: array<vec3i>;

// proposal for texture binding arrays in WebGPU: https://github.com/gpuweb/gpuweb/blob/main/proposals/sized-binding-arrays.md
// Already present in WGPU and desktop graphics apis
// var textures: binding_array<texture_2d<f32>>;
@group(2) @binding(0)
var textures: texture_2d_array<f32>;

@group(2) @binding(1)
var texture_sampler: sampler;

struct InstanceInput {
    @location(0) attributes: u32,
    @location(1) ao_attributes: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coordinates: vec2<f32>,
    @location(1) @interpolate(flat) tex_index: u32,
};

@vertex
fn vs_main(
    instance: InstanceInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    let chunk_index = vertex_index >> 2;
    let real_vertex_index = vertex_index % 4;

    let chunk_relative_coords = vec3i(
        i32((instance.attributes >>  0) & 0x1F),
        i32((instance.attributes >>  5) & 0x1F),
        i32((instance.attributes >> 10) & 0x1F),
    );

    let tex_index = (instance.attributes >> 15) & 0xFF;
    let direction = (instance.attributes >> 23) & 0x7;

    var quad_index = 2u * direction;

    let vertex = vertices[quad_index * 4 + real_vertex_index];
    let global_position = vec3f(32 * chunks[chunk_index] + chunk_relative_coords) + vertex.position;

    var out: VertexOutput;
    out.clip_position = globals.light_view_proj * vec4f(global_position, 1);
    out.tex_coordinates = vertex.tex_coordinates;
    out.tex_index = tex_index;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var frag_color = textureSample(textures, texture_sampler, in.tex_coordinates, in.tex_index);

    if frag_color.w < 0.1 {
        discard;
    }

    let depth = pow(in.clip_position.z, 10);

    // return vec4(depth * vec3(1), 1);
    return vec4(1.0, 0.0, 0.0, 1.0);
}
