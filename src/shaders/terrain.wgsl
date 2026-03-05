struct Globals {
    view_proj: mat4x4<f32>,
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
    @location(2) @interpolate(flat) direction: u32,
    @location(3) ao_intensity: f32,
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

    let ao_0 = instance.ao_attributes & 3;
    let ao_1 = (instance.ao_attributes >> 2) & 3;
    let ao_2 = (instance.ao_attributes >> 4) & 3;
    let ao_3 = (instance.ao_attributes >> 6) & 3;

    var vertex_ao_factor_index = real_vertex_index;
    var quad_index = 2u * direction;

    if (ao_0 + ao_3 < ao_1 + ao_2) {
        // Use the next quad that is flipped
        quad_index += 1u;

        // Map old to new AO attribute index
        // Specifically map 0 -> 1, 1 -> 3, 2 -> 0, 3 -> 2
        // maybte the following code is better? vertex_ao_factor_index = 0x8Du >> (2 * real_vertex_index) & 0x3u;
        var ao_index_permutation = array<u32, 4>(1, 3, 0, 2);
        vertex_ao_factor_index = ao_index_permutation[real_vertex_index];
    }

    let ao_intensity = (instance.ao_attributes >> (2 * vertex_ao_factor_index)) & 0x3;
    let vertex = vertices[quad_index * 4 + real_vertex_index];
    let global_position = vec3f(32 * chunks[chunk_index] + chunk_relative_coords) + vertex.position;

    var out: VertexOutput;
    out.clip_position = globals.view_proj * vec4f(global_position, 1);
    out.tex_coordinates = vertex.tex_coordinates;
    out.tex_index = tex_index;
    out.direction = direction;
    out.ao_intensity = f32(ao_intensity);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let lighting_factor = 1.0 - in.ao_intensity * 0.3;

    var frag_color = textureSample(textures, texture_sampler, in.tex_coordinates, in.tex_index);

    if frag_color.w < 0.1 {
        discard;
    }

    // Hack to render grayscale grass/leaves textures green
    var color_multiplier = vec3f(1.0, 1.0, 1.0);
    if in.tex_index == 1 {
        // Color for minecraft forest biome
        color_multiplier = vec3f(0.47, 0.75, 0.35);
    } else if in.tex_index == 11 {
        // Oak leaves, random color
        color_multiplier = vec3f(0.243, 0.569, 0.208);
    } else if in.tex_index == 12 {
        // Spruce leves, random color
        color_multiplier = vec3f(0.216, 0.439, 0.298);
    }
    // Convert sRGB to linear RGB color
    color_multiplier = pow(color_multiplier, vec3f(2.2));
    frag_color = frag_color * vec4f(color_multiplier, 1.0);

    return lighting_factor * frag_color;
}
