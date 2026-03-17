struct Globals {
    view_proj: mat4x4f,
    light_view_projs: array<mat4x4f, 4>,
    light_direction: vec3f,
};

struct Vertex {
    position: vec3f,
    tex_coordinates: vec2f,
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

@group(3) @binding(0)
var shadow_map: texture_depth_2d_array;

@group(3) @binding(1)
var shadow_map_sampler: sampler_comparison;

struct InstanceInput {
    @location(0) attributes: u32,
    @location(1) ao_attributes: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) tex_coordinates: vec2f,
    @location(1) @interpolate(flat) tex_index: u32,
    @location(2) @interpolate(flat) direction: u32,
    @location(3) ao_intensity: f32,
    @location(4) distance: f32,
    @location(5) light_position: vec3f,
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
    // Since we use orthographic projections, perspective dividde is not needed
    // x,y are in [-1, 1], z in [0, 1]
    let light_ndc = globals.light_view_projs[0] * vec4f(global_position, 1);
    // x,y are now in [0, 1], z unchanged
    // Negating the y component is neccessary because in NDC, x=0, y=0 is in the bottom left,
    // but in texture coordinates, x=0, y=0 is in the top left
    out.light_position = vec3(light_ndc.xy * vec2(0.5, -0.5) + vec2(0.5), light_ndc.z);
    return out;
}

fn shadow(light_position: vec3f, normal: vec3f) -> f32 {
    // let bias = max(0.0005 * (1.0 - dot(normal, globals.light_direction)), 0.00005);
    // let bias = 0.05;
    let bias = 0.0;
    // `textureSampleCompare` with `shadow_map_sampler` returns 1.0 if `light_position.z` is less than all sampled values, or a number between 0 and 1 if the provided depth is greater than some or all samples
    let shadow = textureSampleCompare(shadow_map, shadow_map_sampler, light_position.xy, 1, light_position.z - bias);
    return shadow;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var normal: vec3f;
    if in.direction == 0 {
        normal = vec3(-1, 0, 0);
        // return vec4(1, 0, 0, 1);
    } else if in.direction == 1 {
        normal = vec3(1, 0, 0);
    } else if in.direction == 2 {
        normal = vec3(0, -1, 0);
        // return vec4(0, 1, 0, 1);
    } else if in.direction == 3 {
        normal = vec3(0, 1, 0);
    } else if in.direction == 4 {
        normal = vec3(0, 0, -1);
        // return vec4(0, 0, 1, 1);
    } else {
        normal = vec3(0, 0, 1);
    }
    // return vec4(vec3(1) * max(0, dot(normal, -globals.light_direction)), 1);
    var lighting_factor = (1.0 - in.ao_intensity * 0.3) * mix(0.5, 1.0, shadow(in.light_position, normal));
    // if (shadow(in.light_position) == 0.0) {
    //     lighting_factor = lighting_factor * 0.2;
    // }
    // // let lighting_factor =

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
