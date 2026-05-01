const NUM_CASCADES = 4u;
override SHADOW_MAP_RESOLUTION = 2048u;

struct Globals {
    view_proj: mat4x4f,
    light_view_projs: array<mat4x4f, NUM_CASCADES>,
    light_direction: vec3f,
    // frustum slices far distances in view space
    cascades_view_space_far: vec4f
};

struct Vertex {
    position: vec3f,
    tex_coordinates: vec2f,
};

struct ShadowSettings {
    bias_min: f32,
    bias_max: f32,
    pcf: u32,
}

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

@group(3) @binding(2)
var<uniform> shadow_settings: ShadowSettings;

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
    @location(4) depth_vs: f32,
    @location(5) position_ws: vec3f,
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
    // when using a perspective projection, clip_position.w is the view space depth
    out.depth_vs =  out.clip_position.w;
    out.position_ws = global_position;
    return out;
}

// Returns the layer in [0, NUM_CASCADES-1] for the given depth or NUM_CASCADES if the depth is beyond the shadow draw distance.
fn csm_layer(position_ws: vec3f, depth_vs: f32) -> u32 {
    var layer = NUM_CASCADES;
    for (var i = 0u; i < NUM_CASCADES; i++) {
        if depth_vs <= globals.cascades_view_space_far[i] {
            layer = i;
            break;
        }
    }
    return layer;
}

// Returns 1.0 if all samples are fully visible, or a value within [0.0, 1.0) if some samples are occluded.
fn shadow(position_ws: vec3f, depth_vs: f32, normal: vec3f) -> f32 {
    var layer = csm_layer(position_ws, depth_vs);
    if layer == NUM_CASCADES {
        return 1.0;
    }
    let light_ndc = globals.light_view_projs[layer] * vec4f(position_ws, 1.0);
    let light_uv = light_ndc.xy * vec2(0.5, -0.5) + vec2(0.5);

    // if light_ndc.z < 0.0
    //     || light_ndc.z > 1.0
    //     || light_uv.x < 0.0
    //     || light_uv.x > 1.0
    //     || light_uv.y < 0.0
    //     || light_uv.y > 1.0
    // {
    //     return 1.0;
    // }

    let bias = max(shadow_settings.bias_max * (1 - dot(normal, -globals.light_direction)), shadow_settings.bias_min);

    if shadow_settings.pcf > 0 {
        let uv_offset = 1.0 / vec2f(textureDimensions(shadow_map, layer));
        // textureSampleCompare with shadow_map_sampler returns 1.0 if light_ndc.z is less than all sampled values,
        // or a number between 0 and 1 if the provided depth is greater than some or all samples.
        // In other words, 1.0 represents a fully-lit fragment, 0.0 represents a pixel fully in the shadow of other fragments.
        var factor = 0.0;
        for (var i = -1; i <= 1; i++) {
            for (var j = -1; j <= 1; j++) {
                factor += textureSampleCompare(
                        shadow_map,
                        shadow_map_sampler,
                        light_uv + vec2f(f32(i), f32(j)) * uv_offset,
                        layer,
                        light_ndc.z - bias
                    );
            }
        }
        return factor / 9.0;
    } else {
        return textureSampleCompare(
            shadow_map,
            shadow_map_sampler,
            light_uv,
            layer,
            light_ndc.z - bias
        );
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(textures, texture_sampler, in.tex_coordinates, in.tex_index);

    if tex_color.w < 0.1 {
        discard;
    }

    var normal: vec3f;
    if in.direction == 0 {
        normal = vec3(-1, 0, 0);
    } else if in.direction == 1 {
        normal = vec3(1, 0, 0);
    } else if in.direction == 2 {
        normal = vec3(0, -1, 0);
    } else if in.direction == 3 {
        normal = vec3(0, 1, 0);
    } else if in.direction == 4 {
        normal = vec3(0, 0, -1);
    } else {
        normal = vec3(0, 0, 1);
    }

    let ambient = 0.4 * (1.0 - in.ao_intensity * 0.3);
    var diffuse = 0.6 * max(dot(normal, -globals.light_direction), 0.0);

    // Skip shadow sampling if fragment faces away from light and thus must be in shadow
    if diffuse > 0.0 {
        diffuse *= shadow(in.position_ws, in.depth_vs, normal);
    }

    // let layer = csm_layer(in.position_ws, in.depth_vs);
    // var layer_color = vec4(0.0, 0.0, 0.0, 1.0);
    // if layer == 0 {
    //     layer_color =  vec4(1.0, 0.0, 0.0, 1.0);
    // }
    // if layer == 1 {
    //     layer_color = vec4(0.0, 1.0, 0.0, 1.0);
    // }
    // if layer == 2 {
    //     layer_color = vec4(0.0, 0.0, 1.0, 1.0);
    // }

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

    return vec4((ambient + diffuse) * tex_color.rgb * color_multiplier, 1.0);
}
