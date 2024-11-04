struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

@group(2) @binding(0)
var<uniform> chunk: vec3i;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coordinates: vec2<f32>,
};

struct InstanceInput {
    @location(2) packed_bits: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coordinates: vec2<f32>,
    @location(1) @interpolate(flat) tex_index: u32,
    @location(2) @interpolate(flat) direction: u32,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput
) -> VertexOutput {
    let chunk_relative_coords = vec3i(
        i32((instance.packed_bits >>  0) & 0x1F),
        i32((instance.packed_bits >>  5) & 0x1F),
        i32((instance.packed_bits >> 10) & 0x1F)
    );

    let tex_index = (instance.packed_bits >> 15) & 0xFF;
    let direction = (instance.packed_bits >> 23) & 0x7;

    var model_coords = model.position;
    var model_tex_coordinates = model.tex_coordinates;
    
    switch direction {
        case 0u: {
            // -X
            model_coords = vec3f(0, model_coords.xy);
            model_tex_coordinates = model_tex_coordinates.yx;
        }
        case 1u: {
            // +X
            model_coords = vec3f(1, model_coords.yx);
        }
        case 2u: {
            // -Y
            model_coords = vec3f(model_coords.y, 0, model_coords.x);
        }
        case 3u: {
            // +Y
            model_coords = vec3f(model_coords.x, 1, model_coords.y);
        }
        case 4u, default {
            // case 4u is case -Z, which is the default direction of the model
        }
        case 5u: {
            // +Z
            model_coords = vec3f(model_coords.yx, 1);
            model_tex_coordinates = model_tex_coordinates.yx;
        }
    }
    
    let global_position = vec3f(32 * chunk + chunk_relative_coords) + model_coords;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4f(global_position, 1);
    out.tex_coordinates = model_tex_coordinates;
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
    var lighting_factor: f32;
    if in.direction < 2 {
        // X or -X
        lighting_factor = 0.7;
    } else if in.direction >= 4 {
        // Z or -Z
        lighting_factor = 0.85;
    } else {
        // Y or -Y
        lighting_factor = 1.1;
    }

    return lighting_factor * textureSample(t_diffuse[in.tex_index], s_diffuse, in.tex_coordinates);
}
