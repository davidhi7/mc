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
    @location(2) attributes: u32,
    @location(3) ao_attributes: u32,
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
    model: VertexInput,
    instance: InstanceInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var model_coords = model.position;
    var model_tex_coordinates = model.tex_coordinates;

    let ao_0 = instance.ao_attributes & 3;
    let ao_1 = (instance.ao_attributes >> 2) & 3;
    let ao_2 = (instance.ao_attributes >> 4) & 3;
    let ao_3 = (instance.ao_attributes >> 6) & 3;

    var ao_intensity = (instance.ao_attributes >> (2 * vertex_index)) & 0x3;

    if (ao_0 + ao_3 < ao_1 + ao_2) {
        // Effectively rotate the two triangles that form a quad to fix AO interpolation issues
        model_coords.x = abs(1.0 - model_coords.x);
        model_coords = model_coords.yxz;
        model_tex_coordinates.x = abs(1.0 - model_tex_coordinates.x);
        model_tex_coordinates = model_tex_coordinates.yx;

        // Map old to new AO attribute index
        var ao_index_permutation = array<u32, 4>(1, 3, 0, 2);
        ao_intensity = (instance.ao_attributes >> (2 * ao_index_permutation[vertex_index])) & 3;
    }

    let chunk_relative_coords = vec3i(
        i32((instance.attributes >>  0) & 0x1F),
        i32((instance.attributes >>  5) & 0x1F),
        i32((instance.attributes >> 10) & 0x1F),
    );

    let tex_index = (instance.attributes >> 15) & 0xFF;
    let direction = (instance.attributes >> 23) & 0x7;
    
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
    out.ao_intensity = f32(ao_intensity);
    return out;
}

@group(0) @binding(0)
var t_diffuse: binding_array<texture_2d<f32>>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let lighting_factor = 1.0 - in.ao_intensity * 0.3;
    return lighting_factor * textureSample(t_diffuse[in.tex_index], s_diffuse, in.tex_coordinates);
}
