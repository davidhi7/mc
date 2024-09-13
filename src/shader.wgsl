struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) tex_index: u32,
    @location(10) direction: u32,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coordinates: vec2<f32>,
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
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    var out: VertexOutput;
    out.tex_coordinates = model.tex_coordinates;
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    out.tex_index = instance.tex_index;
    out.direction = instance.direction;
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
