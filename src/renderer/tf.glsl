#version 450

layout(set = 0, binding = 0) uniform texture2D t_diffuse;
layout(set = 0, binding = 1) uniform sampler s_diffuse;

layout(location = 0) in vec2 v_tex_coordinates;
layout(location = 1) flat in uint v_tex_index;
layout(location = 2) flat in uint v_direction;
layout(location = 3) in float v_ao_intensity;

layout(location = 0) out vec4 frag_color;

void main() {
    float lighting_factor = 1.0 - v_ao_intensity * 0.3;
    frag_color = lighting_factor * texture(sampler2D(t_diffuse, s_diffuse), v_tex_coordinates);
}
