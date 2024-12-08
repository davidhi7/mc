#version 450

layout(set = 1, binding = 0) uniform CameraUniform {
    mat4 view_proj;
};

struct Vertex {
    vec3 position;
    vec2 tex_coordinates;
};

layout(set = 2, binding = 0) uniform Vertices {
    Vertex vertices[48];
};

layout(set = 3, binding = 0) readonly buffer Chunk {
    ivec3 chunk[200];
};

layout(location = 0) in uint instance_attributes;
layout(location = 1) in uint instance_ao_attributes;
layout(location = 0) out vec2 v_tex_coordinates;
layout(location = 1) flat out uint v_tex_index;
layout(location = 2) flat out uint v_direction;
layout(location = 3) out float v_ao_intensity;

void main() {
    uint vertex_index = uint(gl_VertexIndex);

    ivec3 chunk_relative_coords = ivec3(
        int((instance_attributes >>  0) & 0x1F),
        int((instance_attributes >>  5) & 0x1F),
        int((instance_attributes >> 10) & 0x1F)
    );

    uint tex_index = (instance_attributes >> 15) & 0xFF;
    uint direction = (instance_attributes >> 23) & 0x7;

    uint ao_0 = instance_ao_attributes & 3;
    uint ao_1 = (instance_ao_attributes >> 2) & 3;
    uint ao_2 = (instance_ao_attributes >> 4) & 3;
    uint ao_3 = (instance_ao_attributes >> 6) & 3;

    uint vertex_ao_factor_index = vertex_index;
    uint quad_index = 2u * direction;

    if (ao_0 + ao_3 < ao_1 + ao_2) {
        // Use the next quad that is flipped
        quad_index += 1u;

        // Map old to new AO attribute index
        uint ao_index_permutation[4] = uint[4](1, 3, 0, 2);
        vertex_ao_factor_index = ao_index_permutation[vertex_index];
    }

    uint ao_intensity = (instance_ao_attributes >> (2u * vertex_ao_factor_index)) & 0x3u;
    Vertex vertex = vertices[quad_index * 4u + vertex_index];
    vec3 global_position = vec3(32 * chunk[gl_DrawID] + chunk_relative_coords) + vertex.position;

    gl_Position = view_proj * vec4(global_position, 1.0);
    v_tex_coordinates = vertex.tex_coordinates;
    v_tex_index = tex_index;
    v_direction = direction;
    v_ao_intensity = float(ao_intensity);
}
