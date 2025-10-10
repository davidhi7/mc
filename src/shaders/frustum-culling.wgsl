struct Plane {
    normal: vec3f,
    distance: f32,
}

struct CameraFrustum {
    top: Plane,
    bottom: Plane,
    left: Plane,
    right: Plane,
    near: Plane,
    far: Plane,
};

struct BufferBounds {
    chunk_count: u32,
    draw_count: u32
}

struct ChunkWithVisibility {
    uvw: vec3i,
    visible: u32,
}

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

@group(0) @binding(0)
var<uniform> frustum: CameraFrustum;

@group(0) @binding(1)
var<uniform> bounds: BufferBounds;

@group(0) @binding(2)
var<storage, read_write> chunks: array<ChunkWithVisibility>;

@group(0) @binding(3)
var<storage, read_write> draws: array<DrawIndirectArgs>;

fn is_on_or_behind_plane(plane: Plane, point: vec3f) -> bool {
    return dot(plane.normal, point) <= plane.distance;
}

fn chunk_in_frustum(chunk: vec3i) -> bool {
    // TODO more false negatives, but less false positives
    // for (var i = 0u; i < 8; i++) {
    //     let position = 32 * (chunk + vec3i(vec3u(i & 1, (i >> 1) & 1, (i >> 2) & 1)));
    //     if is_on_or_behind_plane(frustum.near, vec3f(position))
    //         && is_on_or_behind_plane(frustum.far, vec3f(position))
    //         && is_on_or_behind_plane(frustum.left, vec3f(position))
    //         && is_on_or_behind_plane(frustum.right, vec3f(position))
    //         && is_on_or_behind_plane(frustum.top, vec3f(position))
    //         && is_on_or_behind_plane(frustum.bottom, vec3f(position))
    //     {
    //         return true;
    //     }
    // }

    // TODO more false positives, but no false negatives
    let center = 32.0 * (vec3f(chunk) + vec3f(0.5, 0.5, 0.5));
    let radius = sqrt(3.0) * 16;
    if dot(frustum.near.normal, center) - frustum.near.distance <= radius
        && dot(frustum.far.normal, center) - frustum.far.distance <= radius
        && dot(frustum.left.normal, center) - frustum.left.distance <= radius
        && dot(frustum.right.normal, center) - frustum.right.distance <= radius
        && dot(frustum.top.normal, center) - frustum.top.distance <= radius
        && dot(frustum.bottom.normal, center) - frustum.bottom.distance <= radius
    {
        return true;
    }
    return false;
}

@compute @workgroup_size(64)
fn compute_chunk_visibility(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index < bounds.chunk_count {
        chunks[index].visible = u32(chunk_in_frustum(chunks[index].uvw));   
    }
}

@compute @workgroup_size(64)
fn write_chunk_data(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index < bounds.draw_count {
        let chunk_index = draws[index].first_vertex >> 2;

        if chunks[chunk_index].visible == 1 {
            draws[index].vertex_count = 4u;
        } else {
            draws[index].vertex_count = 0u;
        }
    }
}
