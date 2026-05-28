const PASS_COUNT: u32 = 5;
const BUCKET_COUNT: u32 = 2;
override INDIRECT_BUFFER_SLOTS: u32;

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

struct ChunkDescriptor {
    /// One entry per chunk bucket
    // todo correct size
    entries: array<ChunkDescriptorEntry, BUCKET_COUNT>,
    /// Index into the chunk uniforms buffer
    uniform_index: u32,
    _padding: array<u32, 3>,
}

struct ChunkDescriptorEntry {
    /// Byte offset into the vertex buffer where this chunk's instances begin.
    first_instance: u32,
    /// Number of quad instances in this chunk+bucket.
    instance_count: u32,
}

struct ChunkUniform {
    uvw: vec3i,
    _padding: u32,
}

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

@group(0) @binding(0)
var<uniform> frustums: array<CameraFrustum, PASS_COUNT>;

@group(0) @binding(1)
var<uniform> descriptor_count: u32;

@group(0) @binding(2)
var<storage, read> chunk_descriptors: array<ChunkDescriptor>;

@group(0) @binding(3)
var<storage, read> chunk_uniforms: array<ChunkUniform>;

@group(1) @binding(0)
var<storage, read_write> draw_calls: array<DrawIndirectArgs>;

@group(1) @binding(1)
var<storage, read_write> draw_counts: array<atomic<u32>>;

fn is_on_or_behind_plane(plane: Plane, point: vec3f) -> bool {
    return dot(plane.normal, point) <= plane.distance;
}

fn chunk_in_frustum(chunk: vec3i, frustum: CameraFrustum) -> bool {
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
fn run(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    // atomicStore(&draw_counts[0], 123);
    // return;

    if index >= descriptor_count {
        return;
    }


    for (var pass_index = 0u; pass_index < PASS_COUNT; pass_index++) {
        // todocheck chunk uniform access
        if !chunk_in_frustum(chunk_uniforms[index].uvw, frustums[pass_index]) {
            continue;
        };

        for (var bucket_index = 0u; bucket_index < BUCKET_COUNT; bucket_index++) {
            if chunk_descriptors[index].entries[bucket_index].instance_count == 0 {
                continue;
            }

            let offset = atomicAdd(&draw_counts[pass_index * BUCKET_COUNT + bucket_index], 1);

            draw_calls[(pass_index * BUCKET_COUNT + bucket_index) * INDIRECT_BUFFER_SLOTS] = DrawIndirectArgs(
                4,
                chunk_descriptors[index].entries[bucket_index].instance_count,
                chunk_descriptors[index].uniform_index << 2,
                chunk_descriptors[index].entries[bucket_index].first_instance,
            );
        }
    }
}
