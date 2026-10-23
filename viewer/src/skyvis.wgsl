// Sky visibility volume: for every cell, the cosine-weighted fraction of the sky hemisphere that is
// not blocked by the stadium (24 orthographic depth maps rendered from directions over the sky).
// r = visibility (0..1), gba = mean unblocked direction (a "bent normal" for directional ambient).

struct Dirs { m: array<mat4x4<f32>, 32>, d: array<vec4<f32>, 32> };
struct GridU {
    origin: vec4<f32>,       // xyz min corner, w = cell size
    dims: vec4<u32>,         // x, y, z, direction count
};

@group(0) @binding(0) var<uniform> dirs: Dirs;
@group(0) @binding(1) var<uniform> grid: GridU;
@group(0) @binding(2) var depth: texture_depth_2d_array;
@group(0) @binding(3) var out_vis: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn build(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= grid.dims.xyz)) { return; }
    let p = grid.origin.xyz + (vec3<f32>(id) + 0.5) * grid.origin.w;
    let size = vec2<f32>(textureDimensions(depth));
    var vis = 0.0;
    var bent = vec3<f32>(0.0);
    let n = grid.dims.w;
    for (var k = 0u; k < n; k++) {
        let c = dirs.m[k] * vec4<f32>(p, 1.0);
        let ndc = c.xyz / c.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
        var open = 1.0;
        if (all(uv >= vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0))) {
            // 2x2 PCF in texels: soft edges between neighbouring cells
            let px = uv * size - 0.5;
            let i0 = vec2<i32>(floor(px));
            let f = fract(px);
            var o = 0.0;
            for (var dy = 0; dy < 2; dy++) {
                for (var dx = 0; dx < 2; dx++) {
                    let q = clamp(i0 + vec2<i32>(dx, dy), vec2<i32>(0), vec2<i32>(size) - 1);
                    let d = textureLoad(depth, q, i32(k), 0);
                    let w = select(1.0 - f.x, f.x, dx == 1) * select(1.0 - f.y, f.y, dy == 1);
                    o += w * select(0.0, 1.0, ndc.z <= d + 0.0005);
                }
            }
            open = o;
        }
        let inv_dir = dirs.d[k].xyz;
        vis += open;
        bent += open * inv_dir;
    }
    vis /= f32(n);
    let b = select(vec3<f32>(0.0, 1.0, 0.0), normalize(bent), length(bent) > 1e-4);
    textureStore(out_vis, vec3<i32>(id), vec4<f32>(vis, b));
}
