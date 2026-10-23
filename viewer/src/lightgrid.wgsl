// Floodlight irradiance volume ("ambient cube" per cell): for every grid cell, the irradiance
// arriving at a surface facing +X, -X, +Y, -Y, +Z, -Z from all spot lights. Built once; shading
// then costs one trilinear lookup instead of a loop over every fixture.

struct Light {
    pos: vec4<f32>,          // w = intensity (W/sr)
    dir: vec4<f32>,          // w = cos(outer)
    color: vec4<f32>,        // w = cos(inner)
};
struct Lights { items: array<Light, 512> };
struct GridU {
    origin: vec4<f32>,       // xyz min corner, w = cell size
    dims: vec4<u32>,         // x, y, z, light count
};

@group(0) @binding(0) var<uniform> lights: Lights;
@group(0) @binding(1) var<uniform> grid: GridU;
@group(0) @binding(2) var px: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var nx: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var py: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var ny: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var pz: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var nz: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn build(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= grid.dims.xyz)) { return; }
    let p = grid.origin.xyz + (vec3<f32>(id) + 0.5) * grid.origin.w;
    var e = array<vec3<f32>, 6>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    for (var k = 0u; k < grid.dims.w; k++) {
        let li = lights.items[k];
        let d = li.pos.xyz - p;
        let d2 = max(dot(d, d), 1.0);
        let l = d * inverseSqrt(d2);
        let mask = smoothstep(li.dir.w, li.color.w, dot(-l, li.dir.xyz));
        if (mask <= 0.0) { continue; }
        let c = li.color.rgb * (li.pos.w * mask / d2);
        e[0] += c * max(l.x, 0.0);
        e[1] += c * max(-l.x, 0.0);
        e[2] += c * max(l.y, 0.0);
        e[3] += c * max(-l.y, 0.0);
        e[4] += c * max(l.z, 0.0);
        e[5] += c * max(-l.z, 0.0);
    }
    let i = vec3<i32>(id);
    textureStore(px, i, vec4<f32>(e[0], 1.0));
    textureStore(nx, i, vec4<f32>(e[1], 1.0));
    textureStore(py, i, vec4<f32>(e[2], 1.0));
    textureStore(ny, i, vec4<f32>(e[3], 1.0));
    textureStore(pz, i, vec4<f32>(e[4], 1.0));
    textureStore(nz, i, vec4<f32>(e[5], 1.0));
}
