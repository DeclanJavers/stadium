// Stadium viewer: PBR forward shading in the kit's units (irradiance in W/m², 1 W/m² ≈ 683 lux;
// a lit Lambert surface shows albedo × irradiance, as in the Blender review renders).

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,      // xyz towards the sun, w = irradiance (W/m²)
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,   // ambient irradiance from above (rgb)
    sky_horizon: vec4<f32>,
    ground: vec4<f32>,
    params: vec4<f32>,       // x exposure, y light count, z night, w shadow on
    pitch: vec4<f32>,        // x stripe width, y angle (rad), z strength, w pattern id
    pitch2: vec4<f32>,       // x natural grass (1) or woven artificial stripes (0)
    grid_origin: vec4<f32>,  // light grid min corner, w = cell size (0 = no grid: loop the lights)
    grid_size: vec4<f32>,    // grid extent in metres
};

struct Light {
    pos: vec4<f32>,          // w = intensity (W/sr)
    dir: vec4<f32>,          // w = cos(outer)
    color: vec4<f32>,        // w = cos(inner)
};

struct MaterialU {
    base_color: vec4<f32>,
    emissive: vec4<f32>,     // rgb × strength
    params: vec4<f32>,       // metallic, roughness, alpha cutoff, transmission
    flags: vec4<f32>,        // has base, has metal-rough, has normal, has emissive
    flags2: vec4<f32>,       // alpha mode (0 opaque, 1 mask, 2 blend), grass, -, -
    xf_base: vec4<f32>,      // KHR_texture_transform: offset.xy, scale.xy
    xf_mr: vec4<f32>,
    xf_normal: vec4<f32>,
    xf_emissive: vec4<f32>,
    rot: vec4<f32>,          // rotation per map
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
struct Lights { items: array<Light, 512> };
@group(0) @binding(3) var<uniform> lights: Lights;
@group(0) @binding(4) var g_px: texture_3d<f32>;
@group(0) @binding(5) var g_nx: texture_3d<f32>;
@group(0) @binding(6) var g_py: texture_3d<f32>;
@group(0) @binding(7) var g_ny: texture_3d<f32>;
@group(0) @binding(8) var g_pz: texture_3d<f32>;
@group(0) @binding(9) var g_nz: texture_3d<f32>;
@group(0) @binding(10) var g_samp: sampler;

// ambient-cube lookup: irradiance on a surface with normal n at world position p
fn grid_irradiance(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let uvw = (p - g.grid_origin.xyz) / g.grid_size.xyz;
    let n2 = n * n;
    let ex = select(textureSampleLevel(g_nx, g_samp, uvw, 0.0).rgb, textureSampleLevel(g_px, g_samp, uvw, 0.0).rgb, n.x > 0.0);
    let ey = select(textureSampleLevel(g_ny, g_samp, uvw, 0.0).rgb, textureSampleLevel(g_py, g_samp, uvw, 0.0).rgb, n.y > 0.0);
    let ez = select(textureSampleLevel(g_nz, g_samp, uvw, 0.0).rgb, textureSampleLevel(g_pz, g_samp, uvw, 0.0).rgb, n.z > 0.0);
    return ex * n2.x + ey * n2.y + ez * n2.z;
}

@group(1) @binding(0) var<uniform> m: MaterialU;
@group(1) @binding(1) var t_base: texture_2d<f32>;
@group(1) @binding(2) var s_base: sampler;
@group(1) @binding(3) var t_mr: texture_2d<f32>;
@group(1) @binding(4) var s_mr: sampler;
@group(1) @binding(5) var t_normal: texture_2d<f32>;
@group(1) @binding(6) var s_normal: sampler;
@group(1) @binding(7) var t_emissive: texture_2d<f32>;
@group(1) @binding(8) var s_emissive: sampler;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) m0: vec4<f32>,
    @location(4) m1: vec4<f32>,
    @location(5) m2: vec4<f32>,
    @location(6) m3: vec4<f32>,
    @location(7) icolor: vec4<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) icolor: vec4<f32>,
};

@vertex
fn vs_main(v: VIn) -> VOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let w = model * vec4<f32>(v.pos, 1.0);
    var o: VOut;
    o.clip = g.view_proj * w;
    o.world = w.xyz;
    o.normal = (model * vec4<f32>(v.normal, 0.0)).xyz;
    o.uv = v.uv;
    o.icolor = v.icolor;
    return o;
}

fn xform(uv: vec2<f32>, xf: vec4<f32>, r: f32) -> vec2<f32> {
    let s = uv * xf.zw;
    let c = cos(r);
    let sn = sin(r);
    return vec2<f32>(c * s.x + sn * s.y, -sn * s.x + c * s.y) + xf.xy;
}

fn shadow(world: vec3<f32>, n: vec3<f32>) -> f32 {
    if (g.params.w < 0.5) { return 1.0; }
    let p = g.light_view_proj * vec4<f32>(world + n * 0.08, 1.0);
    let ndc = p.xyz / p.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) { return 1.0; }
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_map));
    var sum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(f32(x), f32(y)) * texel, ndc.z - 0.0008);
        }
    }
    return sum / 9.0;
}

fn ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let nl = max(dot(n, l), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let nh = max(dot(n, h), 0.0);
    let a = max(rough * rough, 0.002);
    let a2 = a * a;
    let d = a2 / (3.14159265 * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    let k = a / 2.0;
    let gg = (nl / (nl * (1.0 - k) + k)) * (nv / (nv * (1.0 - k) + k));
    let f = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    return d * gg * f / max(4.0 * nl * nv, 1e-4);
}

// Mowing stripes, exactly as docs/SPECS.md: lean direction from the pattern, brightness from the view.
fn stripe_factor(world: vec3<f32>, view_dir: vec3<f32>) -> f32 {
    let pattern = g.pitch.w;
    if (pattern < 0.5) { return 1.0; }
    let bp = vec2<f32>(world.x, -world.z);             // Blender ground coords
    var ang = g.pitch.y;
    if (pattern > 3.5 && pattern < 5.5) { ang += 0.785398; }
    let ca = cos(ang);
    let sa = sin(ang);
    let xr = bp.x * ca + bp.y * sa;
    let yr = bp.y * ca - bp.x * sa;
    let w = g.pitch.x;
    let ex = vec2<f32>(ca, sa);
    let ey = vec2<f32>(-sa, ca);
    var lean = vec2<f32>(0.0);
    var s = 0.0;
    if (pattern < 1.5 || (pattern > 3.5 && pattern < 4.5)) {          // stripes, diagonal
        s = 1.0 - 2.0 * (floor(xr / w) - 2.0 * floor(floor(xr / w) / 2.0));
        lean = ey * s;
    } else if (pattern < 2.5) {                                         // stripes_long
        s = 1.0 - 2.0 * (floor(yr / w) - 2.0 * floor(floor(yr / w) / 2.0));
        lean = ex * s;
    } else if (pattern < 5.5) {                                         // checkerboard, diamond
        let s1 = 1.0 - 2.0 * (floor(xr / w) - 2.0 * floor(floor(xr / w) / 2.0));
        let s2 = 1.0 - 2.0 * (floor(yr / w) - 2.0 * floor(floor(yr / w) / 2.0));
        lean = (ey * s1 + ex * s2) * 0.7071;
        s = (s1 + 0.5 * s2) / 1.5;
    } else {                                                            // circles
        let r = length(bp);
        s = 1.0 - 2.0 * (floor(r / w) - 2.0 * floor(floor(r / w) / 2.0));
        lean = normalize(vec2<f32>(-bp.y, bp.x) + vec2<f32>(1e-5)) * s;
    }
    if (g.pitch2.x < 0.5) { return 1.0 + g.pitch.z * 0.35 * s; }       // artificial: woven-in colour
    let vb = vec2<f32>(view_dir.x, -view_dir.z);
    return 1.0 + g.pitch.z * dot(vb, lean);
}

// Opaque and alpha-tested surfaces after the depth pre-pass: depth EQUAL already rejects the cut-out
// texels, and a shader without `discard` keeps early-z on (it is what makes the pre-pass pay off).
@fragment
fn fs_opaque(i: VOut) -> @location(0) vec4<f32> {
    return shade(i);
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    let c = shade(i);
    if (m.flags2.x > 0.5 && m.flags2.x < 1.5 && c.a < 0.0) { discard; }
    return c;
}

fn shade(i: VOut) -> vec4<f32> {
    // sample every map unconditionally (absent maps are 1x1 defaults): keeps sampling and
    // derivatives in uniform control flow; the alpha test happens at the end
    let tb = textureSample(t_base, s_base, xform(i.uv, m.xf_base, m.rot.x));
    let tmr = textureSample(t_mr, s_mr, xform(i.uv, m.xf_mr, m.rot.y));
    let tnm = textureSample(t_normal, s_normal, xform(i.uv, m.xf_normal, m.rot.z)).xyz;
    let tem = textureSample(t_emissive, s_emissive, xform(i.uv, m.xf_emissive, m.rot.w)).rgb;
    let dp1 = dpdx(i.world);
    let dp2 = dpdy(i.world);
    let duv1 = dpdx(i.uv);
    let duv2 = dpdy(i.uv);

    var base = m.base_color;
    if (i.icolor.w > 0.5) { base = vec4<f32>(i.icolor.rgb, 1.0); }
    base *= mix(vec4<f32>(1.0), tb, m.flags.x);
    let alpha_mode = m.flags2.x;
    let metallic = m.params.x * mix(1.0, tmr.b, m.flags.y);
    let rough = m.params.y * mix(1.0, tmr.g, m.flags.y);
    var n = normalize(i.normal);
    let v = normalize(g.cam_pos.xyz - i.world);
    if (dot(n, v) < 0.0) { n = -n; }                     // double-sided everywhere
    if (m.flags.z > 0.5) {
        let tn = tnm * 2.0 - 1.0;
        let dp2perp = cross(dp2, n);
        let dp1perp = cross(n, dp1);
        let t = dp2perp * duv1.x + dp1perp * duv2.x;
        let b = dp2perp * duv1.y + dp1perp * duv2.y;
        let inv = inverseSqrt(max(max(dot(t, t), dot(b, b)), 1e-12));
        n = normalize(mat3x3<f32>(t * inv, -b * inv, n) * tn);
    }
    var albedo = base.rgb;
    if (m.flags2.y > 0.5) {
        albedo *= stripe_factor(i.world, -v);
    }
    let diffuse = albedo * (1.0 - metallic);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    var color = vec3<f32>(0.0);

    // sun
    let l = normalize(g.sun_dir.xyz);
    let nl = max(dot(n, l), 0.0);
    if (g.sun_dir.w > 0.0 && nl > 0.0) {
        let sh = shadow(i.world, n);
        color += (diffuse + 3.14159265 * ggx(n, v, l, rough, f0)) * g.sun_dir.w * g.sun_color.rgb * nl * sh;
    }
    // sky and ground bounce
    let up = n.y * 0.5 + 0.5;
    let amb = mix(g.ground.rgb, mix(g.sky_horizon.rgb, g.sky_zenith.rgb, clamp(n.y, 0.0, 1.0)), up);
    color += diffuse * amb;
    let r = reflect(-v, n);
    let env = mix(g.sky_horizon.rgb, g.sky_zenith.rgb, clamp(r.y, 0.0, 1.0)) * select(0.3, 1.0, r.y > 0.0);
    let fres = f0 + (max(vec3<f32>(1.0 - rough), f0) - f0) * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    color += env * fres * (1.0 - rough * 0.7);

    // floodlights: precomputed irradiance volume, or the exact per-light loop
    if (g.grid_origin.w > 0.0 && g.params.z > 0.5) {
        color += diffuse * grid_irradiance(i.world, n);
    }
    let count = select(u32(g.params.y), 0u, g.grid_origin.w > 0.0);
    for (var k = 0u; k < count; k++) {
        let li = lights.items[k];
        let d = li.pos.xyz - i.world;
        let d2 = max(dot(d, d), 1.0);
        let ld = d * inverseSqrt(d2);
        let cos_t = dot(-ld, li.dir.xyz);
        let mask = smoothstep(li.dir.w, li.color.w, cos_t);
        let nll = max(dot(n, ld), 0.0);
        if (mask > 0.0 && nll > 0.0) {
            let e = li.pos.w * nll / d2 * mask;
            color += (diffuse + 3.14159265 * ggx(n, v, ld, rough, f0)) * e * li.color.rgb;
        }
    }

    color += m.emissive.rgb * mix(vec3<f32>(1.0), tem, m.flags.w);

    var a = 1.0;
    if (alpha_mode > 1.5) {
        a = base.a;
        if (m.params.w > 0.0) { a = mix(a, 0.22, m.params.w); }
    }
    return vec4<f32>(color, a);
}

// ---------------------------------------------------------------------------------------------
// sky: a full-screen triangle at the far plane

struct SkyOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) vi: u32) -> SkyOut {
    let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
    var o: SkyOut;
    o.clip = vec4<f32>(p, 1.0, 1.0);
    o.ndc = p;
    return o;
}

@fragment
fn fs_sky(i: SkyOut) -> @location(0) vec4<f32> {
    let far = g.inv_view_proj * vec4<f32>(i.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - g.cam_pos.xyz);
    let h = dir.y;
    var c = mix(g.sky_horizon.rgb, g.sky_zenith.rgb, pow(clamp(h, 0.0, 1.0), 0.45)) * 1.6;
    if (h < 0.0) { c = mix(g.sky_horizon.rgb * 0.8, g.ground.rgb * 0.6, clamp(-h * 8.0, 0.0, 1.0)); }
    let s = max(dot(dir, normalize(g.sun_dir.xyz)), 0.0);
    c += g.sun_color.rgb * pow(s, 400.0) * 40.0 * select(0.0, 1.0, g.sun_dir.w > 0.0);
    return vec4<f32>(c, 1.0);
}

// ---------------------------------------------------------------------------------------------
// shadow map (alpha-tested so nets and cards cast their pattern, not a solid sheet)

struct SOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_shadow(v: VIn) -> SOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    var o: SOut;
    o.clip = g.light_view_proj * model * vec4<f32>(v.pos, 1.0);
    o.uv = v.uv;
    return o;
}

@fragment
fn fs_shadow(i: SOut) {
    let a = m.base_color.a * mix(1.0, textureSample(t_base, s_base, xform(i.uv, m.xf_base, m.rot.x)).a, m.flags.x);
    if (m.flags2.x > 0.5 && a < max(m.params.z, 0.35)) { discard; }
}

// depth pre-pass: lay down depth (alpha-tested) so the expensive shading runs once per pixel
@fragment
fn fs_prepass(i: VOut) -> @location(0) vec4<f32> {
    let a = m.base_color.a * mix(1.0, textureSample(t_base, s_base, xform(i.uv, m.xf_base, m.rot.x)).a, m.flags.x);
    if (m.flags2.x > 0.5 && m.flags2.x < 1.5 && a < m.params.z) { discard; }
    return vec4<f32>(0.0);
}
