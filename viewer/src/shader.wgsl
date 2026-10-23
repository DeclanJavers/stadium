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
    grass_color: vec4<f32>,  // procedural grass base (linear rgb), w = wear
    grass_dry: vec4<f32>,    // dry / stressed colour, w = wetness
    grass_dims: vec4<f32>,   // half length, half width, blade height (m), w = 1 if procedural
    sky_origin: vec4<f32>,   // sky-visibility volume min corner, w = cell size (0 = none)
    sky_size: vec4<f32>,
    screen: vec4<f32>,       // viewport w, h, -, time
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
    flags2: vec4<f32>,       // alpha mode (0 opaque, 1 mask, 2 blend), grass, procedural grass, takes instance colour
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
@group(0) @binding(11) var sky_vol: texture_3d<f32>;

// fraction of the sky this point sees (roofs and stands block it); sampled a cell off the surface
// so the solid structure itself does not darken its own faces
fn sky_visibility(p: vec3<f32>, n: vec3<f32>) -> f32 {
    if (g.sky_origin.w <= 0.0) { return 1.0; }
    let q = p + n * g.sky_origin.w * 0.9;
    let uvw = (q - g.sky_origin.xyz) / g.sky_size.xyz;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { return 1.0; }
    return textureSampleLevel(sky_vol, g_samp, uvw, 0.0).r;
}

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

// ---------------------------------------------------------------------------------------------
// procedural grass (docs/SPECS.md#grass): the pitch is a canopy of blades, not a painted plane.
// Colour varies at field, clump and blade scale (anti-aliased by the pixel footprint so distant
// grass averages correctly); thatch shows between blades seen from above and hides at grazing
// angles; wear, dry stress and wetness follow the layout; lighting adds the canopy's opposition
// hotspot and blade translucency and keeps specular to a faint waxy sheen.

fn hash21(p: vec2<f32>) -> f32 {
    var q = vec2<u32>(vec2<i32>(floor(p)) + vec2<i32>(32768));
    var h = q.x * 0x8da6b343u ^ q.y * 0xd8163841u;
    h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
    h = (h ^ (h >> 12u)) * 0x297a2d39u;
    h = h ^ (h >> 15u);
    return f32(h) / 4294967295.0;
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// noise at `freq` cycles/m, faded to its mean where the pixel footprint `fp` cannot resolve it
fn aa_noise(p: vec2<f32>, freq: f32, fp: f32) -> f32 {
    let fade = clamp(1.6 - fp * freq * 2.2, 0.0, 1.0);
    return mix(0.5, vnoise(p * freq), fade);
}

fn gauss2(p: vec2<f32>, c: vec2<f32>, s: vec2<f32>) -> f32 {
    let d = (p - c) / s;
    return exp(-0.5 * dot(d, d));
}

// blade height field (0 = thatch, 1 = blade tips), streaked along the mowing direction
fn blade_height(q: vec2<f32>) -> f32 {
    let a = vnoise(vec2<f32>(q.x * 95.0, q.y * 30.0));
    let b = vnoise(vec2<f32>(q.x * 210.0 + 13.0, q.y * 70.0));
    let h = a * 0.6 + b * 0.4;
    return smoothstep(0.25, 0.85, h);
}

// Parallax ray march into the blade layer: (offset ground point, height reached 0..1).
fn grass_parallax(p: vec2<f32>, ray: vec3<f32>, along: vec2<f32>, depth: f32, fp: f32) -> vec3<f32> {
    // only where blades are resolvable and the view is not straight down
    let near = clamp(1.6 - fp * 120.0, 0.0, 1.0);
    if (near <= 0.0 || ray.y > -0.02) { return vec3<f32>(p, 1.0); }
    let dxy = vec2<f32>(ray.x, -ray.z) / max(-ray.y, 0.08);   // ground travel per metre of descent
    let n = 14;
    var hgt = 1.0;
    var xy = p;
    let step_h = 1.0 / f32(n);
    var prev_h = 1.0;
    var prev_f = 0.0;
    for (var k = 0; k < n; k++) {
        let qq = vec2<f32>(dot(xy, along), dot(xy, vec2<f32>(-along.y, along.x)));
        let f = blade_height(qq);
        if (f >= hgt) {
            // refine between the last two samples
            let t = clamp((prev_h - prev_f) / max((prev_h - prev_f) - (hgt - f), 1e-4), 0.0, 1.0);
            let hh = mix(prev_h, hgt, t);
            return vec3<f32>(mix(p, xy, near), mix(1.0, hh, near));
        }
        prev_h = hgt;
        prev_f = f;
        hgt -= step_h;
        xy += dxy * depth * step_h;
    }
    return vec3<f32>(mix(p, xy, near), mix(1.0, 0.0, near));
}

struct Grass {
    albedo: vec3<f32>,
    rough: f32,
    soil: f32,      // 0 turf .. 1 bare soil
    gaps: f32,      // how much thatch shows (dark), for the canopy shading
};

fn grass_surface(world: vec3<f32>, view_dir: vec3<f32>, fp: f32, baked: vec3<f32>, use_baked: bool) -> Grass {
    let p0 = vec2<f32>(world.x, -world.z);            // kit ground coords
    let along0 = vec2<f32>(cos(g.pitch.y), sin(g.pitch.y));
    let blade_depth = clamp(g.grass_dims.z, 0.01, 0.06);
    let par = grass_parallax(p0, -view_dir, along0, blade_depth, fp);
    let p = par.xy;
    let blade_ao = par.z;
    let natural = g.pitch2.x > 0.5;
    let base = g.grass_color.rgb;
    let dry = g.grass_dry.rgb;
    let wear = g.grass_color.w;
    let wet = g.grass_dry.w;
    let half = g.grass_dims.xy;
    // mowing lean direction: blades laid over along the stripe direction give a streaky texture
    let ang = g.pitch.y;
    let along = vec2<f32>(cos(ang), sin(ang));
    let q = vec2<f32>(dot(p, along), dot(p, vec2<f32>(-along.y, along.x)));
    // field-scale tone (shade of the turf over metres) and dry stress patches
    let n_large = aa_noise(p + 17.3, 0.035, fp) * 0.6 + aa_noise(p, 0.11, fp) * 0.4;
    let n_med = aa_noise(p + 3.1, 0.7, fp) * 0.6 + aa_noise(p - 9.4, 2.3, fp) * 0.4;
    var col = base;
    if (use_baked) {
        col = baked;
    } else {
        let k_large = select(0.05, 0.16, natural);
        let k_med = select(0.04, 0.10, natural);
        col *= (1.0 + (n_large - 0.5) * k_large) * (1.0 + (n_med - 0.5) * k_med);
        if (natural) {
            let stress = smoothstep(0.62, 0.85, aa_noise(p + 41.0, 0.08, fp)) * 0.14;
            col = mix(col, dry, stress);
        }
    }
    // blade scale: fine streaks along the mowing direction (anisotropic), tips and dead blades
    let blade = aa_noise(vec2<f32>(q.x * 38.0, q.y * 110.0), 1.0, fp * 110.0)
              * 0.55 + aa_noise(vec2<f32>(q.x * 90.0, q.y * 260.0), 1.0, fp * 260.0) * 0.45;
    let tips = aa_noise(p + 5.7, 55.0, fp);
    let dead = smoothstep(0.86, 0.97, aa_noise(p - 2.3, 85.0, fp));
    // thatch between blades: visible looking down, hidden behind blades at grazing angles
    let cos_view = clamp(view_dir.y, 0.0, 1.0);
    let depth = clamp(g.grass_dims.z / 0.025, 0.5, 2.0);
    let gaps = smoothstep(0.30, 0.0, blade) * mix(0.25, 1.0, cos_view) * select(0.6, 1.0, natural);
    let thatch = vec3<f32>(0.035, 0.030, 0.012);
    var albedo = col * (0.86 + 0.30 * blade);
    albedo = mix(albedo, albedo * vec3<f32>(1.12, 1.10, 0.80) + vec3<f32>(0.006, 0.006, 0.0), smoothstep(0.55, 0.85, tips) * 0.5);
    if (natural) {
        albedo = mix(albedo, dry * 0.9, dead * 0.35);
    }
    albedo = mix(albedo, thatch, gaps * 0.55);
    // seen across the pitch you look into the blade sides, which are a little darker
    albedo *= 1.0 - 0.08 * (1.0 - cos_view) * depth;

    // wear: goalmouths, keepers' lines, penalty spots, centre spot, assistant referees' lanes
    var wf = 0.7 * gauss2(p, vec2<f32>(0.0), vec2<f32>(2.2));
    for (var k = 0; k < 2; k++) {
        let sg = select(-1.0, 1.0, k == 0);
        wf += gauss2(p, vec2<f32>(sg * (half.x - 5.5), 0.0), vec2<f32>(4.5, 6.0));
        wf += 1.4 * gauss2(p, vec2<f32>(sg * (half.x - 1.2), 0.0), vec2<f32>(1.4, 3.6));
        wf += 0.8 * gauss2(p, vec2<f32>(sg * (half.x - 11.0), 0.0), vec2<f32>(0.8));
        let lane = exp(-0.5 * pow((p.y - sg * (half.y + 1.0)) / 0.7, 2.0)) * smoothstep(0.45, 0.7, aa_noise(p, 0.9, fp));
        wf += 0.4 * lane * smoothstep(-2.0, 2.0, p.x * sg);
    }
    let breakup = aa_noise(p + 7.0, 1.6, fp) * 0.6 + aa_noise(p + 1.0, 4.1, fp) * 0.4;
    let wamt = wear * select(0.25, 1.0, natural);
    let wmask = clamp(wf * (breakup * 1.4 - 0.2) * wamt * 1.6, 0.0, 1.0);
    let soil_col = mix(vec3<f32>(0.068, 0.040, 0.019), vec3<f32>(0.147, 0.090, 0.042), aa_noise(p, 6.0, fp));
    albedo = mix(albedo, dry * (0.9 + 0.2 * blade), smoothstep(0.12, 0.55, wmask) * 0.85);
    let soil = smoothstep(0.45, 0.9, wmask);
    albedo = mix(albedo, soil_col, soil);
    if (!natural) {
        let crumb = smoothstep(0.62, 0.66, aa_noise(p, 260.0, fp));
        albedo = mix(albedo, vec3<f32>(0.010), crumb * 0.5);
    }
    albedo *= 1.0 - 0.3 * wet;
    // inside the blade layer: tips catch the light, the base is deep in shade
    let ao = mix(0.30, 1.0, pow(blade_ao, 0.7));
    albedo *= mix(1.0, ao, 1.0 - soil);
    var out: Grass;
    out.albedo = albedo;
    out.rough = mix(mix(0.82, 0.62, wet), mix(0.9, 0.35, wet), soil);
    out.soil = soil;
    out.gaps = gaps;
    return out;
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
    let fp = max(length(dp1), length(dp2));

    var base = m.base_color;
    if (i.icolor.w > 0.5 && m.flags2.w > 0.5) { base = vec4<f32>(i.icolor.rgb, 1.0); }
    base *= mix(vec4<f32>(1.0), tb, m.flags.x);
    let alpha_mode = m.flags2.x;
    let metallic = m.params.x * mix(1.0, tmr.b, m.flags.y);
    var rough = m.params.y * mix(1.0, tmr.g, m.flags.y);
    var n = normalize(i.normal);
    let v = normalize(g.cam_pos.xyz - i.world);
    if (dot(n, v) < 0.0) { n = -n; }                     // double-sided everywhere
    let n_geo = n;
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
    var spec = 1.0;            // specular scale (grass: a faint waxy sheen only)
    var canopy = 0.0;          // 1 = shade as a grass canopy
    var gaps = 0.0;
    let grass = m.flags2.y > 0.5;
    let paint = m.flags2.z > 1.5;
    if (grass || paint) {
        let procedural = m.flags2.z > 0.5 && m.flags2.z < 1.5;
        let gs = grass_surface(i.world, v, fp, base.rgb, grass && !procedural);
        if (grass) {
            albedo = gs.albedo * stripe_factor(i.world, -v);
            rough = gs.rough;
            canopy = 1.0 - gs.soil;
            gaps = gs.gaps;
            spec = mix(0.25, 1.0, gs.soil);
        } else {
            // line paint sprayed onto blades: grass texture shows through, ragged coverage
            let p = vec2<f32>(i.world.x, -i.world.z);
            let cover = 0.78 + 0.18 * aa_noise(p, 9.0, fp) - 0.25 * gs.gaps;
            albedo = mix(gs.albedo, base.rgb * (0.92 + 0.08 * stripe_factor(i.world, -v)), clamp(cover, 0.0, 1.0));
            rough = 0.75;
            canopy = 0.6;
            spec = 0.4;
        }
        // blades tilt every which way: a little normal jitter breaks the flat-plane look
        let pj = vec2<f32>(i.world.x, -i.world.z);
        let jx = aa_noise(pj, 23.0, fp) - 0.5;
        let jz = aa_noise(pj + 11.0, 23.0, fp) - 0.5;
        n = normalize(n + vec3<f32>(jx, 0.0, jz) * 0.35 * canopy);
    }
    let diffuse = albedo * (1.0 - metallic);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    var color = vec3<f32>(0.0);

    // sun
    let l = normalize(g.sun_dir.xyz);
    let nl = max(dot(n, l), 0.0);
    if (g.sun_dir.w > 0.0) {
        let sh = shadow(i.world, n_geo);
        var direct = (diffuse + 3.14159265 * ggx(n, v, l, rough, f0) * spec) * nl;
        if (canopy > 0.0) {
            // opposition hotspot: with the sun behind you every visible blade is sunlit
            let g_ang = acos(clamp(dot(v, l), -1.0, 1.0));
            let hotspot = 1.0 + 0.45 * exp(-g_ang / 0.18);
            // self-shadowing inside the canopy away from the hotspot, deeper for long grass
            let selfsh = mix(1.0, mix(0.72, 1.0, exp(-g_ang / 0.9)), clamp(g.grass_dims.z / 0.03, 0.4, 1.3));
            // light through the blades when looking towards the sun
            let trans = pow(max(dot(-v, l), 0.0), 6.0) * 0.22;
            direct = diffuse * (nl * hotspot * mix(1.0, selfsh, canopy) + trans * canopy * vec3<f32>(0.9, 1.15, 0.35))
                   + 3.14159265 * ggx(n, v, l, rough, f0) * spec * nl;
        }
        color += direct * g.sun_dir.w * g.sun_color.rgb * sh;
    }
    // sky (blocked by roofs and stands: sky visibility) and bounce from the ground
    let vis = sky_visibility(i.world, n_geo);
    let ny = clamp(n.y, -1.0, 1.0);
    // cosine-weighted sky: an up-facing surface sees zenith and horizon alike
    let sky_amb = mix(g.sky_horizon.rgb, g.sky_zenith.rgb, clamp(ny, 0.0, 1.0) * 0.55) * (0.5 + 0.5 * ny);
    let gnd_amb = g.ground.rgb * (0.5 - 0.5 * ny);
    var amb = sky_amb * mix(0.08, 1.0, vis) + gnd_amb * mix(0.5, 1.0, vis);
    amb *= 1.0 - canopy * (0.12 + 0.15 * gaps);       // blades shade each other from the sky
    color += diffuse * amb;
    let r = reflect(-v, n);
    let env = mix(g.sky_horizon.rgb, g.sky_zenith.rgb, clamp(r.y, 0.0, 1.0)) * select(0.3, 1.0, r.y > 0.0);
    let fres = f0 + (max(vec3<f32>(1.0 - rough), f0) - f0) * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    // rough surfaces barely mirror the sky; occluded ones not at all
    let spec_occ = vis * vis * clamp(1.0 - rough * 0.9, 0.05, 1.0);
    color += env * fres * spec_occ * spec;
    if (canopy > 0.0) {
        // waxy blades catch the sky at grazing angles: a soft, desaturating sheen
        let graze = pow(1.0 - clamp(dot(n_geo, v), 0.0, 1.0), 3.0);
        color += mix(g.sky_horizon.rgb, g.sky_zenith.rgb, 0.3) * graze * 0.07 * vis * canopy;
    }

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
            color += (diffuse + 3.14159265 * ggx(n, v, ld, rough, f0) * spec) * e * li.color.rgb;
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
