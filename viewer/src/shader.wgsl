// Stadium viewer: PBR forward shading in the kit's units (irradiance in W/m², 1 W/m² ≈ 683 lux;
// a lit Lambert surface shows albedo × irradiance, as in the Blender review renders).

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    light_view_proj: array<mat4x4<f32>, 3>,   // per shadow cascade
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,      // xyz towards the sun, w = irradiance (W/m²)
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,   // ambient irradiance from above (rgb)
    sky_horizon: vec4<f32>,
    ground: vec4<f32>,
    params: vec4<f32>,       // x exposure, y light count, z night, w shadow on
    grid_origin: vec4<f32>,  // light grid min corner, w = cell size (0 = no grid: loop the lights)
    grid_size: vec4<f32>,    // grid extent in metres
    sky_origin: vec4<f32>,   // sky-visibility volume min corner, w = cell size (0 = none)
    sky_size: vec4<f32>,
    screen: vec4<f32>,       // viewport w, h, -, time
    shadow: vec4<f32>,       // cascade far distances (x, y), w = the cascade a shadow pass renders
    shadow_texel: vec4<f32>, // world size of a shadow texel per cascade
    shadow_range: vec4<f32>, // depth range per cascade (m)
    gmap: vec4<f32>,         // ground map: min corner (x, z), size (x, z); size 0 = none
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
    // procedural grass, per material: every pitch and lawn has its own turf and frame
    grass_a: vec4<f32>,      // x stripe width, y angle (rad), z strength, w pattern id
    grass_b: vec4<f32>,      // x natural (1) or woven artificial (0), y wear zones on, zw frame origin (kit)
    grass_c: vec4<f32>,      // x cos yaw, y sin yaw, z blade height (m)
    grass_color: vec4<f32>,  // base (linear rgb), w = wear
    grass_dry: vec4<f32>,    // dry / stressed colour, w = wetness
    grass_half: vec4<f32>,   // half length, half width
    kind: vec4<f32>,         // x foliage, y LED panel (emissive texels are diodes), z glazing (2 = solar-control), w structural concrete
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var shadow_map: texture_depth_2d_array;
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
@group(0) @binding(12) var ground_map: texture_2d<f32>;

// Ground occlusion from the top-down ground map (x = height of the topmost surface): skylight on
// the ground is blocked by whatever stands taller close by (a bin, a bench leg, a goal post, a
// cabin wall), so props sit in a soft contact shade. Overhead structure (more than 6 m above) is
// left to the sky-visibility volume. 1 = open, down to about 0.45 right against a wall.
fn ground_ao(p: vec3<f32>) -> f32 {
    if (g.gmap.z <= 0.0) { return 1.0; }
    let uv0 = (vec2<f32>(p.x, p.z) - g.gmap.xy) / g.gmap.zw;
    if (any(uv0 < vec2<f32>(0.0)) || any(uv0 > vec2<f32>(1.0))) { return 1.0; }
    let h0 = textureSampleLevel(ground_map, g_samp, uv0, 0.0).x;
    var occ = 0.0;
    for (var k = 0; k < 8; k++) {
        let a = f32(k) * 0.785398 + 0.39;
        let d = vec2<f32>(cos(a), sin(a));
        var best = 0.0;
        for (var j = 0; j < 3; j++) {
            let r = 0.22 * pow(2.4, f32(j));
            let uv = uv0 + d * r / g.gmap.zw;
            let dh = textureSampleLevel(ground_map, g_samp, uv, 0.0).x - h0;
            if (dh > 0.03 && dh < 6.0) {
                best = max(best, clamp(min(dh, 2.5) / (r + min(dh, 2.5)), 0.0, 1.0));
            }
        }
        occ += best;
    }
    return 1.0 - 0.55 * occ / 8.0;
}

// fraction of the sky this point sees (roofs and stands block it); sampled a cell off the surface
// so the solid structure itself does not darken its own faces
fn sky_visibility(p: vec3<f32>, n: vec3<f32>) -> f32 {
    if (g.sky_origin.w <= 0.0) { return 1.0; }
    // two samples out along the normal: next to a thin solid object (a screen case, a board) the
    // nearer one blends in cells inside the object and reads as fully occluded
    var vis = 0.0;
    for (var k = 1; k <= 2; k++) {
        let q = p + n * g.sky_origin.w * 0.9 * f32(k);
        let uvw = (q - g.sky_origin.xyz) / g.sky_size.xyz;
        if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { return 1.0; }
        vis = max(vis, textureSampleLevel(sky_vol, g_samp, uvw, 0.0).r);
    }
    return vis;
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
    @location(8) vcolor: vec4<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) icolor: vec4<f32>,
    @location(4) vcolor: vec4<f32>,
    // a stable random number per instance (from where it stands): no two trees, hedges or bins
    // come out quite the same colour
    @location(5) @interpolate(flat) ivar: f32,
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
    o.vcolor = v.vcolor;
    o.ivar = fract(sin(dot(v.m3.xz, vec2<f32>(12.9898, 78.233)) + v.m3.y * 3.7) * 43758.5453);
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
    // cascade by distance from the camera: fine near, coarse far
    let dist = length(world - g.cam_pos.xyz);
    var k = 2u;
    if (dist < g.shadow.x) { k = 0u; } else if (dist < g.shadow.y) { k = 1u; }
    let tw = g.shadow_texel[k];
    // bias in texels: a normal offset of one and a half texels, a depth bias growing with slope
    let nl = clamp(dot(n, normalize(g.sun_dir.xyz)), 0.0, 1.0);
    let slope = min(sqrt(1.0 - nl * nl) / max(nl, 0.05), 4.0);
    let p = g.light_view_proj[k] * vec4<f32>(world + n * tw * 1.5, 1.0);
    let ndc = p.xyz / p.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) { return 1.0; }
    let bias = tw * (0.6 + slope) / g.shadow_range[k];
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_map).xy);
    var sum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(f32(x), f32(y)) * texel, i32(k), ndc.z - bias);
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

// kit ground coordinates (x, y) -> the grass frame of the current material (its pitch's own axes)
fn grass_local(world: vec3<f32>) -> vec2<f32> {
    let d = vec2<f32>(world.x, -world.z) - m.grass_b.zw;
    return vec2<f32>(d.x * m.grass_c.x + d.y * m.grass_c.y, d.y * m.grass_c.x - d.x * m.grass_c.y);
}

// a glTF direction expressed in the grass frame (still glTF: y up, frame x along +x, frame y along -z)
fn grass_dir(v: vec3<f32>) -> vec3<f32> {
    let k = vec2<f32>(v.x, -v.z);
    let l = vec2<f32>(k.x * m.grass_c.x + k.y * m.grass_c.y, k.y * m.grass_c.x - k.x * m.grass_c.y);
    return vec3<f32>(l.x, v.y, -l.y);
}

// Mowing: the lean direction of the blades (grass frame) and the stripe sign at a point.
// xy = lean (unit, grass-local), z = stripe sign (-1..1); pattern 0 (none) gives no lean.
fn mow(world: vec3<f32>) -> vec3<f32> {
    let pattern = m.grass_a.w;
    if (pattern < 0.5) { return vec3<f32>(0.0); }
    let bp = grass_local(world);                        // the pitch's ground coords
    var ang = m.grass_a.y;
    if (pattern > 3.5 && pattern < 5.5) { ang += 0.785398; }
    let ca = cos(ang);
    let sa = sin(ang);
    let xr = bp.x * ca + bp.y * sa;
    let yr = bp.y * ca - bp.x * sa;
    let w = m.grass_a.x;
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
    return vec3<f32>(lean, s);
}

// Mowing stripes, exactly as docs/SPECS.md: lean direction from the pattern, brightness from the view.
fn stripe_factor(world: vec3<f32>, view_dir_world: vec3<f32>) -> f32 {
    let pattern = m.grass_a.w;
    if (pattern < 0.5) { return 1.0; }
    let mw = mow(world);
    let lean = mw.xy;
    let s = mw.z;
    if (m.grass_b.x < 0.5) { return 1.0 + m.grass_a.z * 0.35 * s; }   // artificial: woven-in colour
    let view_dir = grass_dir(view_dir_world);
    let vb = vec2<f32>(view_dir.x, -view_dir.z);
    if (pattern > 5.5) {
        // circles: the lean flips round each ring (a smooth swing in brightness), so a little of the
        // roller's crushed-blade contrast keeps the ring edges readable all the way round
        return 1.0 + m.grass_a.z * (0.75 * dot(vb, lean) + 0.3 * s);
    }
    return 1.0 + m.grass_a.z * dot(vb, lean);
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
// blades are 3-5 mm wide: narrow ridges across the mowing direction, pointed at the tips
fn blade_height(q: vec2<f32>) -> f32 {
    let a = vnoise(vec2<f32>(q.x * 240.0, q.y * 38.0));
    let b = vnoise(vec2<f32>(q.x * 470.0 + 13.0, q.y * 85.0));
    let h = a * 0.55 + b * 0.45;
    let t = smoothstep(0.2, 0.9, h);
    return t * t * (1.6 - 0.6 * t);
}

// Parallax ray march into the blade layer: (offset ground point, height reached 0..1).
fn grass_parallax(p: vec2<f32>, ray: vec3<f32>, along: vec2<f32>, depth: f32, fp: f32) -> vec3<f32> {
    // only where blades are resolvable and the view is not straight down
    let near = clamp(1.6 - fp * 260.0, 0.0, 1.0);
    if (near <= 0.0 || ray.y > -0.02) { return vec3<f32>(p, 1.0); }
    let dxy = vec2<f32>(ray.x, -ray.z) / max(-ray.y, 0.08);   // ground travel per metre of descent
    let n = 20;
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

fn grass_surface(world: vec3<f32>, view_dir_world: vec3<f32>, fp: f32, baked: vec3<f32>, use_baked: bool) -> Grass {
    let p0 = grass_local(world);                       // the pitch's ground coords
    let view_dir = grass_dir(view_dir_world);
    let along0 = vec2<f32>(cos(m.grass_a.y), sin(m.grass_a.y));
    let blade_depth = clamp(m.grass_c.z, 0.01, 0.06);
    let par = grass_parallax(p0, -view_dir, along0, blade_depth, fp);
    let p = par.xy;
    let blade_ao = par.z;
    let natural = m.grass_b.x > 0.5;
    let base = m.grass_color.rgb;
    let dry = m.grass_dry.rgb;
    let wear = m.grass_color.w;
    let wet = m.grass_dry.w;
    let half = m.grass_half.xy;
    // mowing lean direction: blades laid over along the stripe direction give a streaky texture
    let ang = m.grass_a.y;
    let along = vec2<f32>(cos(ang), sin(ang));
    let q = vec2<f32>(dot(p, along), dot(p, vec2<f32>(-along.y, along.x)));
    // field-scale tone (shade of the turf over metres) and dry stress patches
    let n_large = aa_noise(p + 17.3, 0.035, fp) * 0.6 + aa_noise(p, 0.11, fp) * 0.4;
    let n_med = aa_noise(p + 3.1, 0.7, fp) * 0.6 + aa_noise(p - 9.4, 2.3, fp) * 0.4;
    var col = base;
    var albedo_k = 1.0;
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
    // tufts and clumps (5-15 cm): the grain that keeps turf crisp from a few metres to a few tens
    // of metres, between the metre-scale tone and the blade streaks
    let tuft = aa_noise(p + 31.0, 7.5, fp) * 0.55 + aa_noise(p - 12.0, 16.0, fp) * 0.45;
    albedo_k = (0.9 + 0.2 * tuft);
    // blade scale: fine streaks along the mowing direction (anisotropic), tips and dead blades
    let blade = aa_noise(vec2<f32>(q.x * 38.0, q.y * 110.0), 1.0, fp * 110.0)
              * 0.55 + aa_noise(vec2<f32>(q.x * 90.0, q.y * 260.0), 1.0, fp * 260.0) * 0.45;
    let tips = aa_noise(p + 5.7, 55.0, fp);
    let dead = smoothstep(0.86, 0.97, aa_noise(p - 2.3, 85.0, fp));
    // thatch between blades: visible looking down, hidden behind blades at grazing angles
    let cos_view = clamp(view_dir.y, 0.0, 1.0);
    let depth = clamp(m.grass_c.z / 0.025, 0.5, 2.0);
    let gaps = smoothstep(0.30, 0.0, blade) * mix(0.25, 1.0, cos_view) * select(0.6, 1.0, natural);
    let thatch = vec3<f32>(0.035, 0.030, 0.012);
    var albedo = col * (0.86 + 0.30 * blade) * albedo_k;
    albedo = mix(albedo, albedo * vec3<f32>(1.12, 1.10, 0.80) + vec3<f32>(0.006, 0.006, 0.0), smoothstep(0.55, 0.85, tips) * 0.5);
    if (natural) {
        albedo = mix(albedo, dry * 0.9, dead * 0.35);
    }
    albedo = mix(albedo, thatch, gaps * 0.55);
    // seen across the pitch you look into the blade sides, which are a little darker
    albedo *= 1.0 - 0.08 * (1.0 - cos_view) * depth;

    // wear where play happens (kit/assets/pitch.py has the same field): an oval in each goalmouth,
    // the keeper's crease along the goal line, the penalty spots, the centre spot, a faint channel
    // down the middle, and the assistant referees' lanes (each on their own half). Sized for a
    // 105 m pitch and scaled down with smaller ones (youth formats: the play is smaller too, and
    // there are no assistant referees)
    let ks = clamp(half.x / 52.5, 0.35, 1.0);
    var wf = 0.7 * gauss2(p, vec2<f32>(0.0), vec2<f32>(2.2 * ks)) + 0.25 * gauss2(p, vec2<f32>(0.0), vec2<f32>(25.0, 7.0) * ks);
    for (var k = 0; k < 2; k++) {
        let sg = select(-1.0, 1.0, k == 0);
        wf += gauss2(p, vec2<f32>(sg * (half.x - 4.0 * ks), 0.0), vec2<f32>(3.2, 5.0) * ks);
        wf += 1.3 * gauss2(p, vec2<f32>(sg * (half.x - 0.9), 0.0), vec2<f32>(1.0, 3.4 * ks));
        wf += 0.9 * gauss2(p, vec2<f32>(sg * (half.x - 11.0 * ks), 0.0), vec2<f32>(0.9));
        let lane = exp(-0.5 * pow((p.y - sg * (half.y + 1.0)) / 0.7, 2.0)) * smoothstep(0.45, 0.7, aa_noise(p, 0.9, fp));
        wf += 0.45 * lane * smoothstep(-2.0, 2.0, p.x * sg) * smoothstep(0.6, 0.9, ks);
    }
    // a lawn wears evenly in patches (paths of feet, mower turns) instead of in goalmouths
    wf = mix(0.35 * smoothstep(0.55, 0.85, aa_noise(p + 23.0, 0.15, fp)), wf, m.grass_b.y);
    // solid cores, ragged edges: low-frequency breakup only where the wear thins out
    let breakup = aa_noise(p + 7.0, 0.5, fp) * 0.6 + aa_noise(p + 1.0, 2.0, fp) * 0.4;
    let wamt = wear * select(0.25, 1.0, natural);
    let wcore = wf * wamt * 1.35;
    let wmask = clamp(wcore + (breakup - 0.5) * 0.7 * min(wcore * 2.5, 1.0), 0.0, 1.0);
    // bare soil is a muddy grey-brown (sand-dressed in places), thin turf a washed-out straw
    let soil_col = mix(vec3<f32>(0.095, 0.074, 0.048), vec3<f32>(0.16, 0.13, 0.088), aa_noise(p, 6.0, fp));
    let thin = mix(dry, vec3<f32>(dot(dry, vec3<f32>(0.3, 0.55, 0.15))), 0.35);
    albedo = mix(albedo, thin * (0.9 + 0.2 * blade), smoothstep(0.12, 0.55, wmask) * 0.8);
    let soil = smoothstep(0.45, 0.9, wmask);
    albedo = mix(albedo, soil_col, soil);
    if (!natural) {
        // rubber crumb between the fibres: grains a few millimetres across where a pixel resolves
        // them, their mean cover once it doesn't (no band where the noise fades out); seen from
        // above it shows, across the pitch the fibres hide it
        let speck = smoothstep(0.6, 0.66, aa_noise(p, 260.0, fp));
        let crumb = mix(0.16, speck, smoothstep(0.012, 0.004, fp)) * mix(0.3, 1.0, cos_view);
        albedo = mix(albedo, vec3<f32>(0.016, 0.015, 0.014), crumb * 0.55);
    }
    albedo *= 1.0 - 0.3 * wet;
    // inside the blade layer: tips catch the light, the base is deep in shade
    let ao = mix(0.30, 1.0, pow(blade_ao, 0.7));
    albedo *= mix(1.0, ao, 1.0 - soil);
    var out: Grass;
    out.albedo = albedo;
    // wet soil is satin, not a mirror: only scattered puddles in the churned mud get glossy
    let puddle = smoothstep(0.62, 0.85, aa_noise(p + 3.0, 2.3, fp));
    out.rough = mix(mix(0.82, 0.62, wet), mix(0.9, mix(0.78, 0.55, puddle), wet), soil);
    out.soil = soil;
    out.gaps = gaps;
    return out;
}

// What a glossy surface mirrors outdoors: the sky graded to a bright hazy horizon, the distant
// tree line and buildings as an uneven dark band just above it, and the ground below. The horizon
// line is what makes paint, glass and polished metal read as real; rough surfaces see it blurred
// towards the plain upper/lower hemisphere average.
// The light a surface gets from the whole sky: the blue dome, but also the bright haze, sunlit
// clouds and sunlit buildings round about, so shade is a cool grey, not the blue of the zenith
// (a clear sky's light is about 10,000 K: red to blue about 0.7, where the zenith colour is 0.24).
// Same brightness, less colour; the drawn sky and its reflections keep the full blue.
fn sky_light(c: vec3<f32>) -> vec3<f32> {
    if (g.params.z > 0.5) { return c; }
    return mix(c, vec3<f32>(dot(c, vec3<f32>(0.2126, 0.7152, 0.0722))), 0.45);
}

fn env_radiance(r: vec3<f32>, rough: f32) -> vec3<f32> {
    let up = r.y;
    let sky = mix(g.sky_horizon.rgb * 1.15, g.sky_zenith.rgb, pow(clamp(up, 0.0, 1.0), 0.55));
    let below = g.ground.rgb * 0.55;
    let far_line = mix(g.ground.rgb * 0.45, g.sky_horizon.rgb * 0.7, 0.35);
    let az = atan2(r.z, r.x);
    let ridge = 0.02 + 0.025 * (0.5 + 0.5 * sin(az * 7.0 + 1.3) * sin(az * 3.0 + 0.4));
    var c = mix(far_line, sky, smoothstep(ridge * 0.6, ridge, up));
    c = mix(below, c, smoothstep(-0.01, 0.003, up));
    let blur = mix(below, mix(g.sky_horizon.rgb, g.sky_zenith.rgb, 0.4), smoothstep(-0.3, 0.3, up));
    return mix(c, blur, smoothstep(0.15, 0.6, rough));
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
    if (i.icolor.w > 0.5 && i.icolor.w < 1.5 && m.flags2.w > 0.5) { base = vec4<f32>(i.icolor.rgb, 1.0); }
    base *= mix(vec4<f32>(1.0), tb, m.flags.x);
    base = vec4<f32>(base.rgb * i.vcolor.rgb, base.a);   // COLOR_0: baked occlusion and tint
    let foliage = m.kind.x > 0.5;
    if (m.params.z < 0.0) {
        // goal net: alpha is the distance to the nearest cord (net.rs). Draw the cord at its true
        // width, at least a pixel wide with its alpha scaled down to match, and fade to the mean
        // coverage (2 rn) once a pixel spans a fair part of a cell
        // (the pixel footprint comes from the uv derivatives: d's own derivatives flip sign across a
        // cord and would make the width depend on how cords fall on 2x2 pixel quads)
        let rn = -m.params.z;
        let d = tb.a;
        let tuv = xform(i.uv, m.xf_base, m.rot.x) * vec2<f32>(textureDimensions(t_base));
        let texels = max(length(dpdx(tuv)), length(dpdy(tuv)));
        let w = max(texels / 48.0, 1e-4);             // d spans half a cell = 48 texels
        let eff = max(rn, 0.5 * w);
        var cov = clamp((eff - d) / w + 0.5, 0.0, 1.0) * (rn / eff);
        let cells = texels / 96.0;
        cov = mix(cov, min(2.0 * rn, 1.0), smoothstep(0.2, 0.5, cells));
        base.a = m.base_color.a * cov;
    }
    let alpha_mode = m.flags2.x;
    let metallic = m.params.x * mix(1.0, tmr.b, m.flags.y);
    var rough = m.params.y * mix(1.0, tmr.g, m.flags.y);
    var n = normalize(i.normal);
    let v = normalize(g.cam_pos.xyz - i.world);
    // double-sided everywhere; foliage keeps its authored normals (bent out of the crown) so a
    // card shades with the volume it belongs to whichever side faces the camera
    if (dot(n, v) < 0.0 && !foliage) { n = -n; }
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
    if (foliage) { spec = 0.12; }    // leaves: a faint sheen, no mirror of the sky at the crown's rim
    // vehicle paint: a soft metallic base sheen under a mirror-smooth clear coat (added below)
    let coat = m.kind.w > 1.5;
    if (coat) { rough = max(rough, 0.3); }
    let grass = m.flags2.y > 0.5;
    let paint = m.flags2.z > 1.5;
    if (grass || paint) {
        let procedural = m.flags2.z > 0.5 && m.flags2.z < 1.5;
        let gs = grass_surface(i.world, v, fp, base.rgb, grass && !procedural);
        if (grass) {
            // bare soil carries no mowing stripes (kit/assets/pitch.py lays the soil over them too)
            albedo = gs.albedo * mix(stripe_factor(i.world, -v), 1.0, gs.soil);
            rough = gs.rough;
            canopy = 1.0 - gs.soil;
            gaps = gs.gaps;
            spec = mix(0.25, 0.45, gs.soil);
        } else {
            // line paint sprayed onto blades: grass texture shows through, ragged coverage
            let p = vec2<f32>(i.world.x, -i.world.z);
            let cover = 0.78 + 0.18 * aa_noise(p, 9.0, fp) - 0.25 * gs.gaps;
            // the edges are sprayed, not ruled: coverage thins over the outer centimetre or two
            // of the line (v runs 0-1 across it; spots carry v = 0.5), raggedly along its length
            let across = min(i.uv.y, 1.0 - i.uv.y);
            let rag = aa_noise(vec2<f32>(i.uv.x * 11.0, 3.7), 1.0, fp * 11.0) * 0.6 + aa_noise(vec2<f32>(i.uv.x * 37.0, 9.1), 1.0, fp * 37.0) * 0.4;
            let edge = smoothstep(0.0, 0.08 + 0.14 * rag, across);
            albedo = mix(gs.albedo, base.rgb * (0.92 + 0.08 * stripe_factor(i.world, -v)), clamp(cover, 0.0, 1.0) * edge);
            rough = 0.75;
            canopy = 0.6;
            spec = 0.4;
        }
        // blades tilt every which way: a little normal jitter breaks the flat-plane look
        let pj = vec2<f32>(i.world.x, -i.world.z);
        let jx = aa_noise(pj, 23.0, fp) - 0.5;
        let jz = aa_noise(pj + 11.0, 23.0, fp) - 0.5;
        // (bare soil: clods and studs marks break the highlight up the same way)
        let jitter = select(0.35 * canopy, 0.35 * canopy + 0.3 * (1.0 - canopy), grass);
        n = normalize(n + vec3<f32>(jx, 0.0, jz) * jitter);
    }
    // instance variation: trees and hedges drift in hue and depth of green between individuals
    // (some yellower, some bluer, some darker); everything else differs a little in weathering
    if (foliage) {
        let iv2 = fract(i.ivar * 7.31);
        albedo *= mix(0.84, 1.12, i.ivar);
        albedo = mix(albedo, albedo * vec3<f32>(1.08, 1.04, 0.8), smoothstep(0.6, 1.0, iv2) * 0.65);
        albedo = mix(albedo, albedo * vec3<f32>(0.9, 1.0, 1.1), smoothstep(0.4, 0.0, iv2) * 0.5);
    } else if (!grass && !paint && m.kind.z < 0.5 && m.kind.y < 0.5) {
        albedo *= 0.96 + 0.08 * i.ivar;
    }
    if (m.kind.w > 0.5 && m.kind.w < 1.5) {
        // weathered concrete: pour-to-pour tone, rain streaks down vertical faces, and gum and
        // spill stains on treads and walkways
        let p = i.world;
        let tone = aa_noise(vec2<f32>(p.x + 0.7 * p.y, p.z - 0.4 * p.y), 0.18, fp) * 0.6 + aa_noise(vec2<f32>(p.x, p.z) + p.y * 0.3, 0.9, fp) * 0.4;
        albedo *= 0.92 + 0.16 * tone;
        let vert = smoothstep(0.6, 0.2, abs(n_geo.y));
        let streak = aa_noise(vec2<f32>((p.x + p.z) * 3.5, p.y * 0.35), 1.0, fp * 3.5);
        albedo *= 1.0 - 0.2 * vert * smoothstep(0.55, 0.85, streak);
        let up = smoothstep(0.7, 0.9, n_geo.y);
        let gum = smoothstep(0.9, 0.96, aa_noise(vec2<f32>(p.x, p.z), 7.0, fp));
        albedo = mix(albedo, albedo * 0.55, up * gum * 0.6);
    }
    let diffuse = albedo * (1.0 - metallic);
    // (glazing: a double-glazed unit reflects from four faces, about 8 % head on)
    // (solar-control curtain walls are coated: about 22 %)
    let f0 = mix(vec3<f32>(select(select(0.04, 0.085, m.kind.z > 0.5), 0.22, m.kind.z > 1.5)), albedo, metallic);
    var color = vec3<f32>(0.0);

    // sun
    var sun_vis = 0.0;
    let l = normalize(g.sun_dir.xyz);
    let nl = max(dot(n, l), 0.0);
    if (g.sun_dir.w > 0.0) {
        // (foliage looks up its shadow a few centimetres towards the sun, so a card does not
        // darken the leaf layer it lies on: the crown still shades itself as a whole)
        let sh = shadow(i.world + select(vec3<f32>(0.0), l * 0.06, foliage), n_geo) * cloud_shadow(i.world);
        sun_vis = sh;
        var direct = (diffuse + 3.14159265 * ggx(n, v, l, rough, f0) * spec) * nl;
        if (canopy > 0.0) {
            // opposition hotspot: with the sun behind you every visible blade is sunlit
            let g_ang = acos(clamp(dot(v, l), -1.0, 1.0));
            let hotspot = 1.0 + 0.45 * exp(-g_ang / 0.18);
            // self-shadowing inside the canopy away from the hotspot, deeper for long grass
            let selfsh = mix(1.0, mix(0.72, 1.0, exp(-g_ang / 0.9)), clamp(m.grass_c.z / 0.03, 0.4, 1.3));
            // light through the blades when looking towards the sun
            let trans = pow(max(dot(-v, l), 0.0), 6.0) * 0.22;
            direct = diffuse * (nl * hotspot * mix(1.0, selfsh, canopy) + trans * canopy * vec3<f32>(0.9, 1.15, 0.35))
                   + 3.14159265 * ggx(n, v, l, rough, f0) * spec * nl;
        }
        if (foliage) {
            // leaves: wrapped diffuse (light scatters through the crown's outer layer) and light
            // through the leaves when looking towards the sun, yellower for the leaf's thickness
            // (a leaf passes on ~10% of the light, so the glow belongs to views into the sun)
            let wrap = 0.35;
            let ndl = max((dot(n, l) + wrap) / (1.0 + wrap), 0.0);
            let back = pow(max(dot(-v, l), 0.0), 4.0) * 0.45 + 0.04 * max(-dot(n, l), 0.0);
            direct = diffuse * ndl + albedo * vec3<f32>(1.0, 1.12, 0.5) * back
                   + 3.14159265 * ggx(n, v, l, rough, f0) * spec * nl;
        }
        color += direct * g.sun_dir.w * g.sun_color.rgb * sh;
    }
    // sky (blocked by roofs and stands: sky visibility) and bounce from the ground
    let vis = sky_visibility(i.world, n_geo);
    let ny = clamp(n.y, -1.0, 1.0);
    // cosine-weighted sky: an up-facing surface sees zenith and horizon alike
    // (the viewer has no bounce lighting: a stronger sky term and the sunlit ground's bounce stand
    // in for it, so faces turned from the sun read as shade, not as black)
    let sky_amb = sky_light(mix(g.sky_horizon.rgb, g.sky_zenith.rgb, clamp(ny, 0.0, 1.0) * 0.55)) * (0.5 + 0.5 * ny) * 1.4;
    let gnd_amb = g.ground.rgb * (0.5 - 0.5 * ny) * 1.7;
    var amb = sky_amb * mix(0.08, 1.0, vis) + gnd_amb * mix(0.5, 1.0, vis);
    if (n_geo.y > 0.6) { amb *= ground_ao(i.world); }      // the ground beside things standing on it
    amb *= 1.0 - canopy * (0.12 + 0.15 * gaps);       // blades shade each other from the sky
    color += diffuse * amb;
    let r = reflect(-v, n);
    var env = env_radiance(r, rough);
    if (m.kind.z > 0.5 && rough < 0.3) {
        // smooth glazing mirrors the clouds too
        let cl = clouds(r);
        env = mix(env, cl.rgb, cl.a * (1.0 - smoothstep(0.1, 0.3, rough)));
    }
    let fres = f0 + (max(vec3<f32>(1.0 - rough), f0) - f0) * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    // rough surfaces barely mirror the sky; occluded ones not at all (a mirror-smooth one sees
    // along its reflected ray, mostly open sky and horizon, so it is occluded less than its
    // hemisphere suggests)
    let spec_occ = mix(vis * vis, vis, smoothstep(0.35, 0.05, rough)) * clamp(1.0 - rough * 0.9, 0.05, 1.0);
    color += env * fres * spec_occ * spec;
    if (coat) {
        // clear coat: a thin lacquer over the paint mirrors the sky, the clouds, the horizon and
        // the ground beneath it (the dark line along a car's flank), and the sun as a pin-point
        let fc = 0.04 + 0.96 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
        color *= 1.0 - fc;
        var cenv = env_radiance(r, 0.0);
        let cl = clouds(r);
        cenv = mix(cenv, cl.rgb, cl.a);
        color += cenv * fc * mix(vis * vis, vis, 0.85);
        if (g.sun_dir.w > 0.0) {
            color += 3.14159265 * ggx(n, v, l, 0.05, vec3<f32>(0.04)) * nl * g.sun_dir.w * g.sun_color.rgb * sun_vis;
        }
    }
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

    var em = m.emissive.rgb * mix(vec3<f32>(1.0), tem, m.flags.w);
    if (m.kind.y > 0.5) {
        // LED panels: one diode per texel, lit dots on a dark mask; once a pixel spans a few
        // diodes they average out (the mean is kept, so brightness doesn't change with distance)
        let tuv = xform(i.uv, m.xf_emissive, m.rot.w) * vec2<f32>(textureDimensions(t_emissive));
        let texels = max(length(dpdx(tuv)), length(dpdy(tuv)));
        let r = length(fract(tuv) - 0.5);
        let dots = 1.0 - smoothstep(0.26, 0.48, r) * 0.75;
        em *= mix(dots, 0.6, smoothstep(0.25, 0.7, texels)) / 0.6;
    }
    color += em;
    if (i.icolor.w > 1.5) {
        // editor selection: a warm tint and a rim that keeps the outline readable
        let rim = pow(1.0 - clamp(abs(dot(n_geo, v)), 0.0, 1.0), 2.0);
        color = color * 0.75 + i.icolor.rgb * (0.08 + 0.5 * rim);
    }

    color = haze(color, i.world);

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

// aerial perspective: the air between the eye and a surface scatters skylight towards the eye
// (and sunlight, brighter looking towards the sun), so distance fades to the horizon's haze
const HAZE_M: f32 = 9000.0;          // metres for the haze to reach 63 %

fn haze_color(dir: vec3<f32>) -> vec3<f32> {
    let s = max(dot(dir, normalize(g.sun_dir.xyz)), 0.0);
    let sun = select(0.0, 1.0, g.sun_dir.w > 0.0);
    return g.sky_horizon.rgb * 1.6 + g.sun_color.rgb * pow(s, 6.0) * 0.18 * sun;
}

fn haze(color: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let d = world - g.cam_pos.xyz;
    let dist = length(d);
    // thinner with height: denser in the low air over the ground
    let thick = exp(-max(world.y, 0.0) / 600.0);
    let f = 1.0 - exp(-dist / HAZE_M * mix(0.6, 1.0, thick));
    return mix(color, haze_color(d / max(dist, 1e-3)), f);
}

// far field: below the horizon, beyond the scene's own ground, a patchwork of fields and
// hedgerows out to the haze (so the world does not end in a grey band)
fn far_field(p: vec2<f32>) -> vec3<f32> {
    let warp = vec2<f32>(vnoise(p / 700.0), vnoise(p / 700.0 + 17.0)) * 120.0;
    let q = (p + warp) / vec2<f32>(260.0, 190.0);
    let cell = floor(q);
    let f = fract(q);
    let r = hash21(cell);
    var c = vec3<f32>(0.075, 0.12, 0.035);                                  // pasture
    if (r > 0.55) { c = vec3<f32>(0.1, 0.14, 0.04); }                       // fresh grass
    if (r > 0.75) { c = vec3<f32>(0.2, 0.17, 0.07); }                       // ripe cereal
    if (r > 0.88) { c = vec3<f32>(0.11, 0.08, 0.05); }                      // ploughed
    if (r < 0.08) { c = vec3<f32>(0.035, 0.06, 0.025); }                    // woodland
    c *= 0.85 + 0.3 * vnoise(p / 40.0);
    // hedgerows along the field edges: a couple of metres wide, gappy, with clumps of hedgerow
    // trees; none near the scene itself (its own fields and hedges are geometry), where a
    // dead-straight dark line running up to the ground reads as a cast shadow
    let e = min(min(f.x, 1.0 - f.x) * 260.0, min(f.y, 1.0 - f.y) * 190.0);
    let gaps = smoothstep(0.22, 0.34, vnoise(p / 35.0));
    let trees = smoothstep(0.68, 0.82, vnoise(p / 22.0 + 5.0)) * (1.0 - smoothstep(3.0, 12.0, e));
    var near = 0.0;
    if (g.sky_origin.w > 0.0) {
        let ctr = g.sky_origin.xz + g.sky_size.xz * 0.5;
        let rad = length(g.sky_size.xz) * 0.5 + 120.0;
        near = 1.0 - smoothstep(rad, rad + 250.0, length(p - ctr));
    }
    let hedge = (1.0 - smoothstep(0.6, 2.0, e)) * gaps;
    let dark = vec3<f32>(0.03, 0.05, 0.02);
    c = mix(c, dark, max(hedge, trees * 0.85) * (1.0 - near));
    // next to the scene: plain pasture (a field edge with no hedge reads as a seam)
    return mix(c, vec3<f32>(0.08, 0.125, 0.037) * (0.9 + 0.2 * vnoise(p / 40.0)), near);
}

// Fair-weather cumulus on a deck about 1.8 km up: fractal noise for the cloud field, a second
// lookup a little towards the sun for the lit and shaded sides, faded into the horizon haze.
// Returns (colour, coverage).
fn cloud_fbm(q0: vec2<f32>) -> f32 {
    var q = q0;
    var f = 0.0;
    var a = 0.5;
    for (var k = 0; k < 5; k++) {
        f += a * vnoise(q);
        q = q * 2.03 + vec2<f32>(17.1, 9.7);
        a *= 0.5;
    }
    return f;
}

// the cloud field is fixed to the world (kit metres on the 1.8 km deck)
fn cloud_uv(xz: vec2<f32>) -> vec2<f32> {
    return xz * 0.0011 + vec2<f32>(3.0, 7.0);
}

// sunlight let through the cloud deck to a point on the ground: patches of shade drift over the
// fields; thin edges let most light through
fn cloud_shadow(world: vec3<f32>) -> f32 {
    if (g.params.z > 0.5) { return 1.0; }
    let l = normalize(g.sun_dir.xyz);
    if (l.y < 0.05) { return 1.0; }
    let xz = vec2<f32>(world.x, world.z) + vec2<f32>(l.x, l.z) * ((1800.0 - world.y) / l.y);
    let cov = smoothstep(0.5, 0.68, cloud_fbm(cloud_uv(xz)));
    // the sun is out over the ground itself (a match, a session): shade only beyond the scene
    var clear = 0.0;
    if (g.sky_origin.w > 0.0) {
        let ctr = g.sky_origin.xz + g.sky_size.xz * 0.5;
        let rad = length(g.sky_size.xz) * 0.5 + 60.0;
        clear = 1.0 - smoothstep(rad, rad + 250.0, length(vec2<f32>(world.x, world.z) - ctr));
    }
    return 1.0 - 0.72 * cov * (1.0 - clear);
}

fn clouds(dir: vec3<f32>) -> vec4<f32> {
    if (dir.y <= 0.015 || g.params.z > 0.5) { return vec4<f32>(0.0); }
    let t = max(1800.0 - g.cam_pos.y, 50.0) / dir.y;
    let p = cloud_uv(vec2<f32>(g.cam_pos.x, g.cam_pos.z) + vec2<f32>(dir.x, dir.z) * t);
    let f = cloud_fbm(p);
    let cov = smoothstep(0.5, 0.68, f);
    if (cov <= 0.0) { return vec4<f32>(0.0); }
    let sd = normalize(vec2<f32>(g.sun_dir.x, g.sun_dir.z) + vec2<f32>(1e-4));
    let f2 = cloud_fbm(p + sd * 0.06);
    let lit = clamp(0.55 + (f - f2) * 7.0, 0.15, 1.0);
    let base = g.sky_horizon.rgb * 1.25 + vec3<f32>(0.04, 0.04, 0.05);
    let bright = vec3<f32>(1.75, 1.72, 1.66);
    // thick cores go grey underneath, thin edges stay bright and translucent
    let core = smoothstep(0.62, 0.85, f);
    let col = mix(bright, base, (1.0 - lit) * 0.85 + core * 0.25);
    let fade = smoothstep(0.015, 0.16, dir.y);
    return vec4<f32>(col, cov * fade * 0.95);
}

@fragment
fn fs_sky(i: SkyOut) -> @location(0) vec4<f32> {
    let far = g.inv_view_proj * vec4<f32>(i.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - g.cam_pos.xyz);
    let h = dir.y;
    var c = mix(g.sky_horizon.rgb, g.sky_zenith.rgb, pow(clamp(h, 0.0, 1.0), 0.45)) * 1.6;
    let cl = clouds(dir);
    c = mix(c, cl.rgb, cl.a);
    // haze thickens towards the horizon
    c = mix(c, haze_color(dir), exp(-max(h, 0.0) * 18.0) * 0.6);
    if (g.params.z > 0.5) {
        // a floodlit night: never a void. A dark navy zenith, and a warm veil (floodlight and the
        // town's glow scattered in the air) rising from the horizon, strongest just above the
        // stand rims
        let hh = max(h, 0.0);
        c = vec3<f32>(0.0016, 0.0022, 0.0042) + vec3<f32>(0.0105, 0.0092, 0.0078) * exp(-hh * 4.5) + vec3<f32>(0.006, 0.0058, 0.0056) * exp(-hh * 14.0);
    }
    if (h < 0.0) {
        let t = g.cam_pos.y / max(-h, 1e-4);
        let p = g.cam_pos.xyz + dir * t;
        let alb = far_field(vec2<f32>(p.x, p.z));
        let sun = normalize(g.sun_dir.xyz);
        let lit = g.sun_dir.w * g.sun_color.rgb * max(sun.y, 0.0) * cloud_shadow(p) + mix(g.sky_horizon.rgb, g.sky_zenith.rgb, 0.55);
        c = haze(alb * lit, p);
    }
    let s = max(dot(dir, normalize(g.sun_dir.xyz)), 0.0);
    c += g.sun_color.rgb * pow(s, 400.0) * 40.0 * select(0.0, 1.0, g.sun_dir.w > 0.0 && h > 0.0);
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
    o.clip = g.light_view_proj[u32(g.shadow.w)] * model * vec4<f32>(v.pos, 1.0);
    o.uv = v.uv;
    return o;
}

@fragment
fn fs_shadow(i: SOut) {
    let a = m.base_color.a * mix(1.0, textureSample(t_base, s_base, xform(i.uv, m.xf_base, m.rot.x)).a, m.flags.x);
    if (m.params.z < 0.0) {
        // goal net distance field: cords only, widened to about a shadow texel so the lattice
        // survives the filtered lookup as the soft net shadow seen on the grass behind a goal
        if (a > max(-m.params.z, 0.14)) { discard; }
    } else if (m.flags2.x > 0.5 && a < max(m.params.z, 0.35)) { discard; }
}

// editor selection overlay, drawn through everything (no depth test), additively faint
@fragment
fn fs_highlight(i: VOut) -> @location(0) vec4<f32> {
    let n = normalize(i.normal);
    let v = normalize(g.cam_pos.xyz - i.world);
    let rim = pow(1.0 - abs(dot(n, v)), 1.5);
    return vec4<f32>(i.icolor.rgb * 0.9, 0.10 + 0.25 * rim);
}

// depth pre-pass: lay down depth (alpha-tested) so the expensive shading runs once per pixel
@fragment
fn fs_prepass(i: VOut) -> @location(0) vec4<f32> {
    let a = m.base_color.a * mix(1.0, textureSample(t_base, s_base, xform(i.uv, m.xf_base, m.rot.x)).a, m.flags.x);
    if (m.flags2.x > 0.5 && m.flags2.x < 1.5 && a < m.params.z) { discard; }
    return vec4<f32>(0.0);
}

// ---------------------------------------------------------------------------------------------
// grass blades near the camera (render.rs GrassField): real blades instead of the parallax
// height field where a blade is several pixels tall. A top-down ground map, rendered once per
// scene, says where the topmost surface is grass (so paths, kerbs, floors and props mask it), the
// turf's height there, its blade length, colour and mowing lean; each frame a world-anchored grid
// of blades round the point the camera looks at is grown from it, thinning out into the parallax.

struct GMapOut {
    @location(0) a: vec4<f32>,   // ground height (glTF y), blade length (m, 0 = no grass), lean x, lean z
    @location(1) b: vec4<f32>,   // sqrt(albedo), natural (1) or artificial (0)
};

@fragment
fn fs_grassmap(i: VOut) -> GMapOut {
    let tb = textureSample(t_base, s_base, xform(i.uv, m.xf_base, m.rot.x));
    let fp = max(length(dpdx(i.world)), length(dpdy(i.world)));
    var o: GMapOut;
    o.a = vec4<f32>(i.world.y, 0.0, 0.0, 0.0);
    o.b = vec4<f32>(0.0);
    let grass = m.flags2.y > 0.5;
    let paint = m.flags2.z > 1.5;
    if (!(grass || paint)) { return o; }
    let procedural = m.flags2.z > 0.5 && m.flags2.z < 1.5;
    let base = m.base_color.rgb * mix(vec3<f32>(1.0), tb.rgb, m.flags.x);
    let gs = grass_surface(i.world, vec3<f32>(0.0, 1.0, 0.0), fp, base, grass && !procedural);
    var alb = gs.albedo;
    if (paint) { alb = base * 0.95; }
    // mowing lean: grass frame -> kit (x, y) -> glTF (x, -y)
    let mw = mow(i.world);
    let c = m.grass_c.x;
    let sn = m.grass_c.y;
    let kit = vec2<f32>(mw.x * c - mw.y * sn, mw.x * sn + mw.y * c);
    o.a = vec4<f32>(i.world.y, m.grass_c.z * (1.0 - gs.soil), kit.x, -kit.y);
    o.b = vec4<f32>(sqrt(clamp(alb, vec3<f32>(0.0), vec3<f32>(1.0))), select(0.0, 1.0, m.grass_b.x > 0.5));
    return o;
}

struct BladeU {
    map: vec4<f32>,     // map min corner (glTF x, z), map size (x, z) in metres
    grid: vec4<f32>,    // blades per side, spacing (m), radius (m), -
};
@group(2) @binding(0) var<uniform> bu: BladeU;
@group(2) @binding(1) var gmap_a: texture_2d<f32>;
@group(2) @binding(2) var gmap_b: texture_2d<f32>;

struct BOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) t: f32,
    @location(4) side: f32,      // -1..1 across the blade (the midrib is at 0)
    @location(5) across: vec3<f32>,
    @location(6) thin: f32,      // 1 = drawn at its true width; less where widened to a pixel
};

fn gmap_texel(p: vec2<f32>) -> vec2<i32> {
    let uv = (p - bu.map.xy) / bu.map.zw;
    if (any(uv < vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0))) { return vec2<i32>(-1); }
    return vec2<i32>(uv * vec2<f32>(textureDimensions(gmap_a)));
}

@vertex
fn vs_blade(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> BOut {
    var o: BOut;
    o.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);          // outside the view: no blade here
    o.world = vec3<f32>(0.0);
    o.normal = vec3<f32>(0.0, 1.0, 0.0);
    o.color = vec3<f32>(0.0);
    o.t = 0.0;
    o.side = 0.0;
    o.across = vec3<f32>(1.0, 0.0, 0.0);
    o.thin = 1.0;
    let n = u32(bu.grid.x);
    let sp = bu.grid.y;
    let rad = bu.grid.z;
    // the focus: where the view meets the ground, never further than half the radius ahead
    let eye = g.cam_pos.xyz;
    let f4 = g.inv_view_proj * vec4<f32>(0.0, 0.0, 1.0, 1.0);
    let fwd = normalize(f4.xyz / f4.w - eye);
    let fxz = normalize(vec2<f32>(fwd.x, fwd.z) + vec2<f32>(1e-6));
    var gy = 0.0;
    let te = gmap_texel(vec2<f32>(eye.x, eye.z));
    if (te.x >= 0) { gy = textureLoad(gmap_a, te, 0).x; }
    let above = max(eye.y - gy, 0.2);
    var reach = rad * 0.5;
    if (fwd.y < -0.01) { reach = min(above / -fwd.y * length(vec2<f32>(fwd.x, fwd.z)), rad * 0.5); }
    let focus = vec2<f32>(eye.x, eye.z) + fxz * reach;
    // a world-anchored grid (blades stay put as the camera moves), one jittered blade per cell
    let snap = floor(focus / sp) * sp;
    let cell = snap + (vec2<f32>(f32(ii % n), f32(ii / n)) - f32(n) * 0.5) * sp;
    let h1 = hash21(cell * 17.31 + 0.5);
    let h2 = hash21(cell * 7.77 + 3.1);
    let h3 = hash21(cell * 3.31 + 9.2);
    let h4 = hash21(cell * 11.13 + 1.7);
    let p = cell + vec2<f32>(h1, h2) * sp;
    // thin out towards the edge of the field (the parallax canopy carries on beyond)
    let keep = 1.0 - smoothstep(rad * 0.45, rad, length(p - focus));
    if (h3 > keep) { return o; }
    let tc = gmap_texel(p);
    if (tc.x < 0) { return o; }
    let a = textureLoad(gmap_a, tc, 0);
    if (a.y < 0.001) { return o; }
    let b = textureLoad(gmap_b, tc, 0);
    let natural = b.w > 0.5;
    // the blade: length varies a lot blade to blade; ryegrass leaves are 2-4 mm wide, 3G fibres
    // flat ribbons about 1.5 mm across
    let hb = a.y * (0.5 + 0.8 * h4) * mix(0.35, 1.0, keep);
    var wb = select(0.0016, 0.0032, natural) * (0.7 + 0.6 * h1);
    let yaw = h2 * 6.2831853 + h1 * 3.0;
    let across = vec3<f32>(cos(yaw), 0.0, sin(yaw));
    // never thinner than about a pixel: a sub-pixel blade only lands on the odd pixel and the
    // sward breaks into dark speckle a few metres out (worst for 1.5 mm 3G fibres); a widened
    // blade stands for the few around it, so it is shaded flatter (see fs_blade)
    let rootc = g.view_proj * vec4<f32>(p.x, a.x, p.y, 1.0);
    let p11 = length(vec3<f32>(g.view_proj[0][1], g.view_proj[1][1], g.view_proj[2][1]));   // the projection's y scale
    let px_m = 2.0 * rootc.w / max(p11 * g.screen.y, 1.0);   // metres per pixel at the root
    let wmin = 0.9 * px_m;
    // (towards the edge of the field too, so the last blades melt into the turf beyond)
    o.thin = min(clamp(wb / wmin, 0.0, 1.0), smoothstep(0.0, 0.6, keep));
    wb = max(wb, wmin);
    // lean: mostly each blade's own (they stand up again within hours of mowing), with a gentle
    // lay along the mower's direction that is what makes the stripes; 3G fibres splay every way
    let own = vec2<f32>(-across.z, across.x) * (h3 * 2.0 - 1.0);
    var lv = a.zw * 0.35 + own * 0.7;
    if (!natural) { lv = a.zw * 0.3 + own * 1.1; }
    let bend = clamp(0.12 + 0.5 * length(lv) * (0.4 + h4), 0.0, 0.8);
    let ld = vec3<f32>(lv.x, 0.0, lv.y) / max(length(lv), 1e-4);
    // 15 vertices: two quads up the blade and a pointed tip
    var rows = array<u32, 15>(0u, 0u, 1u, 0u, 1u, 1u, 1u, 1u, 2u, 1u, 2u, 2u, 2u, 2u, 3u);
    var sides = array<f32, 15>(-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 0.0);
    var ts = array<f32, 4>(0.0, 0.42, 0.78, 1.0);
    let t = ts[rows[vi]];
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let root = vec3<f32>(p.x, a.x - 0.004, p.y);
    let pos = root + up * hb * t * (1.0 - 0.55 * bend * t) + ld * hb * bend * t * t
            + across * sides[vi] * wb * 0.5 * (1.0 - pow(t, 1.5));
    let tangent = up * hb * (1.0 - 1.1 * bend * t) + ld * hb * 2.0 * bend * t;
    o.normal = normalize(cross(tangent, across));
    o.world = pos;
    o.clip = g.view_proj * vec4<f32>(pos, 1.0);
    // colour: the turf's own (pitch wear, stripes, lawn tone) with blade-to-blade variation, a few
    // dead straw blades among the green
    var col = b.rgb * b.rgb * (0.82 + 0.36 * h1);
    col *= vec3<f32>(1.0 + 0.06 * (h2 - 0.5), 1.0, 1.0 - 0.12 * (h3 - 0.5));
    if (natural && h4 > 0.985) { col = mix(col, vec3<f32>(0.20, 0.17, 0.09), 0.5 + 0.3 * h1); }
    o.color = col;
    o.t = t;
    o.side = sides[vi];
    o.across = across;
    o.thin = clamp(o.thin, 0.0, 1.0);
    return o;
}

@fragment
fn fs_blade(i: BOut) -> @location(0) vec4<f32> {
    let v = normalize(g.cam_pos.xyz - i.world);
    var nf = normalize(i.normal);
    if (dot(nf, v) < 0.0) { nf = -nf; }
    // the leaf is folded along its midrib (a shallow V), and catches light like a soft card
    let crease = normalize(nf - i.across * i.side * 0.45 * sign(dot(nf, cross(i.across, vec3<f32>(0.0, 1.0, 0.0)) + vec3<f32>(1e-4))));
    let n = normalize(crease + vec3<f32>(0.0, 0.55, 0.0));
    let l = normalize(g.sun_dir.xyz);
    // deep in the sward the blades shade one another; the sheath at the base is paler, the
    // mid-blade the deepest green, and mown tips show their pale cut ends
    // (a blade widened to a pixel shows less of its dark base: it averages the sward around it)
    // (td: how deep in the sward this point is, eased off for a widened or fading blade)
    let td = mix(0.85, i.t, i.thin);
    var alb = i.color * 1.12 * mix(0.38, 1.0, pow(td, 0.7));
    alb = mix(alb, alb * vec3<f32>(1.15, 1.12, 0.85), (1.0 - smoothstep(0.0, 0.25, i.t)) * 0.4);
    alb = mix(alb, alb * vec3<f32>(1.1, 1.08, 0.78), smoothstep(0.75, 1.0, i.t) * 0.35);
    alb = mix(alb, vec3<f32>(0.42, 0.40, 0.25) * dot(i.color, vec3<f32>(1.0)) * 2.0, smoothstep(0.93, 1.0, i.t) * 0.5);
    // the midrib is a touch lighter than the leaf either side
    alb *= 1.0 + 0.12 * (1.0 - abs(i.side));
    var col = vec3<f32>(0.0);
    if (g.sun_dir.w > 0.0) {
        let sh = shadow(i.world, up_or(nf)) * cloud_shadow(i.world);
        let nl = max(dot(n, l), 0.0);
        // light through the leaf (it passes ~10%), strongest looking towards the sun
        // (a face turned from the sun still glows: it is lit from behind through the leaf)
        let through = pow(max(dot(-v, l), 0.0), 4.0) * 0.4 + 0.32 * max(-dot(nf, l), 0.0);
        let sheen = 3.14159265 * ggx(nf, v, l, 0.42, vec3<f32>(0.04)) * 0.35 * max(dot(nf, l), 0.0);
        col += (alb * nl + alb * vec3<f32>(0.95, 1.12, 0.45) * through + vec3<f32>(sheen)) * g.sun_dir.w * g.sun_color.rgb * sh * mix(0.55, 1.0, td);
    }
    let vis = sky_visibility(i.world, vec3<f32>(0.0, 1.0, 0.0));
    let sky_amb = sky_light(mix(g.sky_horizon.rgb, g.sky_zenith.rgb, 0.4)) * 1.1;
    let gnd_amb = g.ground.rgb * 0.5;
    col += alb * (sky_amb * mix(0.08, 1.0, vis) + gnd_amb) * mix(0.45, 1.0, td) * ground_ao(i.world);
    // waxy leaves (and polyethylene fibres) mirror a little of the sky at grazing angles, as the
    // turf beneath does: without it the blades read darker and yellower than the ground they stand on
    let fr = 0.04 + 0.96 * pow(1.0 - abs(dot(nf, v)), 5.0);
    col += env_radiance(reflect(-v, nf), 0.7) * fr * 0.6 * vis * mix(0.5, 1.0, i.t);
    if (g.grid_origin.w > 0.0 && g.params.z > 0.5) {
        col += alb * grid_irradiance(i.world, n);
    }
    return vec4<f32>(haze(col, i.world), 1.0);
}

// shadow lookups offset along a mostly-up normal (a blade's own normal would push the sample
// sideways into its neighbours)
fn up_or(n: vec3<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(0.0, 1.0, 0.0) + n * 0.3);
}
