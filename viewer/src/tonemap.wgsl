// HDR -> display: exposure, then AgX (as the Blender review renders use), sRGB encoding by the
// target format. AgX desaturates bright colours towards white as film does; a plain ACES fit keeps
// sunlit grass a saturated lime, which reads as CG.
// AgX: Troy Sobotka's design; matrices and the contrast polynomial after Benjamin Wrensch's fit.

@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> exposure: vec4<f32>;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> Out {
    let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var o: Out;
    o.clip = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}

fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx(c: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
    let outset = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var v = inset * max(c, vec3<f32>(1e-10));
    v = clamp(log2(v), vec3<f32>(min_ev), vec3<f32>(max_ev));
    v = (v - min_ev) / (max_ev - min_ev);
    v = agx_contrast(v);
    // look: a little extra contrast, like Blender's "AgX - Medium High Contrast"
    let luma = dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
    v = pow(max(v, vec3<f32>(0.0)), vec3<f32>(1.25));
    v = luma + (v - luma) * 1.15;
    v = outset * v;
    return pow(clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));   // linear, for the sRGB target
}

@fragment
fn fs(i: Out) -> @location(0) vec4<f32> {
    let c = textureSample(hdr, samp, i.uv).rgb * exposure.x;
    return vec4<f32>(agx(c), 1.0);
}
