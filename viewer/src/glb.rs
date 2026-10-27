//! Minimal GLB reader for Stadium Kit exports: meshes, PBR materials (with KHR_texture_transform,
//! emissive strength, transmission, alpha modes), embedded PNG images, node hierarchy,
//! EXT_mesh_gpu_instancing (with the kit's _SEAT_COLOR attribute) and KHR_lights_punctual spots.

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Copy, Default)]
pub struct TexXform {
    pub offset: Vec2,
    pub scale: Vec2,
    pub rotation: f32,
}

#[derive(Clone, Copy)]
pub struct TexRef {
    pub image: usize,
    pub repeat: bool,
    pub xform: TexXform,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Alpha {
    Opaque,
    Mask,
    Blend,
}

#[derive(Clone)]
pub struct Material {
    pub name: String,
    pub base_color: Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: Vec3,
    pub emissive_strength: f32,
    pub base_tex: Option<TexRef>,
    pub mr_tex: Option<TexRef>,
    pub normal_tex: Option<TexRef>,
    pub emissive_tex: Option<TexRef>,
    pub alpha: Alpha,
    pub cutoff: f32,
    pub transmission: f32,
    /// glTF material extras (tint_gain for runtime-recolourable TINT_ materials)
    pub extras: Value,
    /// procedural grass and line paint: the turf and its pitch frame (see GrassParams)
    pub grass: Option<GrassParams>,
}

/// Per-material grass uniforms (shader.wgsl MaterialU.grass_*): every pitch and lawn carries its
/// own turf and frame, so a site can hold many pitches facing different ways.
#[derive(Clone, Copy, Debug, Default)]
pub struct GrassParams {
    /// stripe width, angle (rad), strength, pattern id
    pub a: [f32; 4],
    /// natural (1) or woven artificial (0), wear zones on, frame origin x, y (kit metres)
    pub b: [f32; 4],
    /// cos yaw, sin yaw, blade height (m), -
    pub c: [f32; 4],
    /// base colour (linear), w = wear
    pub color: [f32; 4],
    /// dry colour (linear), w = wetness
    pub dry: [f32; 4],
    /// half length, half width (m)
    pub half: [f32; 4],
}

pub fn pattern_id(p: &str) -> f32 {
    match p {
        "stripes" => 1.0,
        "stripes_long" => 2.0,
        "checkerboard" => 3.0,
        "diagonal" => 4.0,
        "diamond" => 5.0,
        "circles" => 6.0,
        _ => 0.0,
    }
}

impl GrassParams {
    pub fn from_pitch(p: &PitchInfo, origin: Vec2, yaw: f32, wear_zones: bool) -> Self {
        GrassParams {
            a: [p.width, p.angle.to_radians(), p.strength, pattern_id(&p.pattern)],
            b: [p.natural as u8 as f32, wear_zones as u8 as f32, origin.x, origin.y],
            c: [yaw.cos(), yaw.sin(), p.height_mm / 1000.0, 0.0],
            color: p.color.extend(p.wear).to_array(),
            dry: p.dry.extend(p.wetness).to_array(),
            half: [p.half.x, p.half.y, 0.0, 0.0],
        }
    }
}

pub struct Primitive {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// COLOR_0 (linear rgba, multiplies the base colour); empty = white
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub material: usize,
}

#[derive(Clone, Copy)]
pub struct Instance {
    pub model: Mat4,
    pub color: Vec4, // w = 1: override base colour (seat colours)
}

pub struct Light {
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub candela: f32,
    pub inner: f32,
    pub outer: f32,
}

pub struct Image {
    /// unique per decoded image: the renderer caches GPU textures by it across scene rebuilds
    pub id: u64,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

static NEXT_IMAGE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl Image {
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        Image { id: NEXT_IMAGE.fetch_add(1, std::sync::atomic::Ordering::Relaxed), width, height, rgba }
    }
}

pub struct PitchInfo {
    pub pattern: String,
    pub width: f32,
    pub angle: f32,
    pub strength: f32,
    pub natural: bool,
    /// grass colours (linear), wear, wetness, blade height and pitch half-size, for the shader
    pub color: Vec3,
    pub dry: Vec3,
    pub wear: f32,
    pub wetness: f32,
    pub height_mm: f32,
    pub half: Vec2,
}

pub fn srgb_hex(h: &str) -> Vec3 {
    let h = h.trim_start_matches('#');
    let c = |i: usize| u8::from_str_radix(h.get(i..i + 2).unwrap_or("80"), 16).unwrap_or(128) as f32 / 255.0;
    let lin = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    Vec3::new(lin(c(0)), lin(c(2)), lin(c(4)))
}

pub struct Scene {
    pub meshes: Vec<Vec<Primitive>>,
    pub instances: Vec<Vec<Instance>>, // per mesh
    pub materials: Vec<Material>,
    pub images: Vec<std::sync::Arc<Image>>,
    pub lights: Vec<Light>,
    pub bounds: (Vec3, Vec3),
    pub pitch: Option<PitchInfo>,
    pub capacity: Option<u64>,
    pub main_camera: Option<Vec3>,
    /// (base mesh, coarse mesh) pairs from MSFT_lod: the coarse mesh draws the far instances
    pub lods: Vec<(usize, usize)>,
    /// every mesh node: (node name, name of its top-level ancestor, mesh, world transform)
    pub nodes: Vec<(String, String, usize, Mat4)>,
}

struct Doc<'a> {
    json: Value,
    bin: &'a [u8],
}

impl<'a> Doc<'a> {
    fn accessor(&self, idx: usize) -> (Vec<f32>, usize) {
        let acc = &self.json["accessors"][idx];
        let count = acc["count"].as_u64().unwrap() as usize;
        let comps = match acc["type"].as_str().unwrap() {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT4" => 16,
            t => panic!("unsupported accessor type {t}"),
        };
        let ctype = acc["componentType"].as_u64().unwrap();
        let normalized = acc["normalized"].as_bool().unwrap_or(false);
        let mut out = vec![0.0f32; count * comps];
        let Some(bv_idx) = acc["bufferView"].as_u64() else { return (out, comps) };
        let bv = &self.json["bufferViews"][bv_idx as usize];
        let base = bv["byteOffset"].as_u64().unwrap_or(0) as usize + acc["byteOffset"].as_u64().unwrap_or(0) as usize;
        let csize = match ctype {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            _ => panic!("component type {ctype}"),
        };
        let stride = bv["byteStride"].as_u64().map(|s| s as usize).unwrap_or(csize * comps);
        for i in 0..count {
            for c in 0..comps {
                let o = base + i * stride + c * csize;
                let b = &self.bin[o..o + csize];
                out[i * comps + c] = match ctype {
                    5126 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                    5125 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
                    5123 => {
                        let v = u16::from_le_bytes([b[0], b[1]]) as f32;
                        if normalized { v / 65535.0 } else { v }
                    }
                    5121 => {
                        let v = b[0] as f32;
                        if normalized { v / 255.0 } else { v }
                    }
                    5122 => i16::from_le_bytes([b[0], b[1]]) as f32,
                    5120 => b[0] as i8 as f32,
                    _ => 0.0,
                };
            }
        }
        (out, comps)
    }

    fn indices(&self, idx: usize) -> Vec<u32> {
        let acc = &self.json["accessors"][idx];
        let count = acc["count"].as_u64().unwrap() as usize;
        let ctype = acc["componentType"].as_u64().unwrap();
        let bv = &self.json["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
        let base = bv["byteOffset"].as_u64().unwrap_or(0) as usize + acc["byteOffset"].as_u64().unwrap_or(0) as usize;
        (0..count)
            .map(|i| match ctype {
                5125 => {
                    let o = base + i * 4;
                    u32::from_le_bytes(self.bin[o..o + 4].try_into().unwrap())
                }
                5123 => {
                    let o = base + i * 2;
                    u16::from_le_bytes(self.bin[o..o + 2].try_into().unwrap()) as u32
                }
                _ => self.bin[base + i] as u32,
            })
            .collect()
    }
}

fn tex_ref(doc: &Value, info: &Value) -> Option<TexRef> {
    let tex_idx = info["index"].as_u64()? as usize;
    let tex = &doc["textures"][tex_idx];
    let image = tex["source"].as_u64()? as usize;
    let repeat = match tex["sampler"].as_u64() {
        Some(s) => doc["samplers"][s as usize]["wrapS"].as_u64().unwrap_or(10497) == 10497,
        None => true,
    };
    let mut xform = TexXform { offset: Vec2::ZERO, scale: Vec2::ONE, rotation: 0.0 };
    if let Some(t) = info["extensions"].get("KHR_texture_transform") {
        if let Some(o) = t["offset"].as_array() {
            xform.offset = Vec2::new(o[0].as_f64().unwrap() as f32, o[1].as_f64().unwrap() as f32);
        }
        if let Some(s) = t["scale"].as_array() {
            xform.scale = Vec2::new(s[0].as_f64().unwrap() as f32, s[1].as_f64().unwrap() as f32);
        }
        xform.rotation = t["rotation"].as_f64().unwrap_or(0.0) as f32;
    }
    Some(TexRef { image, repeat, xform })
}

fn f(v: &Value, d: f32) -> f32 {
    v.as_f64().map(|x| x as f32).unwrap_or(d)
}

fn node_local(n: &Value) -> Mat4 {
    if let Some(m) = n["matrix"].as_array() {
        let a: Vec<f32> = m.iter().map(|x| x.as_f64().unwrap() as f32).collect();
        return Mat4::from_cols_slice(&a);
    }
    let t = n["translation"].as_array().map(|a| Vec3::new(f(&a[0], 0.0), f(&a[1], 0.0), f(&a[2], 0.0))).unwrap_or(Vec3::ZERO);
    let r = n["rotation"].as_array().map(|a| Quat::from_xyzw(f(&a[0], 0.0), f(&a[1], 0.0), f(&a[2], 0.0), f(&a[3], 1.0))).unwrap_or(Quat::IDENTITY);
    let s = n["scale"].as_array().map(|a| Vec3::new(f(&a[0], 1.0), f(&a[1], 1.0), f(&a[2], 1.0))).unwrap_or(Vec3::ONE);
    Mat4::from_scale_rotation_translation(s, r, t)
}

pub fn load(path: &std::path::Path) -> Result<Scene, String> {
    let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if data.len() < 20 || &data[0..4] != b"glTF" {
        return Err("not a .glb file".into());
    }
    let jlen = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let json: Value = serde_json::from_slice(&data[20..20 + jlen]).map_err(|e| e.to_string())?;
    let boff = 20 + jlen;
    let blen = u32::from_le_bytes(data[boff..boff + 4].try_into().unwrap()) as usize;
    let bin = &data[boff + 8..boff + 8 + blen];
    let doc = Doc { json, bin };
    let j = &doc.json;

    // images
    let mut images = Vec::new();
    for im in j["images"].as_array().cloned().unwrap_or_default() {
        let bv = &j["bufferViews"][im["bufferView"].as_u64().unwrap() as usize];
        let o = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
        let l = bv["byteLength"].as_u64().unwrap() as usize;
        let decoded = image::load_from_memory(&bin[o..o + l]).map_err(|e| e.to_string())?.to_rgba8();
        images.push(std::sync::Arc::new(Image::new(decoded.width(), decoded.height(), decoded.into_raw())));
    }

    // materials
    let mut materials = Vec::new();
    for m in j["materials"].as_array().cloned().unwrap_or_default() {
        let pbr = &m["pbrMetallicRoughness"];
        let bc = pbr["baseColorFactor"].as_array().map(|a| Vec4::new(f(&a[0], 1.0), f(&a[1], 1.0), f(&a[2], 1.0), f(&a[3], 1.0))).unwrap_or(Vec4::ONE);
        let em = m["emissiveFactor"].as_array().map(|a| Vec3::new(f(&a[0], 0.0), f(&a[1], 0.0), f(&a[2], 0.0))).unwrap_or(Vec3::ZERO);
        let ext = &m["extensions"];
        materials.push(Material {
            name: m["name"].as_str().unwrap_or("").to_string(),
            base_color: bc,
            metallic: f(&pbr["metallicFactor"], 1.0),
            roughness: f(&pbr["roughnessFactor"], 1.0),
            emissive: em,
            emissive_strength: f(&ext["KHR_materials_emissive_strength"]["emissiveStrength"], 1.0),
            base_tex: tex_ref(j, &pbr["baseColorTexture"]),
            mr_tex: tex_ref(j, &pbr["metallicRoughnessTexture"]),
            normal_tex: tex_ref(j, &m["normalTexture"]),
            emissive_tex: tex_ref(j, &m["emissiveTexture"]),
            alpha: match m["alphaMode"].as_str() {
                Some("MASK") => Alpha::Mask,
                Some("BLEND") => Alpha::Blend,
                _ => Alpha::Opaque,
            },
            cutoff: f(&m["alphaCutoff"], 0.5),
            transmission: f(&ext["KHR_materials_transmission"]["transmissionFactor"], 0.0),
            extras: m["extras"].clone(),
            grass: None,
        });
    }
    if materials.is_empty() {
        materials.push(Material {
            name: "default".into(), base_color: Vec4::ONE, metallic: 0.0, roughness: 0.6, emissive: Vec3::ZERO,
            emissive_strength: 1.0, base_tex: None, mr_tex: None, normal_tex: None, emissive_tex: None,
            alpha: Alpha::Opaque, cutoff: 0.5, transmission: 0.0, extras: Value::Null, grass: None,
        });
    }

    // meshes
    let mut meshes = Vec::new();
    for mesh in j["meshes"].as_array().cloned().unwrap_or_default() {
        let mut prims = Vec::new();
        for p in mesh["primitives"].as_array().cloned().unwrap_or_default() {
            if p["mode"].as_u64().unwrap_or(4) != 4 {
                continue;
            }
            let a = &p["attributes"];
            let (pos, _) = doc.accessor(a["POSITION"].as_u64().unwrap() as usize);
            let n = pos.len() / 3;
            let nrm = a["NORMAL"].as_u64().map(|i| doc.accessor(i as usize).0).unwrap_or_else(|| vec![0.0, 1.0, 0.0].repeat(n));
            let uv = a["TEXCOORD_0"].as_u64().map(|i| doc.accessor(i as usize).0).unwrap_or_else(|| vec![0.0; n * 2]);
            let colors: Vec<[f32; 4]> = match a["COLOR_0"].as_u64().map(|i| doc.accessor(i as usize)) {
                Some((c, 3)) => c.chunks(3).map(|c| [c[0], c[1], c[2], 1.0]).collect(),
                Some((c, 4)) => c.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect(),
                _ => Vec::new(),
            };
            let indices = match p["indices"].as_u64() {
                Some(i) => doc.indices(i as usize),
                None => (0..n as u32).collect(),
            };
            prims.push(Primitive {
                positions: pos.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
                normals: nrm.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
                uvs: uv.chunks(2).map(|c| [c[0], c[1]]).collect(),
                colors,
                indices,
                material: p["material"].as_u64().unwrap_or(0) as usize,
            });
        }
        meshes.push(prims);
    }

    // nodes: world transforms, instances, lights
    let mut instances: Vec<Vec<Instance>> = vec![Vec::new(); meshes.len()];
    let mut lights = Vec::new();
    let mut pitch = None;
    let mut capacity = None;
    let mut main_camera = None;
    let mut node_list = Vec::new();
    let nodes = j["nodes"].as_array().cloned().unwrap_or_default();
    let mut parent: HashMap<usize, usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        for c in n["children"].as_array().cloned().unwrap_or_default() {
            parent.insert(c.as_u64().unwrap() as usize, i);
        }
    }
    let mut world_cache: HashMap<usize, Mat4> = HashMap::new();
    fn world(i: usize, nodes: &[Value], parent: &HashMap<usize, usize>, cache: &mut HashMap<usize, Mat4>) -> Mat4 {
        if let Some(m) = cache.get(&i) {
            return *m;
        }
        let local = node_local(&nodes[i]);
        let w = match parent.get(&i) {
            Some(&p) => world(p, nodes, parent, cache) * local,
            None => local,
        };
        cache.insert(i, w);
        w
    }
    let light_defs = j["extensions"]["KHR_lights_punctual"]["lights"].as_array().cloned().unwrap_or_default();
    // MSFT_lod targets are detached from the scene: never draw them directly
    let mut lod_targets = std::collections::HashSet::new();
    let mut lods = Vec::new();
    for n in &nodes {
        if let (Some(base), Some(ids)) = (n["mesh"].as_u64(), n["extensions"]["MSFT_lod"]["ids"].as_array()) {
            for id in ids {
                let id = id.as_u64().unwrap() as usize;
                lod_targets.insert(id);
                if let Some(lm) = nodes[id]["mesh"].as_u64() {
                    lods.push((base as usize, lm as usize));
                }
            }
        }
    }
    for (i, n) in nodes.iter().enumerate() {
        if lod_targets.contains(&i) {
            continue;
        }
        let w = world(i, &nodes, &parent, &mut world_cache);
        let extras = &n["extras"];
        if extras["asset_family"].as_str() == Some("pitch") {
            if let Some(params) = extras["params"].as_str().and_then(|s| serde_json::from_str::<Value>(s).ok()) {
                let mow = &params["mow"];
                let grass = &params["_grass"];
                let natural = grass["natural"].as_bool().unwrap_or(!params["grass"].as_str().unwrap_or("").starts_with("artificial"));
                pitch = Some(PitchInfo {
                    pattern: mow["pattern"].as_str().unwrap_or("stripes").to_string(),
                    width: f(&mow["width"], 5.25),
                    angle: f(&mow["angle"], 0.0),
                    strength: f(&grass["stripe_strength"], 0.2) * (f(&params["height_mm"], 25.0) / 25.0).clamp(0.4, 1.6),
                    natural,
                    color: srgb_hex(grass["color"].as_str().unwrap_or("#4a6d27")),
                    dry: srgb_hex(grass["color_dry"].as_str().unwrap_or("#8a8247")),
                    wear: f(&params["wear"], 0.25),
                    wetness: f(&params["wetness"], 0.0),
                    height_mm: f(&params["height_mm"], 25.0),
                    half: Vec2::new(f(&params["length"], 105.0) / 2.0, f(&params["width"], 68.0) / 2.0),
                });
            }
        }
        if n["name"].as_str() == Some("SOCKET:camera:main") {
            main_camera = Some(w.transform_point3(Vec3::ZERO));
        }
        if extras["asset_family"].as_str() == Some("stands") {
            capacity = extras["capacity"].as_u64();
        }
        if let Some(li) = n["extensions"]["KHR_lights_punctual"]["light"].as_u64() {
            let d = &light_defs[li as usize];
            if d["type"].as_str() == Some("spot") {
                let col = d["color"].as_array().map(|a| Vec3::new(f(&a[0], 1.0), f(&a[1], 1.0), f(&a[2], 1.0))).unwrap_or(Vec3::ONE);
                lights.push(Light {
                    position: w.transform_point3(Vec3::ZERO),
                    direction: w.transform_vector3(-Vec3::Z).normalize(),
                    color: col,
                    candela: f(&d["intensity"], 1.0),
                    inner: f(&d["spot"]["innerConeAngle"], 0.0),
                    outer: f(&d["spot"]["outerConeAngle"], 0.785),
                });
            }
        }
        let Some(mi) = n["mesh"].as_u64() else { continue };
        let mi = mi as usize;
        let mut top = i;
        while let Some(&p) = parent.get(&top) {
            top = p;
        }
        node_list.push((n["name"].as_str().unwrap_or("").to_string(), nodes[top]["name"].as_str().unwrap_or("").to_string(), mi, w));
        if let Some(gi) = n["extensions"].get("EXT_mesh_gpu_instancing") {
            let at = &gi["attributes"];
            let (t, _) = doc.accessor(at["TRANSLATION"].as_u64().unwrap() as usize);
            let r = at["ROTATION"].as_u64().map(|i| doc.accessor(i as usize).0);
            let s = at["SCALE"].as_u64().map(|i| doc.accessor(i as usize).0);
            let c = at["_SEAT_COLOR"].as_u64().map(|i| doc.accessor(i as usize).0);
            let count = t.len() / 3;
            for k in 0..count {
                let tr = Vec3::new(t[3 * k], t[3 * k + 1], t[3 * k + 2]);
                let rot = r.as_ref().map(|r| Quat::from_xyzw(r[4 * k], r[4 * k + 1], r[4 * k + 2], r[4 * k + 3])).unwrap_or(Quat::IDENTITY);
                let sc = s.as_ref().map(|s| Vec3::new(s[3 * k], s[3 * k + 1], s[3 * k + 2])).unwrap_or(Vec3::ONE);
                let color = c.as_ref().map(|c| Vec4::new(c[3 * k], c[3 * k + 1], c[3 * k + 2], 1.0)).unwrap_or(Vec4::ZERO);
                instances[mi].push(Instance { model: w * Mat4::from_scale_rotation_translation(sc, rot, tr), color });
            }
        } else {
            instances[mi].push(Instance { model: w, color: Vec4::ZERO });
        }
    }

    // bounds of everything drawn
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for (mi, prims) in meshes.iter().enumerate() {
        if instances[mi].is_empty() {
            continue;
        }
        let (mut plo, mut phi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in prims {
            for v in &p.positions {
                let v = Vec3::from(*v);
                plo = plo.min(v);
                phi = phi.max(v);
            }
        }
        // transformed AABB of every instance (approximate with the 8 corners for the first and last)
        for inst in instances[mi].iter().step_by((instances[mi].len() / 64).max(1)) {
            for k in 0..8 {
                let c = Vec3::new(
                    if k & 1 == 0 { plo.x } else { phi.x },
                    if k & 2 == 0 { plo.y } else { phi.y },
                    if k & 4 == 0 { plo.z } else { phi.z },
                );
                let w = inst.model.transform_point3(c);
                lo = lo.min(w);
                hi = hi.max(w);
            }
        }
    }
    // the exported pitch's grass and paint take its turf (baked albedo, shader stripes)
    if let Some(p) = &pitch {
        for m in materials.iter_mut().filter(|m| m.name.starts_with("Pitch_Grass") || m.name.starts_with("TINT_Pitch_Paint")) {
            m.grass = Some(GrassParams::from_pitch(p, Vec2::ZERO, 0.0, true));
        }
    }
    Ok(Scene { meshes, instances, materials, images, lights, bounds: (lo, hi), pitch, capacity, main_camera, lods, nodes: node_list })
}

