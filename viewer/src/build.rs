//! Turn a generated stadium (stadium_gen::stadium::Stadium, kit coordinates) into a renderable
//! scene (glTF coordinates: Y up, the kit's -Y is +Z) using the asset library.

use crate::glb::{self, Alpha, Image, Instance, Light, Material, PitchInfo, Primitive, Scene, TexRef, TexXform};
use crate::library::Library;
use glam::{DVec3, Mat4, Vec2, Vec3, Vec4};
use stadium_gen::mesh::to_trimesh;
use stadium_gen::stadium::{Mat, NetSpec, Stadium, Tex};
use std::collections::HashMap;
use std::sync::Arc;

/// kit (x, y, z) -> glTF (x, z, -y)
pub fn to_gltf(v: DVec3) -> Vec3 {
    Vec3::new(v.x as f32, v.z as f32, -v.y as f32)
}

fn kit_to_gltf() -> Mat4 {
    Mat4::from_cols(Vec4::new(1.0, 0.0, 0.0, 0.0), Vec4::new(0.0, 0.0, -1.0, 0.0), Vec4::new(0.0, 1.0, 0.0, 0.0), Vec4::W)
}

fn lin(c: DVec3) -> Vec3 {
    let f = |v: f64| (if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }) as f32;
    Vec3::new(f(c.x), f(c.y), f(c.z))
}

fn plain(name: &str, base: Vec3, rough: f32) -> Material {
    Material {
        name: name.into(),
        base_color: base.extend(1.0),
        metallic: 0.0,
        roughness: rough,
        emissive: Vec3::ZERO,
        emissive_strength: 1.0,
        base_tex: None,
        mr_tex: None,
        normal_tex: None,
        emissive_tex: None,
        alpha: Alpha::Opaque,
        cutoff: 0.5,
        transmission: 0.0,
        extras: serde_json::Value::Null,
    }
}

struct Builder<'a> {
    lib: &'a Library,
    st: &'a Stadium,
    scene: Scene,
    mats: HashMap<String, usize>,
    lib_images: HashMap<usize, usize>,
    extra_images: HashMap<u64, usize>,
    lib_meshes: HashMap<(usize, String), usize>,
    pub warnings: Vec<String>,
}

impl<'a> Builder<'a> {
    fn image(&mut self, img: Arc<Image>) -> usize {
        if let Some(&i) = self.extra_images.get(&img.id) {
            return i;
        }
        self.scene.images.push(img.clone());
        let i = self.scene.images.len() - 1;
        self.extra_images.insert(img.id, i);
        i
    }

    fn lib_image(&mut self, i: usize) -> usize {
        if let Some(&k) = self.lib_images.get(&i) {
            return k;
        }
        self.scene.images.push(self.lib.scene.images[i].clone());
        let k = self.scene.images.len() - 1;
        self.lib_images.insert(i, k);
        k
    }

    /// A library material copied into the scene (texture indices remapped).
    fn copy_lib(&mut self, m: &Material) -> Material {
        let mut m = m.clone();
        for t in [&mut m.base_tex, &mut m.mr_tex, &mut m.normal_tex, &mut m.emissive_tex] {
            if let Some(tr) = t.as_mut() {
                tr.image = self.lib_image(tr.image);
            }
        }
        m
    }

    fn material(&mut self, mat: &Mat) -> usize {
        let key = format!("{mat:?}");
        if let Some(&i) = self.mats.get(&key) {
            return i;
        }
        let m = self.make(mat);
        self.scene.materials.push(m);
        let i = self.scene.materials.len() - 1;
        self.mats.insert(key, i);
        i
    }

    fn make(&mut self, mat: &Mat) -> Material {
        let tex = |image: usize, repeat: bool| Some(TexRef { image, repeat, xform: TexXform { offset: Vec2::ZERO, scale: Vec2::ONE, rotation: 0.0 } });
        let mut m = match mat.key.as_str() {
            "Pitch_Grass" => {
                let mut m = plain("Pitch_Grass", Vec3::ONE, 0.72);
                m.extras = serde_json::json!({"procedural": true});
                m
            }
            "TINT_Pitch_Paint" => plain("TINT_Pitch_Paint", Vec3::new(0.82, 0.83, 0.8), 0.6),
            "Ads_LED" | "Ads_Ribbon" | "Screen_LED" => {
                let mut m = plain(&mat.key, Vec3::splat(0.004), 0.25);
                m.emissive = Vec3::ONE;
                let img = match &mat.texture {
                    Some(Tex::BoardAtlas(keys)) => self.lib.board_atlas(keys),
                    Some(Tex::Scoreboard) => self.lib.scoreboard(),
                    _ => Err("no texture".into()),
                };
                match img {
                    Ok(img) => {
                        let i = self.image(img);
                        m.emissive_tex = tex(i, mat.key != "Screen_LED");
                    }
                    Err(e) => self.warnings.push(e),
                }
                m
            }
            "Ads_Print" | "Carpet" => {
                let mut m = plain(&mat.key, Vec3::ONE, if mat.key == "Carpet" { 0.85 } else { 0.55 });
                let img = match &mat.texture {
                    Some(Tex::BoardAtlas(keys)) => self.lib.board_atlas(keys),
                    Some(Tex::Carpet(k)) => self.lib.carpet(k),
                    _ => Err("no texture".into()),
                };
                match img {
                    Ok(img) => {
                        let i = self.image(img);
                        m.base_tex = tex(i, mat.key != "Carpet");
                    }
                    Err(e) => self.warnings.push(e),
                }
                m
            }
            "TINT_net_card" if matches!(mat.texture, Some(Tex::Net(_))) => {
                let Some(Tex::Net(spec)) = &mat.texture else { unreachable!() };
                let lm = self.lib.material("TINT_net_card").cloned().unwrap_or_else(|| plain("TINT_net_card", Vec3::ONE, 0.7));
                let mut m = self.copy_lib(&lm);
                let (img, scale) = crate::net::texture(spec);
                let i = self.image(img);
                m.base_tex = Some(TexRef { image: i, repeat: true, xform: TexXform { offset: Vec2::ZERO, scale: Vec2::new(scale[0], scale[1]), rotation: 0.0 } });
                m.base_color = Vec4::ONE;
                m.alpha = Alpha::Mask;
                return m;
            }
            key => match self.lib.material(key) {
                Some(lm) => {
                    let lm = lm.clone();
                    self.copy_lib(&lm)
                }
                None => {
                    self.warnings.push(format!("material {key} is not in the library"));
                    plain(key, Vec3::splat(0.5), 0.6)
                }
            },
        };
        if let Some(c) = mat.color {
            let c = lin(c);
            let gain = m.extras.get("tint_gain").and_then(|v| v.as_f64());
            let base = m.extras.get("base_hex").and_then(|v| v.as_str()).map(glb::srgb_hex);
            let rgb = match (gain, base) {
                (Some(g), _) => c * g as f32,
                (None, Some(b)) => m.base_color.truncate() * c / b.max(Vec3::splat(1e-4)),
                (None, None) => c,
            };
            m.base_color = rgb.min(Vec3::ONE).extend(m.base_color.w);
        }
        if let Some((c, s)) = mat.emission {
            m.emissive = lin(c);
            m.emissive_strength = s as f32;
        }
        m
    }

    /// Copy a library mesh into the scene with material recolours (and a goal's net card);
    /// cached per (mesh, options).
    fn lib_mesh(&mut self, mesh: usize, tints: &[(String, DVec3)], net: Option<&NetSpec>) -> usize {
        let key = (mesh, format!("{tints:?}{net:?}"));
        if let Some(&i) = self.lib_meshes.get(&key) {
            return i;
        }
        let prims: Vec<Primitive> = self
            .lib
            .prims(mesh)
            .iter()
            .map(|p| {
                let lm = &self.lib.scene.materials[p.material];
                let tint = tints.iter().find(|(n, _)| *n == lm.name).map(|t| t.1);
                let texture = match net {
                    Some(n) if lm.name == "TINT_net_card" => Some(Tex::Net(n.clone())),
                    _ => None,
                };
                let color = if texture.is_some() { None } else { tint };
                let mi = self.material(&Mat { key: lm.name.clone(), color, emission: None, texture });
                Primitive { positions: p.positions.clone(), normals: p.normals.clone(), uvs: p.uvs.clone(), indices: p.indices.clone(), material: mi }
            })
            .collect();
        self.scene.meshes.push(prims);
        self.scene.instances.push(Vec::new());
        let i = self.scene.meshes.len() - 1;
        self.lib_meshes.insert(key, i);
        i
    }
}

/// Build the render scene. Returns the scene and any warnings (missing assets).
pub fn scene(st: &Stadium, lib: &Library) -> (Scene, Vec<String>) {
    let mut b = Builder {
        lib,
        st,
        scene: Scene {
            meshes: Vec::new(),
            instances: Vec::new(),
            materials: Vec::new(),
            images: Vec::new(),
            lights: Vec::new(),
            bounds: (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            pitch: None,
            capacity: Some(st.report.capacity.total as u64),
            main_camera: Some(to_gltf(st.camera.0)),
            lods: Vec::new(),
            nodes: Vec::new(),
        },
        mats: HashMap::new(),
        lib_images: HashMap::new(),
        extra_images: HashMap::new(),
        lib_meshes: HashMap::new(),
        warnings: Vec::new(),
    };
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    // generated parts: triangulate and smooth on every core
    let parts = &b.st.parts;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(parts.len().max(1));
    let mut tms: Vec<Option<stadium_gen::mesh::TriMesh>> = vec![None; parts.len()];
    std::thread::scope(|sc| {
        for (k, chunk) in tms.chunks_mut(parts.len().div_ceil(threads).max(1)).enumerate() {
            let base = k * parts.len().div_ceil(threads).max(1);
            sc.spawn(move || {
                for (j, slot) in chunk.iter_mut().enumerate() {
                    let part = &parts[base + j];
                    *slot = Some(to_trimesh(&part.mesh, part.smooth_deg, 1.0));
                }
            });
        }
    });
    for (part, tm) in parts.iter().zip(tms) {
        let tm = tm.unwrap();
        let mut prims = Vec::new();
        let positions: Vec<[f32; 3]> = tm.pos.iter().map(|p| [p.x, p.z, -p.y]).collect();
        let normals: Vec<[f32; 3]> = tm.nrm.iter().map(|n| [n.x, n.z, -n.y]).collect();
        // Blender UVs have v up; glTF (and image rows) have v down, as the exporter writes them
        let uvs: Vec<[f32; 2]> = tm.uv.iter().map(|u| [u.x, 1.0 - u.y]).collect();
        for p in &positions {
            lo = lo.min(Vec3::from(*p));
            hi = hi.max(Vec3::from(*p));
        }
        for (tag, idx) in &tm.groups {
            let mat = part.mats.get(*tag as usize).or(part.mats.last()).cloned().unwrap_or_else(|| Mat::named("Galvanised"));
            let mi = b.material(&mat);
            // compact each group's vertices so primitives stay small
            let mut remap: HashMap<u32, u32> = HashMap::new();
            let mut pp = Primitive { positions: Vec::new(), normals: Vec::new(), uvs: Vec::new(), indices: Vec::new(), material: mi };
            for &i in idx {
                let j = *remap.entry(i).or_insert_with(|| {
                    pp.positions.push(positions[i as usize]);
                    pp.normals.push(normals[i as usize]);
                    pp.uvs.push(uvs[i as usize]);
                    pp.positions.len() as u32 - 1
                });
                pp.indices.push(j);
            }
            prims.push(pp);
        }
        b.scene.meshes.push(prims);
        b.scene.instances.push(vec![Instance { model: Mat4::IDENTITY, color: Vec4::ZERO }]);
        // names for click-to-select
        b.scene.nodes.push((part.name.clone(), "part".into(), b.scene.meshes.len() - 1, Mat4::IDENTITY));
    }
    // props
    let c = kit_to_gltf();
    let ci = c.inverse();
    for ps in &b.st.props {
        let Some(prop) = lib.props.get(&ps.key) else {
            b.warnings.push(format!("prop {} is not in the library", ps.key));
            continue;
        };
        let lod = lib.props.get(&format!("{}.lod2", ps.key));
        for (k, (mesh, local)) in prop.parts.iter().enumerate() {
            let mi = b.lib_mesh(*mesh, &ps.tints, ps.net.as_ref());
            if !b.scene.nodes.iter().any(|n| n.2 == mi) {
                b.scene.nodes.push((ps.key.clone(), "prop".into(), mi, Mat4::IDENTITY));
            }
            if let Some(l) = lod.and_then(|l| l.parts.get(k)) {
                let li = b.lib_mesh(l.0, &ps.tints, ps.net.as_ref());
                if !b.scene.lods.iter().any(|x| x.0 == mi) {
                    b.scene.lods.push((mi, li));
                }
            }
            for it in &ps.items {
                let m = Mat4::from_translation(it.pos.as_vec3())
                    * Mat4::from_rotation_z(it.yaw as f32)
                    * Mat4::from_rotation_x(it.tilt as f32)
                    * Mat4::from_scale(it.scale.as_vec3());
                let model = c * m * ci * *local;
                let color = it.color.map(|v| v.as_vec4()).unwrap_or(Vec4::ZERO);
                b.scene.instances[mi].push(Instance { model, color });
                let p = to_gltf(it.pos);
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
    }
    // lights
    for s in &b.st.spots {
        let outer = (s.angle_deg / 2.0).to_radians();
        let ca = outer.cos();
        let inner = (ca + s.blend * (1.0 - ca)).min(1.0).acos();
        b.scene.lights.push(Light {
            position: to_gltf(s.pos),
            direction: to_gltf(s.dir).normalize(),
            color: s.color.as_vec3(),
            candela: (s.power / (4.0 * std::f64::consts::PI) * 683.0) as f32,
            inner: inner as f32,
            outer: outer as f32,
        });
    }
    let p = &b.st.pitch;
    let g = &p.grass_spec;
    let hex = |k: &str, d: &str| glb::srgb_hex(g.get(k).and_then(|v| v.as_str()).unwrap_or(d));
    b.scene.pitch = Some(PitchInfo {
        pattern: p.mow_pattern.clone(),
        width: p.mow_width as f32,
        angle: p.mow_angle as f32,
        strength: (g.get("stripe_strength").and_then(|v| v.as_f64()).unwrap_or(0.2) * (p.height_mm / 25.0).clamp(0.4, 1.6)) as f32,
        natural: p.natural(),
        color: hex("color", "#4a6d27"),
        dry: hex("color_dry", "#8a8247"),
        wear: p.wear as f32,
        wetness: p.wetness as f32,
        height_mm: p.height_mm as f32,
        half: Vec2::new(p.length as f32 / 2.0, p.width as f32 / 2.0),
        procedural: true,
    });
    b.scene.bounds = (lo, hi);
    let w = std::mem::take(&mut b.warnings);
    (b.scene, w)
}
