//! The editor's runtime asset library: out/library/library.game.glb (materials as MAT:<name>
//! swatches, props as PROP:<key> roots; see tools/export_library.py) plus the ad creatives and the
//! scoreboard rendered by kit/ads/studio.py.

use crate::glb::{self, Image, Material, Primitive};
use glam::Mat4;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One prop: its meshes (as primitive lists) with transforms relative to the prop root (glTF space).
pub struct Prop {
    pub parts: Vec<(usize, Mat4)>,
}

pub struct Library {
    pub root: PathBuf,
    pub scene: glb::Scene,
    pub materials: HashMap<String, usize>,
    pub props: HashMap<String, Prop>,
    boards: std::sync::Mutex<HashMap<String, Arc<Image>>>,
    atlases: std::sync::Mutex<HashMap<Vec<String>, Arc<Image>>>,
}

fn load_png(path: &Path) -> Result<Image, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgba8();
    Ok(Image::new(img.width(), img.height(), img.into_raw()))
}

impl Library {
    /// `root`: the repository (holds out/library, out/ads and catalog/).
    pub fn load(root: &Path) -> Result<Library, String> {
        let path = root.join("out/library/library.game.glb");
        let scene = glb::load(&path).map_err(|e| format!("{e} (run: blender -b --factory-startup -P tools/export_library.py)"))?;
        let materials = scene.materials.iter().enumerate().map(|(i, m)| (m.name.clone(), i)).collect();
        let mut props: HashMap<String, Prop> = HashMap::new();
        for (name, top, mesh, world) in &scene.nodes {
            let key = if let Some(k) = top.strip_prefix("PROP:") {
                k.to_string()
            } else if let Some(k) = name.strip_prefix("PROP:") {
                k.to_string()
            } else {
                continue;
            };
            props.entry(key).or_insert_with(|| Prop { parts: Vec::new() }).parts.push((*mesh, *world));
        }
        Ok(Library {
            root: root.to_path_buf(),
            scene,
            materials,
            props,
            boards: Default::default(),
            atlases: Default::default(),
        })
    }

    pub fn material(&self, name: &str) -> Option<&Material> {
        self.materials.get(name).map(|&i| &self.scene.materials[i])
    }

    pub fn prims(&self, mesh: usize) -> &[Primitive] {
        &self.scene.meshes[mesh]
    }

    fn creative(&self, kind: &str, key: &str) -> Result<Arc<Image>, String> {
        let id = format!("{kind}_{key}");
        if let Some(i) = self.boards.lock().unwrap().get(&id) {
            return Ok(i.clone());
        }
        let img = Arc::new(load_png(&self.root.join("out/ads").join(format!("{id}.png")))?);
        self.boards.lock().unwrap().insert(id, img.clone());
        Ok(img)
    }

    /// Horizontal atlas of board creatives in this order (cached by the key list).
    pub fn board_atlas(&self, keys: &[String]) -> Result<Arc<Image>, String> {
        if let Some(a) = self.atlases.lock().unwrap().get(keys) {
            return Ok(a.clone());
        }
        let imgs: Vec<Arc<Image>> = keys.iter().map(|k| self.creative("board", k)).collect::<Result<_, _>>()?;
        let (w, h) = (imgs[0].width, imgs[0].height);
        let mut rgba = vec![0u8; (w as usize * imgs.len()) * h as usize * 4];
        let row = w as usize * imgs.len() * 4;
        for (k, im) in imgs.iter().enumerate() {
            for y in 0..h as usize {
                let src = &im.rgba[y * w as usize * 4..(y + 1) * w as usize * 4];
                rgba[y * row + k * w as usize * 4..y * row + (k + 1) * w as usize * 4].copy_from_slice(src);
            }
        }
        let atlas = Arc::new(Image::new(w * imgs.len() as u32, h, rgba));
        self.atlases.lock().unwrap().insert(keys.to_vec(), atlas.clone());
        Ok(atlas)
    }

    pub fn carpet(&self, key: &str) -> Result<Arc<Image>, String> {
        self.creative("carpet", key)
    }

    pub fn scoreboard(&self) -> Result<Arc<Image>, String> {
        self.creative("screen", "scoreboard")
    }
}
