//! Goal net card textures, generated at runtime (port of goal._pattern_image with colours):
//! the cord pattern (hex or square cells, cord thickness) as an alpha mask, coloured solid, in
//! bands, or as a checker. The library's card has UVs in metres divided by one repeat of the
//! default net (hex, 120 mm); `uv_scale` maps those UVs onto this texture's tile.

use crate::glb::Image;
use stadium_gen::stadium::NetSpec;
use std::sync::{Arc, Mutex};

/// The library card's pattern repeat (goal.py defaults: hex mesh, 0.12 m cells).
const LIB_W: f64 = 0.12;
const LIB_H: f64 = 0.12 * 1.732_050_807_568_877_2;
const PX_PER_REPEAT: f64 = 96.0;

type Seg = ([f64; 2], [f64; 2]);

fn hex_segments(x0: f64, x1: f64, y0: f64, y1: f64, size: f64) -> Vec<Seg> {
    let r = size / 3f64.sqrt();
    let (dx, dy) = (size, 1.5 * r);
    let mut out: Vec<Seg> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for row in 0..((y1 - y0) / dy) as i64 + 3 {
        for c in 0..((x1 - x0) / dx) as i64 + 3 {
            let cx = x0 - dx + c as f64 * dx + if row % 2 == 1 { dx / 2.0 } else { 0.0 };
            let cy = y0 - dy + row as f64 * dy;
            let corners: Vec<[f64; 2]> = (0..6).map(|k| {
                let a = (30.0 + 60.0 * k as f64).to_radians();
                [cx + r * a.cos(), cy + r * a.sin()]
            }).collect();
            for k in 0..6 {
                let (a, b) = (corners[k], corners[(k + 1) % 6]);
                let key = (((a[0] + b[0]) * 500.0).round() as i64, ((a[1] + b[1]) * 500.0).round() as i64);
                if seen.insert(key) {
                    out.push((a, b));
                }
            }
        }
    }
    out
}

fn square_segments(x0: f64, x1: f64, y0: f64, y1: f64, size: f64) -> Vec<Seg> {
    let mut out = Vec::new();
    let mut x = x0;
    while x <= x1 + 1e-6 {
        out.push(([x, y0], [x, y1]));
        x += size;
    }
    let mut y = y0;
    while y <= y1 + 1e-6 {
        out.push(([x0, y], [x1, y]));
        y += size;
    }
    out
}

fn seg_dist(p: [f64; 2], s: &Seg) -> f64 {
    let (a, b) = s;
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let t = ((ap[0] * ab[0] + ap[1] * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-12)).clamp(0.0, 1.0);
    let d = [ap[0] - ab[0] * t, ap[1] - ab[1] * t];
    (d[0] * d[0] + d[1] * d[1]).sqrt()
}

/// (texture, uv scale for the library card's UVs)
pub fn texture(spec: &NetSpec) -> (Arc<Image>, [f32; 2]) {
    static CACHE: Mutex<Vec<(NetSpec, Arc<Image>, [f32; 2])>> = Mutex::new(Vec::new());
    if let Some(c) = CACHE.lock().unwrap().iter().find(|c| c.0 == *spec) {
        return (c.1.clone(), c.2);
    }
    let size = (spec.mesh_size_mm.max(20) as f64) / 1000.0;
    let hex = spec.mesh != "square";
    let (w_rep, h_rep) = if hex { (size, 3f64.sqrt() * size) } else { (size, size) };
    let bw = spec.band_cells.max(1) as f64 * size;
    // a tile spans whole pattern repeats and one period of the colour pattern (two bands)
    let reps = |rep: f64, banded: bool| if banded { ((2.0 * bw / rep).round() as usize).max(2) & !1 } else { 1 };
    let bands = spec.pattern == "bands" || spec.pattern == "checker";
    let (nu, nv) = (reps(w_rep, spec.pattern == "checker"), reps(h_rep, bands));
    let (tw, th) = (nu as f64 * w_rep, nv as f64 * h_rep);
    let (w, h) = ((nu as f64 * PX_PER_REPEAT) as u32, (nv as f64 * PX_PER_REPEAT * h_rep / w_rep).round() as u32);
    let segs = if hex { hex_segments(-w_rep, tw + w_rep, -h_rep, th + h_rep, size) } else { square_segments(-w_rep, tw + w_rep, -h_rep, th + h_rep, size) };
    let r = spec.cord_tenths_mm as f64 / 20000.0 * 1.6; // radius, a touch thicker so it holds up at distance
    let px_m = tw / w as f64;
    let colors: Vec<[u8; 3]> = if spec.colors.is_empty() { vec![[247, 247, 245]] } else { spec.colors.clone() };
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for row in 0..h {
        // image row 0 is the top: v (glTF) runs down the image
        let y = th - (row as f64 + 0.5) / h as f64 * th;
        for col in 0..w {
            let x = (col as f64 + 0.5) / w as f64 * tw;
            let d = segs.iter().map(|s| seg_dist([x, y], s)).fold(f64::MAX, f64::min);
            let alpha = ((r - d) / (0.5 * px_m) + 0.5).clamp(0.0, 1.0);
            let band_v = ((y / (th / 2.0)).floor() as i64).rem_euclid(2) as usize;
            let band_u = ((x / (tw / 2.0)).floor() as i64).rem_euclid(2) as usize;
            let idx = match spec.pattern.as_str() {
                "bands" => band_v,
                "checker" => band_v ^ band_u,
                _ => 0,
            };
            let c = colors[idx % colors.len()];
            let o = ((row * w + col) * 4) as usize;
            rgba[o..o + 4].copy_from_slice(&[c[0], c[1], c[2], (alpha * 255.0).round() as u8]);
        }
    }
    let img = Arc::new(Image::new(w, h, rgba));
    let scale = [(LIB_W / tw) as f32, (LIB_H / th) as f32];
    CACHE.lock().unwrap().push((spec.clone(), img.clone(), scale));
    (img, scale)
}
