//! Ad and furniture placement (port of kit/ads/placement.py): perimeter board runs, anamorphic 3D
//! carpets, TV weights, the main camera.

use crate::pitch::Pitch;
use crate::py::round_to;
use crate::runs::Run;
use glam::{DVec2, DVec3};

pub const BOARD_SIDE_OFFSET: f64 = 4.0;
pub const BOARD_END_OFFSET: f64 = 5.5;
pub const BOARD_HEIGHT: f64 = 0.9;

/// (camera position, look-at target)
pub fn main_camera(p: &Pitch, mount: Option<DVec3>) -> (DVec3, DVec3) {
    match mount {
        Some(m) => (m, DVec3::new(0.0, 6.0, 0.0)),
        None => (DVec3::new(0.0, -(p.width / 2.0 + 32.0), 24.0), DVec3::new(0.0, 4.0, 0.0)),
    }
}

pub fn board_runs(p: &Pitch, gap: f64, side_offset: f64, end_offset: f64) -> Vec<Run> {
    let (l, w) = (p.length, p.width);
    let (ys, xe) = (w / 2.0 + side_offset, l / 2.0 + end_offset);
    let v = DVec2::new;
    let mut runs = Vec::new();
    if gap > 0.0 {
        runs.push(Run::straight("south_a", v(-l / 2.0, -ys), v(1.0, 0.0), l / 2.0 - gap / 2.0));
        runs.push(Run::straight("south_b", v(gap / 2.0, -ys), v(1.0, 0.0), l / 2.0 - gap / 2.0));
    } else {
        runs.push(Run::straight("south", v(-l / 2.0, -ys), v(1.0, 0.0), l));
    }
    runs.push(Run::straight("east", v(xe, -w / 2.0 + 2.0), v(0.0, 1.0), w - 4.0));
    runs.push(Run::straight("north", v(l / 2.0, ys), v(-1.0, 0.0), l));
    runs.push(Run::straight("west", v(-xe, w / 2.0 - 2.0), v(0.0, -1.0), w - 4.0));
    runs
}

pub fn tv_weight(centre: DVec3, facing: DVec3, cam: DVec3) -> f64 {
    let d = cam - centre;
    let dist = d.length();
    let f = facing / facing.length();
    let face = f.dot(d / dist).max(0.0);
    round_to(face * 1.0 / (1.0 + (dist / 120.0).powi(2)), 3)
}

/// Anamorphic carpet: verts, quads, per-vertex uvs.
pub fn carpet(cam: DVec3, base: DVec3, width: f64, height: f64, nu: usize, nv: usize, ground: &dyn Fn(f64, f64) -> f64) -> (Vec<DVec3>, Vec<[u32; 4]>, Vec<DVec2>) {
    let to_cam = (cam.truncate() - base.truncate()).normalize();
    let across = DVec2::new(-to_cam.y, to_cam.x);
    let mut v = Vec::new();
    let mut uv = Vec::new();
    for j in 0..=nv {
        for i in 0..=nu {
            let (u, vv) = (i as f64 / nu as f64, j as f64 / nv as f64);
            let xy = base.truncate() + across * (u - 0.5) * width;
            let p = DVec3::new(xy.x, xy.y, vv * height);
            let t = cam.z / (cam.z - p.z);
            let mut g = cam + (p - cam) * t;
            g.z = ground(g.x, g.y) + 0.008;
            v.push(g);
            uv.push(DVec2::new(u, vv));
        }
    }
    let mut f = Vec::new();
    let w = nu as u32 + 1;
    for j in 0..nv as u32 {
        for i in 0..nu as u32 {
            f.push([j * w + i, j * w + i + 1, (j + 1) * w + i + 1, (j + 1) * w + i]);
        }
    }
    (v, f, uv)
}

/// Base centres for carpets: (end sign, side sign, base).
pub fn carpet_spots(p: &Pitch, width: f64, behind: f64) -> Vec<(f64, f64, DVec3)> {
    let mut out = Vec::new();
    for s in [1.0, -1.0] {
        for side in [1.0, -1.0] {
            out.push((s, side, DVec3::new(s * (p.length / 2.0 + behind), side * (3.66 + 1.6 + width / 2.0), 0.0)));
        }
    }
    out
}

/// Keep a carpet off the field of play (see placement.place_carpet): push it back until its whole
/// footprint lies behind the goal line and inside the touchline extended, then fit its height in
/// front of the end boards. Returns (base, height).
pub fn place_carpet(cam: DVec3, base: DVec3, width: f64, height: f64, p: &Pitch, limit_x: f64) -> (DVec3, f64) {
    let margin = 0.3;
    let s = if base.x > 0.0 { 1.0 } else { -1.0 };
    let mut base = base;
    let mut h = height;
    for _ in 0..8 {
        h = fit_carpet_height(cam, base, width, height, limit_x);
        let (v, _, _) = carpet(cam, base, width, h, 4, 2, &|_, _| 0.0);
        let inner = v.iter().map(|q| s * q.x).fold(f64::MAX, f64::min) - (p.length / 2.0 + margin);
        let side = v.iter().map(|q| q.y.abs()).fold(f64::MIN, f64::max) - (p.width / 2.0 - 0.5);
        if inner >= -1e-6 && side <= 1e-6 {
            break;
        }
        if inner < 0.0 {
            base.x += s * -inner;
        }
        if side > 0.0 {
            base.y -= base.y.signum() * side;
        }
    }
    (base, h)
}

pub fn fit_carpet_height(cam: DVec3, base: DVec3, width: f64, height: f64, limit_x: f64) -> f64 {
    let mut h = height;
    for _ in 0..30 {
        let (v, _, _) = carpet(cam, base, width, h, 4, 2, &|_, _| 0.0);
        if v.iter().map(|p| p.x.abs()).fold(0.0, f64::max) <= limit_x {
            return h;
        }
        h *= 0.92;
    }
    h
}
