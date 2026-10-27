//! Pitch parameters, crown, Law 1 markings (port of the pure parts of kit/assets/pitch.py).

use crate::catalog::{merge, Catalog};
use crate::mesh::MeshAcc;
use crate::prng::uniform;
use crate::py::{clip, interp, linspace};
use glam::{DVec2, DVec3};
use serde_json::{json, Value};
use std::f64::consts::PI;

pub const PENALTY_AREA: f64 = 16.5;
pub const GOAL_AREA: f64 = 5.5;
pub const GOAL_HALF: f64 = 7.32 / 2.0;
pub const CIRCLE_R: f64 = 9.15;
pub const PEN_MARK: f64 = 11.0;
pub const SPOT_R: f64 = 0.11;
pub const CORNER_R: f64 = 1.0;

pub fn defaults() -> Value {
    json!({
        "length": 105.0, "width": 68.0, "line_width": 0.12, "runoff_side": 6.0, "runoff_end": 7.5,
        "crown": 0.06, "grass": "perennial_ryegrass", "height_mm": null,
        "mow": {"pattern": "stripes", "width": 5.25, "angle": 0.0, "offset": 0.0},
        "wear": 0.25, "wetness": 0.0, "line_color": "#f5f5f0", "hand_painted": 0.0,
        "optional_marks": true, "markings": "full", "seed": 1
    })
}

/// Resolved pitch parameters.
#[derive(Clone, Debug)]
pub struct Pitch {
    pub length: f64,
    pub width: f64,
    pub line_width: f64,
    pub runoff_side: f64,
    pub runoff_end: f64,
    pub crown: f64,
    pub grass: String,
    pub height_mm: f64,
    pub mow_pattern: String,
    pub mow_width: f64,
    pub mow_angle: f64,
    pub mow_offset: f64,
    pub wear: f64,
    pub wetness: f64,
    pub line_color: String,
    pub hand_painted: f64,
    pub optional_marks: bool,
    /// "full", "outline" (small-sided: touchlines, halfway, centre circle) or "none"
    pub markings: String,
    pub seed: i64,
    /// the grass catalog entry
    pub grass_spec: Value,
    pub params: Value,
}

/// What the renderer's procedural grass needs for one grass surface: the turf (colours, blade
/// height, mowing, wear, wetness) and where the pitch sits (origin and yaw in kit coordinates, so
/// stripes, wear zones and lines are laid out in the pitch's own frame). `wear_zones` is false for
/// plain lawns, which wear evenly instead of in goalmouths.
#[derive(Clone, Debug, PartialEq)]
pub struct GrassSpec {
    pub pattern: String,
    pub stripe_width: f64,
    pub angle_deg: f64,
    pub strength: f64,
    pub natural: bool,
    pub color: String,
    pub dry: String,
    pub wear: f64,
    pub wetness: f64,
    pub height_mm: f64,
    pub half: DVec2,
    pub origin: DVec2,
    pub yaw: f64,
    pub wear_zones: bool,
}

impl Pitch {
    /// The grass surface of this pitch placed at `origin`, rotated by `yaw` (radians).
    pub fn grass_spec(&self, origin: DVec2, yaw: f64) -> GrassSpec {
        let g = &self.grass_spec;
        let s = |k: &str, d: &str| g.get(k).and_then(|v| v.as_str()).unwrap_or(d).to_string();
        GrassSpec {
            pattern: self.mow_pattern.clone(),
            stripe_width: self.mow_width,
            angle_deg: self.mow_angle,
            strength: g.get("stripe_strength").and_then(|v| v.as_f64()).unwrap_or(0.2) * (self.height_mm / 25.0).clamp(0.4, 1.6),
            natural: self.natural(),
            color: s("color", "#4a6d27"),
            dry: s("color_dry", "#8a8247"),
            wear: self.wear,
            wetness: self.wetness,
            height_mm: self.height_mm,
            half: DVec2::new(self.length / 2.0, self.width / 2.0),
            origin,
            yaw,
            wear_zones: true,
        }
    }

    pub fn resolve(params: &Value, cat: Option<&Catalog>) -> Pitch {
        let p = merge(&defaults(), if params.is_object() { params } else { &Value::Null });
        let p = if p.is_object() { p } else { defaults() };
        let n = |k: &str| p[k].as_f64().unwrap_or(0.0);
        let grass = p["grass"].as_str().unwrap_or("perennial_ryegrass").to_string();
        let g = cat.and_then(|c| c.grass.get(&grass).cloned()).unwrap_or_else(|| {
            json!({"natural": true, "height_mm": {"default": 25}, "color": "#4a6d27", "color_dry": "#8a8247",
                   "stripe_strength": 0.2, "blade_width_mm": 3.0})
        });
        let height = p["height_mm"].as_f64().unwrap_or_else(|| g["height_mm"]["default"].as_f64().unwrap_or(25.0));
        Pitch {
            length: n("length"),
            width: n("width"),
            line_width: n("line_width"),
            runoff_side: n("runoff_side"),
            runoff_end: n("runoff_end"),
            crown: n("crown"),
            grass,
            height_mm: height,
            mow_pattern: p["mow"]["pattern"].as_str().unwrap_or("stripes").to_string(),
            mow_width: p["mow"]["width"].as_f64().unwrap_or(5.25),
            mow_angle: p["mow"]["angle"].as_f64().unwrap_or(0.0),
            mow_offset: p["mow"]["offset"].as_f64().unwrap_or(0.0),
            wear: n("wear"),
            wetness: n("wetness"),
            line_color: p["line_color"].as_str().unwrap_or("#f5f5f0").to_string(),
            hand_painted: n("hand_painted"),
            optional_marks: p["optional_marks"].as_bool().unwrap_or(true),
            markings: p["markings"].as_str().unwrap_or("full").to_string(),
            seed: p["seed"].as_i64().unwrap_or(1),
            grass_spec: g,
            params: p,
        }
    }

    pub fn natural(&self) -> bool {
        self.grass_spec["natural"].as_bool().unwrap_or(true)
    }

    pub fn crown_z(&self, _x: f64, y: f64) -> f64 {
        let half = self.width / 2.0 + self.runoff_side;
        self.crown * (1.0 - clip(y.abs() / half, 0.0, 1.0).powi(2))
    }

    /// Size of the playing surface including run-off.
    pub fn surface(&self) -> (f64, f64) {
        (self.length + 2.0 * self.runoff_end, self.width + 2.0 * self.runoff_side)
    }
}

pub fn arc(c: DVec2, r: f64, a0: f64, a1: f64, step: f64) -> Vec<DVec2> {
    let n = 3usize.max(((a1 - a0).abs() * r / step) as i64 as usize + 1);
    linspace(a0, a1, n).iter().map(|a| DVec2::new(c.x + r * a.cos(), c.y + r * a.sin())).collect()
}

/// Resampling step for a marking: `base`, but at least 24 segments on short curves (the 1 m corner
/// arcs would otherwise turn into a three-segment polyline).
pub fn line_step(p: &[DVec2], base: f64) -> f64 {
    let len: f64 = p.windows(2).map(|w| (w[1] - w[0]).length()).sum();
    base.min((len / 24.0).max(0.05))
}

pub fn resample(p: &[DVec2], step: f64) -> Vec<DVec2> {
    let mut s = vec![0.0];
    for w in p.windows(2) {
        s.push(s[s.len() - 1] + (w[1] - w[0]).length());
    }
    let total = s[s.len() - 1];
    let n = 2usize.max((total / step).ceil() as usize + 1);
    let xs: Vec<f64> = p.iter().map(|q| q.x).collect();
    let ys: Vec<f64> = p.iter().map(|q| q.y).collect();
    linspace(0.0, total, n).iter().map(|&t| DVec2::new(interp(t, &s, &xs), interp(t, &s, &ys))).collect()
}

pub fn smooth_noise_1d(t: &[f64], seed: u64, octaves: usize, base_freq: f64) -> Vec<f64> {
    let mut out = vec![0.0; t.len()];
    let (mut amp, mut total) = (1.0, 0.0);
    for o in 0..octaves {
        let f = base_freq * 2.1f64.powi(o as i32) * (0.8 + 0.4 * uniform(seed, 2 * o as u64));
        let ph = 2.0 * PI * uniform(seed, 2 * o as u64 + 1);
        for (v, x) in out.iter_mut().zip(t) {
            *v += amp * (2.0 * PI * f * x + ph).sin();
        }
        total += amp;
        amp *= 0.5;
    }
    out.iter().map(|v| v / total).collect()
}

pub type Line = (Vec<DVec2>, bool);

/// An FA youth format's markings (FA Rules: goalpost and pitch sizes; mini-soccer and 9v9
/// guidance), metres: penalty area (depth, half width), goal area, penalty mark, centre circle;
/// 5v5 has a curved area round the goal instead, 9v9 dashed retreat lines (see kit/assets/pitch.py).
#[derive(Clone, Copy)]
pub struct Youth {
    pub curved_area: Option<f64>,
    pub area: Option<(f64, f64)>,
    pub goal_area: Option<(f64, f64)>,
    pub mark: Option<f64>,
    pub circle: f64,
    pub retreat: bool,
}

pub fn youth(kind: &str) -> Option<Youth> {
    let none = || Youth { curved_area: None, area: None, goal_area: None, mark: None, circle: 0.0, retreat: false };
    match kind {
        "5v5" => Some(Youth { curved_area: Some(5.94), circle: 2.75, ..none() }),
        "7v7" => Some(Youth { area: Some((9.14, 8.23)), mark: Some(7.32), circle: 4.57, ..none() }),
        "9v9" => Some(Youth { area: Some((11.89, 14.63)), goal_area: Some((3.66, 6.10)), mark: Some(8.23), circle: 6.4, retreat: true, ..none() }),
        _ => None,
    }
}

/// Penalty and goal areas, penalty marks, corner arcs and retreat lines of a youth format.
fn youth_marks(y: &Youth, hl: f64, hw: f64, w: f64, lines: &mut Vec<Line>, discs: &mut Vec<(DVec2, f64)>) {
    let v = DVec2::new;
    for s in [1.0f64, -1.0] {
        let gx = s * (hl - w / 2.0);
        if let Some(r) = y.curved_area {
            let a = if s > 0.0 { PI } else { 0.0 };
            lines.push((arc(v(s * hl, 0.0), r - w / 2.0, a - PI / 2.0, a + PI / 2.0, 0.08), false));
        }
        for (depth, half) in [y.area, y.goal_area].into_iter().flatten() {
            let fx = s * (hl - depth + w / 2.0);
            let yy = half - w / 2.0;
            lines.push((vec![v(gx, yy), v(fx, yy), v(fx, -yy), v(gx, -yy)], false));
        }
        if let Some(m) = y.mark {
            discs.push((v(s * (hl - m), 0.0), SPOT_R));
        }
        for t in [1.0f64, -1.0] {
            let a0 = 0.0f64.atan2(-s);
            let a1 = (-t).atan2(0.0);
            let d = (a1 - a0 + PI).rem_euclid(2.0 * PI) - PI;
            lines.push((arc(v(s * hl, t * hw), CORNER_R - w / 2.0, a0, a0 + d, 0.05), false));
        }
        if y.retreat {
            // dashed: a metre of paint, a metre gap, across the pitch
            let x = s * (hl - y.area.map_or(0.0, |a| a.0)) / 2.0;
            let mut yy = -hw + w;
            while yy + 1.0 <= hw - w + 1e-9 {
                lines.push((vec![v(x, yy), v(x, yy + 1.0)], false));
                yy += 2.0;
            }
        }
    }
}

/// Centre-lines of every marking and the spots.
pub fn marking_paths(p: &Pitch) -> (Vec<Line>, Vec<(DVec2, f64)>) {
    let (l, w_, w) = (p.length, p.width, p.line_width);
    let (hl, hw) = (l / 2.0, w_ / 2.0);
    let v = DVec2::new;
    let mut lines: Vec<Line> = Vec::new();
    let mut discs = Vec::new();
    if p.markings == "none" {
        return (lines, discs);
    }
    lines.push((vec![v(hl - w / 2.0, hw - w / 2.0), v(-hl + w / 2.0, hw - w / 2.0), v(-hl + w / 2.0, -hw + w / 2.0), v(hl - w / 2.0, -hw + w / 2.0)], true));
    lines.push((vec![v(0.0, -hw + w / 2.0), v(0.0, hw - w / 2.0)], false));
    let y = youth(&p.markings);
    let r_c = match y {
        Some(y) => y.circle,
        None if p.markings == "full" => CIRCLE_R,
        None => CIRCLE_R.min(w_ / 6.0),
    };
    let mut circle = arc(DVec2::ZERO, r_c - w / 2.0, 0.0, 2.0 * PI, 0.08);
    circle.pop();
    lines.push((circle, true));
    discs.push((DVec2::ZERO, SPOT_R));
    if let Some(y) = y {
        youth_marks(&y, hl, hw, w, &mut lines, &mut discs);
        return (lines, discs);
    }
    if p.markings != "full" {
        return (lines, discs);
    }
    for s in [1.0, -1.0] {
        let gx = s * (hl - w / 2.0);
        for (depth, half) in [(PENALTY_AREA, GOAL_HALF + PENALTY_AREA), (GOAL_AREA, GOAL_HALF + GOAL_AREA)] {
            let fx = s * (hl - depth + w / 2.0);
            let yy = half - w / 2.0;
            lines.push((vec![v(gx, yy), v(fx, yy), v(fx, -yy), v(gx, -yy)], false));
        }
        let mx = s * (hl - PEN_MARK);
        discs.push((v(mx, 0.0), SPOT_R));
        let r = CIRCLE_R - w / 2.0;
        let th = (1.0f64.min((PENALTY_AREA - PEN_MARK) / r)).acos();
        let phi0 = if s > 0.0 { PI } else { 0.0 };
        lines.push((arc(v(mx, 0.0), r, phi0 - th, phi0 + th, 0.08), false));
        for t in [1.0, -1.0] {
            let a0 = 0.0f64.atan2(-s);
            let a1 = (-t as f64).atan2(0.0);
            let d = (a1 - a0 + PI).rem_euclid(2.0 * PI) - PI;
            lines.push((arc(v(s * hl, t * hw), CORNER_R - w / 2.0, a0, a0 + d, 0.05), false));
            if p.optional_marks {
                let y = t * (hw - CORNER_R - CIRCLE_R);
                lines.push((vec![v(s * (hl + 0.05), y), v(s * (hl + 0.55), y)], false));
                let x = s * (hl - CORNER_R - CIRCLE_R);
                lines.push((vec![v(x, t * (hw + 0.05)), v(x, t * (hw + 0.55))], false));
            }
        }
    }
    (lines, discs)
}

/// Hand-painted wobble (sideways smooth noise).
pub fn wobble(pts: &[DVec2], closed: bool, amp: f64, seed: u64) -> Vec<DVec2> {
    if amp <= 0.0 {
        return pts.to_vec();
    }
    let mut q: Vec<DVec2> = pts.to_vec();
    if closed {
        q.push(pts[0]);
    }
    let q = resample(&q, line_step(&q, 0.4));
    let n = q.len();
    let d: Vec<DVec2> = (0..n)
        .map(|i| {
            let g = if i == 0 { q[1] - q[0] } else if i == n - 1 { q[n - 1] - q[n - 2] } else { (q[i + 1] - q[i - 1]) / 2.0 };
            g / g.length()
        })
        .collect();
    let mut s = vec![0.0];
    for w in q.windows(2) {
        s.push(s[s.len() - 1] + (w[1] - w[0]).length());
    }
    let off = smooth_noise_1d(&s, seed, 3, 0.12);
    let mut out: Vec<DVec2> = (0..n).map(|i| q[i] + DVec2::new(-d[i].y, d[i].x) * (amp * off[i])).collect();
    if closed {
        out.pop();
    }
    out
}

/// Flat ribbon along a polyline (mitred joins): verts, quads, per-loop uvs.
pub fn strip(points: &[DVec2], width: f64, closed: bool) -> (Vec<DVec2>, Vec<[u32; 4]>, Vec<DVec2>) {
    let mut p = points.to_vec();
    if closed {
        p.push(points[0]);
    }
    let n = p.len();
    let ns: Vec<DVec2> = p.windows(2).map(|w| {
        let d = (w[1] - w[0]).normalize();
        DVec2::new(-d.y, d.x)
    }).collect();
    let mut nrm = vec![DVec2::ZERO; n];
    nrm[0] = ns[0];
    nrm[n - 1] = ns[ns.len() - 1];
    for i in 1..n - 1 {
        let m = (ns[i - 1] + ns[i]).normalize();
        nrm[i] = m / m.dot(ns[i]).max(0.3);
    }
    if closed {
        let m = (ns[ns.len() - 1] + ns[0]).normalize();
        let m = m / m.dot(ns[0]).max(0.3);
        nrm[0] = m;
        nrm[n - 1] = m;
    }
    let mut v = Vec::with_capacity(2 * n);
    for i in 0..n {
        v.push(p[i] - nrm[i] * width / 2.0);
        v.push(p[i] + nrm[i] * width / 2.0);
    }
    let mut s = vec![0.0];
    for w in p.windows(2) {
        s.push(s[s.len() - 1] + (w[1] - w[0]).length());
    }
    let mut q = Vec::new();
    let mut uv = Vec::new();
    for i in 0..n as u32 - 1 {
        q.push([2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1]);
        let (a, b) = (s[i as usize], s[i as usize + 1]);
        uv.extend([DVec2::new(a, 0.0), DVec2::new(b, 0.0), DVec2::new(b, 1.0), DVec2::new(a, 1.0)]);
    }
    (v, q, uv)
}

/// Line paint geometry following the crown (one MeshAcc, tag 0).
pub fn markings_mesh(p: &Pitch) -> MeshAcc {
    let (lines, discs) = marking_paths(p);
    let amp = 0.035 * p.hand_painted;
    let mut acc = MeshAcc::new();
    let width = p.line_width * (1.0 + 0.15 * p.hand_painted);
    for (i, (pts, closed)) in lines.iter().enumerate() {
        let pts = wobble(pts, *closed, amp, (p.seed * 100 + i as i64) as u64);
        let pts = if *closed {
            let mut c = pts.clone();
            c.push(pts[0]);
            let mut r = resample(&c, line_step(&c, 0.5));
            r.pop();
            r
        } else {
            resample(&pts, line_step(&pts, 0.5))
        };
        let (v, q, uv) = strip(&pts, width, *closed);
        let base = acc.verts.len() as u32;
        acc.verts.extend(v.iter().map(|a| DVec3::new(a.x, a.y, p.crown_z(a.x, a.y) + 0.004)));
        for (k, f) in q.iter().enumerate() {
            acc.add_face(f.iter().map(|i| i + base).collect(), 0, Some(uv[4 * k..4 * k + 4].to_vec()));
        }
    }
    for (c, r) in discs {
        let base = acc.verts.len() as u32;
        acc.verts.push(DVec3::new(c.x, c.y, p.crown_z(c.x, c.y) + 0.004));
        let k = 32u32;
        for j in 0..k {
            let a = 2.0 * PI * j as f64 / k as f64;
            let q = c + DVec2::new(a.cos(), a.sin()) * r;
            acc.verts.push(DVec3::new(q.x, q.y, p.crown_z(q.x, q.y) + 0.004));
        }
        for j in (0..k).step_by(2) {
            acc.add_face(vec![base, base + 1 + j, base + 1 + (j + 1) % k, base + 1 + (j + 2) % k], 0, Some(vec![DVec2::splat(0.5); 4]));
        }
    }
    acc
}

/// The crowned playing surface (1 m grid), uv = metres from the surface centre.
pub fn surface_mesh(p: &Pitch) -> MeshAcc {
    let (sx, sy) = p.surface();
    let (nx, ny) = (sx.round() as usize, sy.round() as usize);
    let xs = linspace(-sx / 2.0, sx / 2.0, nx + 1);
    let ys = linspace(-sy / 2.0, sy / 2.0, ny + 1);
    let mut acc = MeshAcc::new();
    for &x in &xs {
        for &y in &ys {
            acc.verts.push(DVec3::new(x, y, p.crown_z(x, y)));
        }
    }
    for i in 0..nx as u32 {
        for j in 0..ny as u32 {
            let a = i * (ny as u32 + 1) + j;
            let f = vec![a, a + ny as u32 + 1, a + ny as u32 + 2, a + 1];
            let uv = f.iter().map(|&k| acc.verts[k as usize].truncate()).collect();
            acc.add_face(f, 0, Some(uv));
        }
    }
    acc
}

/// Goal and corner-flag sockets: (kind, id, position, yaw degrees).
pub fn sockets(p: &Pitch) -> Vec<(&'static str, &'static str, DVec3, f64)> {
    let (l, w_, w) = (p.length, p.width, p.line_width);
    let mut out = Vec::new();
    for (s, id) in [(1.0, "east"), (-1.0, "west")] {
        let x = s * (l / 2.0 - w / 2.0);
        out.push(("goal", id, DVec3::new(x, 0.0, p.crown_z(x, 0.0)), 90.0 * s));
    }
    for (s, t, id) in [(1.0, 1.0, "NE"), (1.0, -1.0, "SE"), (-1.0, 1.0, "NW"), (-1.0, -1.0, "SW")] {
        let (x, y) = (s * (l / 2.0 - w / 2.0), t * (w_ / 2.0 - w / 2.0));
        out.push(("corner_flag", id, DVec3::new(x, y, p.crown_z(x, y)), x.atan2(-y).to_degrees()));
    }
    out
}
