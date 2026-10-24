//! Floodlight layouts and design (port of kit/stands/floodlights.py and the design loop in
//! kit/assets/floodlight.py): corner pylons, roof ribbons, grassroots poles; greedy aim
//! optimisation for uniformity; lamp power from the target lux.

use crate::py::{clip, interp, linspace};
use crate::stands::Polyline;
use glam::{DVec2, DVec3};
use serde_json::Value;

pub const LUX_PER_W: f64 = 683.0;
pub const MIN_ELEVATION: f64 = 25.0;

#[derive(Clone, Copy, Debug)]
pub struct Mast {
    pub base: DVec2,
    pub height: f64,
}

/// Four masts on the diagonals, outside the stands. reach: how far out the structures go;
/// corner_r: the corner radius of a curved bowl (the mast then stands behind the corner stand).
pub fn pylons(_l: f64, _w: f64, a: f64, b: f64, reach: f64, roof_top: f64, corner_r: Option<f64>) -> Vec<Mast> {
    let mut out = Vec::new();
    for (sx, sy) in [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)] {
        // open corners: in the gap between the stands (which never reach past a, b)
        let gap = 0.5 * reach.min(30.0);
        let mut p = DVec2::new(sx * (a + gap), sy * (b + gap));
        if let Some(r) = corner_r {
            // a bowl: behind the corner stand, clear of its back and the mast head's width
            let c = DVec2::new(sx * (a - r), sy * (b - r));
            let diag = DVec2::new(sx, sy) / std::f64::consts::SQRT_2;
            let q = c + diag * (r + reach + 8.0);
            if q.length() > p.length() {
                p = q;
            }
        }
        let dist = p.length();
        let h = (dist * (MIN_ELEVATION + 3.0).to_radians().tan()).max(roof_top + 8.0).max(30.0);
        out.push(Mast { base: p, height: h });
    }
    out
}

/// Fixture positions on a mast head facing the centre spot; (points, yaw, tilt).
pub fn head_layout(m: &Mast, rows: usize, cols: usize) -> (Vec<DVec3>, f64, f64) {
    let (bx, by) = (m.base.x, m.base.y);
    let h = m.height;
    let yaw = (-by).atan2(-bx);
    let tilt = h.atan2(bx.hypot(by));
    let fwd = DVec3::new(yaw.cos(), yaw.sin(), 0.0);
    let side = DVec3::new(-yaw.sin(), yaw.cos(), 0.0);
    let up = DVec3::Z * tilt.cos() + fwd * tilt.sin();
    let centre = DVec3::new(bx, by, h) + fwd * 1.0;
    let mut pts = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            pts.push(centre + side * (c as f64 - (cols as f64 - 1.0) / 2.0) * 0.8 + up * (r as f64 - (rows as f64 - 1.0) / 2.0) * 0.75);
        }
    }
    (pts, yaw, tilt)
}

/// Points evenly spaced along polylines (cumulative length, interpolated).
pub fn resample(line: &[DVec3], spacing: f64, min_n: usize) -> Vec<DVec3> {
    let mut s = vec![0.0];
    for w in line.windows(2) {
        s.push(s[s.len() - 1] + (w[1] - w[0]).length());
    }
    let total = s[s.len() - 1];
    let n = min_n.max((total / spacing) as i64 as usize);
    let xs: Vec<f64> = line.iter().map(|p| p.x).collect();
    let ys: Vec<f64> = line.iter().map(|p| p.y).collect();
    let zs: Vec<f64> = line.iter().map(|p| p.z).collect();
    (0..n)
        .map(|k| {
            let t = (k as f64 + 0.5) * total / n as f64;
            DVec3::new(interp(t, &s, &xs), interp(t, &s, &ys), interp(t, &s, &zs))
        })
        .collect()
}

/// Horizontal unit normal of a roof edge polyline at its point nearest `p`, pointing away from the
/// pitch centre.
pub fn edge_normal(line: &[DVec3], p: DVec3) -> DVec2 {
    let mut best = (f64::MAX, DVec2::X);
    for w in line.windows(2) {
        let (a, b) = (w[0].truncate(), w[1].truncate());
        let ab = b - a;
        if ab.length_squared() < 1e-12 {
            continue;
        }
        let t = ((p.truncate() - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
        let dist = (a + ab * t - p.truncate()).length();
        if dist < best.0 {
            best = (dist, ab.normalize());
        }
    }
    let n = DVec2::new(best.1.y, -best.1.x);
    if n.dot(p.truncate()) < 0.0 { -n } else { n }
}

pub fn roof_ribbon(edge_lines: &[Vec<DVec3>], spacing: f64) -> Vec<DVec3> {
    edge_lines.iter().flat_map(|l| resample(l, spacing, 1)).collect()
}

pub fn poles(l: f64, w: f64, per_side: usize, height: f64, offset: f64) -> Vec<Mast> {
    let xs = if per_side > 1 {
        let v = linspace(-l / 2.0, l / 2.0, per_side + 2);
        v[1..v.len() - 1].to_vec()
    } else {
        vec![0.0]
    };
    let mut out = Vec::new();
    for sy in [1.0, -1.0] {
        for &x in &xs {
            // the main-stand side keeps halfway clear for the dugouts, tunnel and TV camera
            let x = if sy < 0.0 && x.abs() < POLE_CLEAR_X { if x < 0.0 { -POLE_CLEAR_X } else { POLE_CLEAR_X } } else { x };
            out.push(Mast { base: DVec2::new(x, sy * (w / 2.0 + offset)), height });
        }
    }
    out
}

/// Poles on the main-stand side stay this far from halfway.
pub const POLE_CLEAR_X: f64 = 15.0;

pub fn lamp_power(n_fixtures: usize, spot_deg: f64, lux: f64, area: f64) -> f64 {
    let efficiency = 0.65;
    let a = (spot_deg / 2.0).to_radians();
    let omega = 2.0 * std::f64::consts::PI * (1.0 - a.cos());
    let per = (lux / LUX_PER_W) * area / (n_fixtures as f64 * efficiency);
    per * 4.0 * std::f64::consts::PI / omega
}

pub fn spot_angle(n_fixtures: usize, area: f64, throw: f64) -> f64 {
    let overlap = 25.0;
    let r = (overlap * area / (n_fixtures as f64 * std::f64::consts::PI)).sqrt();
    clip(2.0 * (r / throw.max(1.0)).atan().to_degrees(), 14.0, 70.0)
}

/// (U1 = Emin/Emax, U2 = Emin/Eavg).
pub fn uniformity(e: &[f64]) -> (f64, f64) {
    let mn = e.iter().cloned().fold(f64::MAX, f64::min);
    let mx = e.iter().cloned().fold(f64::MIN, f64::max);
    let mean = e.iter().sum::<f64>() / e.len() as f64;
    (mn / mx, mn / mean)
}

fn illum_one(p: DVec3, t: DVec3, power: f64, spot_deg: f64, blend: f64, points: &[DVec3], out: &mut [f64]) {
    let a = (spot_deg / 2.0).to_radians();
    let i = power / (4.0 * std::f64::consts::PI);
    let axis = (t - p) / (t - p).length();
    let ca = a.cos();
    let band = (blend * (1.0 - ca)).max(1e-6);
    for (k, q) in points.iter().enumerate() {
        let d = *q - p;
        let r = d.length();
        let u = d / r;
        let cos_t = u.x * axis.x + u.y * axis.y + u.z * axis.z;
        let x = clip((cos_t - ca) / band, 0.0, 1.0);
        let w = x * x * (3.0 - 2.0 * x);
        let cos_inc = clip(-u.z, 0.0, 1.0);
        out[k] += i * w * cos_inc / (r * r);
    }
}

/// Horizontal illuminance (lux) at points from fixtures aimed at targets.
pub fn illuminance(fixtures: &[DVec3], targets: &[DVec3], power: f64, spot_deg: f64, blend: f64, points: &[DVec3]) -> Vec<f64> {
    let mut e = vec![0.0; points.len()];
    for (p, t) in fixtures.iter().zip(targets) {
        illum_one(*p, *t, power, spot_deg, blend, points, &mut e);
    }
    e.iter().map(|v| v * LUX_PER_W).collect()
}

fn argmin(v: &[f64]) -> usize {
    let mut b = 0;
    for i in 1..v.len() {
        if v[i] < v[b] {
            b = i;
        }
    }
    b
}

fn argmax(v: &[f64]) -> usize {
    let mut b = 0;
    for i in 1..v.len() {
        if v[i] > v[b] {
            b = i;
        }
    }
    b
}

/// Greedy re-aiming for uniformity. Returns (targets, lux grid at 1 W per fixture).
pub fn optimise_aims(fixtures: &[DVec3], candidates: &[DVec3], points: &[DVec3], spot_deg: f64) -> (Vec<DVec3>, Vec<f64>) {
    let (blend, iters, seed) = (0.9, 600, 3usize);
    let f_n = fixtures.len();
    let a_n = candidates.len();
    let g_n = points.len();
    // C[f][a][g]
    let mut c = vec![0.0f64; f_n * a_n * g_n];
    for (fi, f) in fixtures.iter().enumerate() {
        for (ai, a) in candidates.iter().enumerate() {
            let off = (fi * a_n + ai) * g_n;
            let row = &mut c[off..off + g_n];
            illum_one(*f, *a, 1.0, spot_deg, blend, points, row);
            for v in row.iter_mut() {
                *v *= LUX_PER_W;
            }
        }
    }
    let cr = |f: usize, a: usize| &c[(f * a_n + a) * g_n..(f * a_n + a + 1) * g_n];
    // a fixture never aims back over its own shoulder: its aim point lies in front of it, seen
    // from the pitch's spine (the long axis between the penalty spots' latitudes); the start deals
    // candidates round the fixtures (spread light), skipping those behind each fixture
    let hx = candidates.iter().map(|c| c.x).fold(f64::MIN, f64::max);
    let hy = candidates.iter().map(|c| c.y).fold(f64::MIN, f64::max);
    let spine = (hx - hy).max(0.0);
    let allowed: Vec<bool> = fixtures
        .iter()
        .flat_map(|f| {
            let (ix, iy) = (f.x.clamp(-spine, spine) - f.x, -f.y);
            candidates.iter().map(move |c| {
                let (dx, dy) = (c.x - f.x, c.y - f.y);
                ix.hypot(iy) < 1.0 || dx * ix + dy * iy >= 0.0
            })
        })
        .collect();
    let ok = |f: usize, a: usize| allowed[f * a_n + a];
    let mut aim: Vec<usize> = (0..f_n)
        .map(|f| {
            let a0 = (f * 37 + seed) % a_n;
            (0..a_n).map(|j| (a0 + j * 7) % a_n).find(|&a| ok(f, a)).unwrap_or(a0)
        })
        .collect();
    let mut e = vec![0.0; g_n];
    for f in 0..f_n {
        for (k, v) in cr(f, aim[f]).iter().enumerate() {
            e[k] += v;
        }
    }
    let nearest = |g: usize| {
        let d: Vec<f64> = candidates
            .iter()
            .map(|c| {
                let dx = c.x - points[g].x;
                let dy = c.y - points[g].y;
                (dx * dx + dy * dy).sqrt()
            })
            .collect();
        argmin(&d)
    };
    let scores = |e: &[f64], aim: &[usize], star: usize| -> Vec<f64> {
        (0..f_n)
            .map(|f| {
                if !ok(f, star) {
                    return f64::MIN;
                }
                let (old, new) = (cr(f, aim[f]), cr(f, star));
                let (mut mn, mut mx) = (f64::MAX, f64::MIN);
                for k in 0..g_n {
                    let v = e[k] - old[k] + new[k];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                mn / mx
            })
            .collect()
    };
    for _ in 0..iters {
        let emin = e.iter().cloned().fold(f64::MAX, f64::min);
        let emax = e.iter().cloned().fold(f64::MIN, f64::max);
        let cur = emin / emax + 1e-6;
        let mut star = nearest(argmin(&e));
        let sc = scores(&e, &aim, star);
        let mut f = argmax(&sc);
        if sc[f] <= cur {
            let mut order: Vec<usize> = (0..g_n).collect();
            order.sort_by(|a, b| e[*a].partial_cmp(&e[*b]).unwrap());
            let mut moved = false;
            for &g in order.iter().skip(1).take(7) {
                star = nearest(g);
                let sc = scores(&e, &aim, star);
                f = argmax(&sc);
                if sc[f] > cur {
                    moved = true;
                    break;
                }
            }
            if !moved {
                break;
            }
        }
        let (old, new) = (cr(f, aim[f]).to_vec(), cr(f, star));
        for k in 0..g_n {
            e[k] = e[k] - old[k] + new[k];
        }
        aim[f] = star;
    }
    (aim.iter().map(|&a| candidates[a]).collect(), e)
}

/// Approximate blackbody colour (Tanner Helland), normalised to max 1, then ^2.2.
pub fn kelvin_rgb(k: f64) -> DVec3 {
    let t = k / 100.0;
    let r = if t <= 66.0 { 255.0 } else { 329.7 * (t - 60.0).powf(-0.1332) };
    let g = if t <= 66.0 { 99.47 * t.ln() - 161.1 } else { 288.1 * (t - 60.0).powf(-0.0755) };
    let b = if t >= 66.0 { 255.0 } else if t <= 19.0 { 0.0 } else { 138.5 * (t - 10.0).ln() - 305.0 };
    let c = DVec3::new(clip(r, 0.0, 255.0), clip(g, 0.0, 255.0), clip(b, 0.0, 255.0)) / 255.0;
    let m = c.max_element();
    DVec3::new((c.x / m).powf(2.2), (c.y / m).powf(2.2), (c.z / m).powf(2.2))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mount {
    Mast,
    Roof,
    Pole,
}

#[derive(Clone, Debug, Default)]
pub struct Design {
    pub kind: String,
    pub fixture: String,
    pub kelvin: f64,
    pub fixtures: Vec<(DVec3, DVec3)>, // position, aim target
    pub mounts: Vec<Mount>,
    /// steel: lattice / tube masts, head frames, poles and bars
    pub steel: Vec<Polyline>,
    pub tube_masts: Vec<(DVec3, DVec3, f64)>,
    pub power: f64,
    pub beam: f64,
    pub lux_target: f64,
    pub lux_avg: f64,
    pub u1: f64,
    pub u2: f64,
}

fn lattice(base: DVec2, h: f64) -> Vec<Polyline> {
    let (bx, by) = (base.x, base.y);
    let (w0, w1) = ((h / 14.0).max(2.2), 1.0);
    let segs = 4usize.max((h / 4.0) as usize);
    let zs = linspace(0.0, h, segs + 1);
    let half = |z: f64| (w0 + (w1 - w0) * z / h) / 2.0;
    let corners = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];
    let mut lines = Vec::new();
    for (cx, cy) in corners {
        lines.push((zs.iter().map(|&z| DVec3::new(bx + cx * half(z), by + cy * half(z), z)).collect(), 0.09));
    }
    for w in zs.windows(2) {
        let (a, b) = (w[0], w[1]);
        for k in 0..4 {
            let ((c1x, c1y), (c2x, c2y)) = (corners[k], corners[(k + 1) % 4]);
            let p1a = DVec3::new(bx + c1x * half(a), by + c1y * half(a), a);
            let p2b = DVec3::new(bx + c2x * half(b), by + c2y * half(b), b);
            let p2a = DVec3::new(bx + c2x * half(a), by + c2y * half(a), a);
            let p1b = DVec3::new(bx + c1x * half(b), by + c1y * half(b), b);
            lines.push((vec![p1a, p2b], 0.035));
            lines.push((vec![p2a, p1b], 0.035));
            lines.push((vec![p1b, p2b], 0.04));
        }
    }
    lines
}

/// Stand facts the floodlight layout depends on.
#[derive(Clone, Debug, Default)]
pub struct StandInfo {
    pub a: f64,
    pub b: f64,
    pub reach: f64,
    pub roof_top: f64,
    pub roof_edges: Vec<Vec<DVec3>>,
    /// Some(radius) for a curved bowl
    pub corner_r: Option<f64>,
}

fn n(spec: &Value, k: &str, d: f64) -> f64 {
    spec.get(k).and_then(|v| v.as_f64()).unwrap_or(d)
}

/// Full design for a resolved floodlight spec (preset already merged).
pub fn design(spec: &Value, l: f64, w: f64, runoff_side: f64, stands: Option<&StandInfo>) -> Result<Design, String> {
    let kind = spec.get("type").and_then(|v| v.as_str()).unwrap_or("corner_pylons").to_string();
    let (d_lux, d_fix, d_k) = match kind.as_str() {
        "roof" => (2000.0, "led", 5700.0),
        "poles" => (150.0, "led", 5000.0),
        _ => (1200.0, "halide", 4300.0),
    };
    let mut d = Design {
        kind: kind.clone(),
        fixture: spec.get("fixture").and_then(|v| v.as_str()).unwrap_or(d_fix).to_string(),
        kelvin: n(spec, "kelvin", d_k),
        lux_target: n(spec, "lux", d_lux),
        ..Default::default()
    };
    match kind.as_str() {
        "corner_pylons" => {
            let rows = n(spec, "rows", 5.0) as usize;
            let cols = n(spec, "cols", 7.0) as usize;
            let (reach, roof_top, corner_r) = stands.map(|s| (s.reach, s.roof_top, s.corner_r)).unwrap_or((0.0, 0.0, None));
            let (a, b) = stands.map(|s| (s.a, s.b)).unwrap_or((l / 2.0 + 10.0, w / 2.0 + 8.0));
            let lattice_mast = spec.get("mast").and_then(|v| v.as_str()).unwrap_or("lattice") == "lattice";
            for m in pylons(l, w, a, b, reach, roof_top, corner_r) {
                if lattice_mast {
                    d.steel.extend(lattice(m.base, m.height - 1.5));
                } else {
                    d.tube_masts.push((DVec3::new(m.base.x, m.base.y, 0.0), DVec3::new(m.base.x, m.base.y, m.height - 1.0), 0.55));
                }
                let (pts, _, _) = head_layout(&m, rows, cols);
                for i in 0..pts.len() - 1 {
                    if (i + 1) % cols != 0 {
                        d.steel.push((vec![pts[i], pts[i + 1]], 0.05));
                    }
                }
                for i in 0..pts.len() - cols {
                    d.steel.push((vec![pts[i], pts[i + cols]], 0.05));
                }
                let top = DVec3::new(m.base.x, m.base.y, m.height - 1.5);
                for p in [pts[0], pts[cols - 1], pts[pts.len() - 1], pts[pts.len() - cols]] {
                    d.steel.push((vec![top, p], 0.08));
                }
                for p in pts {
                    d.fixtures.push((p, DVec3::ZERO));
                    d.mounts.push(Mount::Mast);
                }
            }
        }
        "roof" => {
            let lines = stands.map(|s| s.roof_edges.clone()).unwrap_or_default();
            if lines.is_empty() {
                return Err("roof floodlights need roofs on the stands".into());
            }
            // a rail under the roof edge, braced back to the edge girder, carries the fixtures on
            // short hangers
            for line in &lines {
                d.steel.push((line.iter().map(|q| *q - DVec3::new(0.0, 0.0, 0.12)).collect(), 0.06));
                for p in resample(line, n(spec, "spacing", 2.4), 1) {
                    let rail = p - DVec3::new(0.0, 0.0, 0.12);
                    let out = edge_normal(line, p);
                    d.steel.push((vec![rail, rail + DVec3::new(out.x * 0.4, out.y * 0.4, 0.47)], 0.04));
                    d.steel.push((vec![rail, p - DVec3::new(0.0, 0.0, 0.32)], 0.035));
                    d.fixtures.push((p - DVec3::new(0.0, 0.0, 0.45), DVec3::ZERO));
                    d.mounts.push(Mount::Roof);
                }
            }
        }
        "poles" => {
            let offset = runoff_side + 1.2;
            let lamps = n(spec, "lamps", 3.0) as usize;
            for pole in poles(l, w, n(spec, "per_side", 3.0) as usize, n(spec, "height", 16.0), offset) {
                let (bx, by, h) = (pole.base.x, pole.base.y, pole.height);
                d.tube_masts.push((DVec3::new(bx, by, 0.0), DVec3::new(bx, by, h), 0.16));
                for j in 0..lamps {
                    let p = DVec3::new(bx, by, h + 0.3) + DVec3::X * (j as f64 - (lamps as f64 - 1.0) / 2.0) * 0.8;
                    d.fixtures.push((p - DVec3::new(0.0, by.signum() * 0.5, 0.0), DVec3::ZERO));
                    d.mounts.push(Mount::Pole);
                }
                // the crossbar spans every lamp
                let bar = 1.2f64.max(0.4 * (lamps as f64 - 1.0) + 0.4);
                d.tube_masts.push((DVec3::new(bx - bar, by, h), DVec3::new(bx + bar, by, h), 0.05));
            }
        }
        k => return Err(format!("unknown floodlight type {k:?}")),
    }
    if d.fixtures.is_empty() {
        return Ok(d);
    }
    let grid: Vec<DVec3> = linspace(-w / 2.0, w / 2.0, 15)
        .iter()
        .flat_map(|&y| linspace(-l / 2.0, l / 2.0, 22).into_iter().map(move |x| DVec3::new(x, y, 0.0)))
        .collect();
    let cands: Vec<DVec3> = linspace(-w / 2.0, w / 2.0, 9)
        .iter()
        .flat_map(|&y| linspace(-l / 2.0, l / 2.0, 15).into_iter().map(move |x| DVec3::new(x, y, 0.0)))
        .collect();
    let beams: Vec<f64> = match spec.get("beams").and_then(|v| v.as_array()) {
        Some(a) => a.iter().filter_map(|v| v.as_f64()).collect(),
        None if kind == "poles" => vec![70.0, 90.0, 120.0],
        None => vec![30.0, 40.0, 50.0, 60.0],
    };
    let pos: Vec<DVec3> = d.fixtures.iter().map(|f| f.0).collect();
    let mut best: Option<(f64, Vec<DVec3>, f64, f64, Vec<f64>)> = None;
    for beam in beams {
        let (t, e1) = optimise_aims(&pos, &cands, &grid, beam);
        let (u1, u2) = uniformity(&e1);
        if best.as_ref().map_or(true, |b| u1 > b.2) {
            best = Some((beam, t, u1, u2, e1));
        }
    }
    let (beam, targets, u1, u2, e1) = best.unwrap();
    let mean = e1.iter().sum::<f64>() / e1.len() as f64;
    d.power = d.lux_target / mean;
    d.lux_avg = mean * d.power;
    d.beam = beam;
    d.u1 = u1;
    d.u2 = u2;
    for (f, t) in d.fixtures.iter_mut().zip(targets) {
        f.1 = t;
    }
    Ok(d)
}

/// House lights under roof edges so the bowl reads at night: (position, direction) of wide spots
/// with power 4 * area-light power (glTF has no area lights; see kit/core/game.py).
pub fn bowl_lights(roof_edges: &[Vec<DVec3>]) -> (Vec<(DVec3, DVec3)>, f64) {
    let mut out = Vec::new();
    for line in roof_edges {
        for p in resample(line, 12.0, 1) {
            let outward = DVec3::new(p.x, p.y, 0.0).normalize();
            let q = p + outward * 14.0 - DVec3::new(0.0, 0.0, 2.0);
            let dir = DVec3::new(-outward.x * 0.3, -outward.y * 0.3, -1.0).normalize();
            out.push((q, dir));
        }
    }
    let area_power = 150.0 / 683.0 * (12.0 * 30.0) * 2.0;
    (out, 4.0 * area_power)
}
