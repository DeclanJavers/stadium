//! Floodlight layouts and design (port of kit/stands/floodlights.py and the design loop in
//! kit/assets/floodlight.py): corner pylons, roof ribbons, grassroots poles; greedy aim
//! optimisation for uniformity; lamp power from the target lux.

use crate::mesh::{offset_line, outward, MeshAcc};
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
    poles_clear(l, w, per_side, height, offset, POLE_CLEAR_X)
}

/// Poles with the main-stand side kept `clear` metres from halfway (past the main stand's ends).
pub fn poles_clear(l: f64, w: f64, per_side: usize, height: f64, offset: f64, clear: f64) -> Vec<Mast> {
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
            let x = if sy < 0.0 && x.abs() < clear { if x < 0.0 { -clear } else { clear } } else { x };
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
    /// the luminaires drawn: a roof fixture is a 2 x 2 cluster on the catwalk's lamp bars
    pub lamps: Vec<(DVec3, DVec3)>,
    /// scale the luminaires are drawn at (roof floods are larger than the library fixture)
    pub lamp_scale: f64,
    pub mounts: Vec<Mount>,
    /// plates, plinths and cabinets (tags TAG_STEEL, TAG_CONCRETE, TAG_DARK)
    pub mesh: MeshAcc,
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
        lines.push((zs.iter().map(|&z| DVec3::new(bx + cx * half(z), by + cy * half(z), z)).collect(), 0.11));
    }
    for w in zs.windows(2) {
        let (a, b) = (w[0], w[1]);
        for k in 0..4 {
            let ((c1x, c1y), (c2x, c2y)) = (corners[k], corners[(k + 1) % 4]);
            let p1a = DVec3::new(bx + c1x * half(a), by + c1y * half(a), a);
            let p2b = DVec3::new(bx + c2x * half(b), by + c2y * half(b), b);
            let p2a = DVec3::new(bx + c2x * half(a), by + c2y * half(a), a);
            let p1b = DVec3::new(bx + c1x * half(b), by + c1y * half(b), b);
            lines.push((vec![p1a, p2b], 0.03));
            lines.push((vec![p2a, p1b], 0.03));
            lines.push((vec![p1b, p2b], 0.035));
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
    /// how far the main (south) stand reaches either side of halfway (0 = none)
    pub main_half: f64,
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
    // roof fixtures are clusters of lamps; elsewhere every fixture is one lamp
    let mut clusters: Vec<Vec<DVec3>> = Vec::new();
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
                let (st, mesh) = mast_parts(&m, &pts, cols, lattice_mast);
                d.steel.extend(st);
                d.mesh.append(&mesh, 0);
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
            // a catwalk slung under the roof edge carries the fixtures in clusters on its lamp bar
            let (st, mesh, cl) = roof_gantry(&lines, n(spec, "spacing", 2.4));
            d.steel.extend(st);
            d.mesh.append(&mesh, 0);
            for (c, members) in cl {
                d.fixtures.push((c, DVec3::ZERO));
                d.mounts.push(Mount::Roof);
                clusters.push(members);
            }
        }
        "poles" => {
            let offset = runoff_side + 1.2;
            let lamps = n(spec, "lamps", 3.0) as usize;
            // poles on the main-stand side also stand clear of the stand's ends
            let clear = POLE_CLEAR_X.max(stands.map(|s| s.main_half + 4.0).unwrap_or(0.0));
            for pole in poles_clear(l, w, n(spec, "per_side", 3.0) as usize, n(spec, "height", 16.0), offset, clear) {
                let (bx, by, h) = (pole.base.x, pole.base.y, pole.height);
                let (tubes, mesh) = pole_parts(&pole, lamps);
                d.tube_masts.extend(tubes);
                d.mesh.append(&mesh, 0);
                for j in 0..lamps {
                    let p = DVec3::new(bx, by, h + 0.3) + DVec3::X * (j as f64 - (lamps as f64 - 1.0) / 2.0) * 0.8;
                    d.fixtures.push((p - DVec3::new(0.0, by.signum() * 0.5, 0.0), DVec3::ZERO));
                    d.mounts.push(Mount::Pole);
                }
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
    d.lamp_scale = if kind == "roof" { LAMP_SCALE } else { 1.0 };
    d.lamps = if clusters.is_empty() {
        d.fixtures.clone()
    } else {
        d.fixtures.iter().zip(&clusters).flat_map(|(f, m)| m.iter().map(move |q| (*q, f.1))).collect()
    };
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

/// Height of a fixture's trunnion axis (the housing centre) above its origin, the yoke's mounting
/// plate (FIXTURE_PIVOT in kit/assets/floodlight.py).
pub fn fixture_pivot(kind: &str) -> f64 {
    match kind {
        "halide" => 0.33,
        "led" => 0.30,
        _ => 0.0,
    }
}

/// Where a fixture's origin goes so its housing centre sits at `pos` aimed at `target`: the pivot
/// height back along its local -Z (local +Y aims, Blender's track-to Y with Z up).
pub fn mount_point(pos: DVec3, target: DVec3, pivot: f64) -> DVec3 {
    let d = (target - pos).normalize();
    let yaw = (-d.x).atan2(d.y);
    let (st, ct) = d.z.clamp(-1.0, 1.0).asin().sin_cos();
    pos - DVec3::new(st * yaw.sin(), -st * yaw.cos(), ct) * pivot
}

// -- structure detail (catwalks, ladders, platforms, plinths): port of the same in floodlights.py

pub const TAG_STEEL: i32 = 0;
pub const TAG_CONCRETE: i32 = 1;
pub const TAG_DARK: i32 = 2;
/// catwalk deck (the truss's bottom chords) below the roof edge line (m)
pub const GANTRY_DECK: f64 = -1.9;
/// the two lamp bars, above the deck (m)
pub const LAMP_ROWS: [f64; 2] = [1.4, 0.4];
/// lamps hang this far in front of the catwalk's pitch-side face line (m)
pub const LAMP_OUT: f64 = -0.3;
/// roof luminaires are large LED floods (~1.05 x 0.8 m)
pub const LAMP_SCALE: f64 = 1.7;
/// lamps of a cluster sit this far either side of its centre (m)
pub const LAMP_PAIR: f64 = 0.6;

/// A cluster of lamps: (design point, lamp positions).
pub type Cluster = (DVec3, Vec<DVec3>);

/// A catwalk truss slung under the roof's leading edge: top chords under the edge, bottom chords
/// carrying a grating deck with toe boards, a cable tray and handrails, verticals and diagonal
/// lacing on both faces, hangers to the roof, a PA speaker cluster every tenth bay, and on the pitch
/// side two lamp bars carrying the fixtures in 2 x 2 clusters.
pub fn roof_gantry(lines: &[Vec<DVec3>], spacing: f64) -> (Vec<Polyline>, MeshAcc, Vec<Cluster>) {
    let mut acc = MeshAcc::new();
    let mut steel: Vec<Polyline> = Vec::new();
    let mut clusters = Vec::new();
    let v = DVec2::new;
    let d = GANTRY_DECK;
    for line in lines {
        if line.len() < 2 {
            continue;
        }
        let o = outward(line);
        acc.sweep(line, &[v(0.3, d), v(1.5, d), v(1.5, d + 0.07), v(0.3, d + 0.07)], TAG_STEEL, Some(&o), true);
        for s in [0.3, 1.48] {
            acc.sweep(line, &[v(s, d + 0.07), v(s + 0.02, d + 0.07), v(s + 0.02, d + 0.22), v(s, d + 0.22)], TAG_STEEL, Some(&o), true);
        }
        acc.sweep(line, &[v(1.2, d + 0.07), v(1.45, d + 0.07), v(1.45, d + 0.17), v(1.2, d + 0.17)], TAG_DARK, Some(&o), true);
        for (s, dz, r) in [
            (0.3, -0.1, 0.06),
            (1.5, -0.1, 0.06),
            (0.3, d, 0.06),
            (1.5, d, 0.06),
            (0.3, d + 1.05, 0.024),
            (1.5, d + 1.05, 0.024),
            (0.3, d + 0.55, 0.018),
            (1.5, d + 0.55, 0.018),
            (0.0, d + LAMP_ROWS[0], 0.04),
            (0.0, d + LAMP_ROWS[1], 0.04),
        ] {
            steel.push((offset_line(line, &o, s, dz), r));
        }
        for (k, p) in resample(line, spacing, 1).into_iter().enumerate() {
            let n = edge_normal(line, p);
            let tan = DVec3::new(-n.y, n.x, 0.0);
            let at = |s: f64, dz: f64, dt: f64| DVec3::new(p.x + n.x * s, p.y + n.y * s, p.z + dz) + tan * dt;
            for s in [0.3, 1.5] {
                steel.push((vec![at(s, d, 0.0), at(s, -0.1, 0.0)], 0.04));
                steel.push((vec![at(s, d, -0.5 * spacing), at(s, -0.1, 0.0)], 0.03));
                steel.push((vec![at(s, -0.1, 0.0), at(s, d, 0.5 * spacing)], 0.03));
            }
            steel.push((vec![at(0.3, d, 0.0), at(1.5, d, 0.0)], 0.035));
            steel.push((vec![at(0.3, -0.1, 0.0), at(1.5, -0.1, 0.0)], 0.035));
            if k % 2 == 0 {
                steel.push((vec![at(0.9, -0.1, 0.0), at(0.9, 0.35, 0.0)], 0.04));
            }
            if k % 10 == 5 {
                acc.obox(at(0.9, d - 0.6, 0.0), tan, DVec3::new(n.x, n.y, 0.0), 0.3, 0.25, 0.45, TAG_DARK);
            }
            let c = at(LAMP_OUT, d + 0.5 * (LAMP_ROWS[0] + LAMP_ROWS[1]), 0.0);
            let mut members = Vec::new();
            for row in LAMP_ROWS {
                for dt in [-LAMP_PAIR, LAMP_PAIR] {
                    members.push(at(LAMP_OUT, d + row, dt));
                }
            }
            for q in &members {
                steel.push((vec![*q, *q + DVec3::new(n.x, n.y, 0.0) * -LAMP_OUT], 0.025));
            }
            clusters.push((c, members));
        }
    }
    (steel, acc, clusters)
}

fn lattice_half(top: f64) -> impl Fn(f64) -> f64 {
    let (w0, w1) = ((top / 14.0).max(2.2), 1.0);
    move |z| (w0 + (w1 - w0) * z / top) / 2.0
}

/// A corner mast's plinth, caged access ladder (on the face away from the pitch), rest platforms,
/// the service walkway behind the lamp head and a heavier frame round the head.
pub fn mast_parts(m: &Mast, pts: &[DVec3], cols: usize, lattice: bool) -> (Vec<Polyline>, MeshAcc) {
    let mut acc = MeshAcc::new();
    let mut steel: Vec<Polyline> = Vec::new();
    let (bx, by, h) = (m.base.x, m.base.y, m.height);
    let top = h - 1.5;
    let half = lattice_half(top);
    let face = |z: f64| if lattice { half(z) } else { 0.55 };
    if lattice {
        let e = half(0.0) + 0.6;
        acc.zbox(DVec3::new(bx - e, by - e, -0.04), DVec3::new(bx + e, by + e, 0.35), TAG_CONCRETE);
    } else {
        acc.zbox(DVec3::new(bx - 1.3, by - 1.3, -0.04), DVec3::new(bx + 1.3, by + 1.3, 0.3), TAG_CONCRETE);
        acc.zbox(DVec3::new(bx - 0.85, by - 0.85, 0.3), DVec3::new(bx + 0.85, by + 0.85, 0.36), TAG_STEEL);
    }
    let yaw = (-by).atan2(-bx);
    let tilt = h.atan2(bx.hypot(by));
    let fwd = DVec3::new(yaw.cos(), yaw.sin(), 0.0);
    let side = DVec3::new(-yaw.sin(), yaw.cos(), 0.0);
    let up = DVec3::new(0.0, 0.0, tilt.cos()) + fwd * tilt.sin();
    let z_walk = pts.iter().map(|p| p.z).fold(f64::MAX, f64::min) - 0.7;
    let sx = if bx >= 0.0 { 1.0 } else { -1.0 };
    let lx = |z: f64, d: f64| bx + sx * (face(z) + d);
    let z0 = 0.35;
    for dy in [-0.22, 0.22] {
        steel.push((vec![DVec3::new(lx(z0, 0.3), by + dy, z0), DVec3::new(lx(z_walk, 0.3), by + dy, z_walk + 1.1)], 0.025));
    }
    let n_r = ((z_walk - z0) / 0.3) as i64;
    for k in 1..=n_r {
        let z = z0 + k as f64 * 0.3;
        steel.push((vec![DVec3::new(lx(z, 0.3), by - 0.22, z), DVec3::new(lx(z, 0.3), by + 0.22, z)], 0.012));
    }
    let angles = linspace(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2, 7);
    let mut hoops = Vec::new();
    let mut z = 2.5;
    while z < z_walk {
        hoops.push(z);
        z += 1.0;
    }
    let hoop_pt = |z: f64, a: f64| DVec3::new(lx(z, 0.3) + sx * 0.45 * a.cos(), by + 0.38 * a.sin(), z);
    for &z in &hoops {
        steel.push((angles.iter().map(|&a| hoop_pt(z, a)).collect(), 0.014));
    }
    if hoops.len() > 1 {
        for &a in &angles[1..angles.len() - 1] {
            steel.push((hoops.iter().map(|&z| hoop_pt(z, a)).collect(), 0.012));
        }
    }
    let mut z = 12.0;
    while z < z_walk - 4.0 {
        let x0 = lx(z, 0.0);
        acc.obox(DVec3::new(x0 + sx * 0.55, by, z - 0.03), DVec3::X, DVec3::Y, 0.55, 0.65, 0.03, TAG_STEEL);
        let rail = vec![DVec3::new(x0, by - 0.65, z + 1.0), DVec3::new(x0 + sx * 1.1, by - 0.65, z + 1.0), DVec3::new(x0 + sx * 1.1, by + 0.65, z + 1.0), DVec3::new(x0, by + 0.65, z + 1.0)];
        steel.push((rail.clone(), 0.022));
        for q in &rail[1..3] {
            steel.push((vec![DVec3::new(q.x, q.y, z), *q], 0.02));
        }
        z += 12.0;
    }
    let hs = (cols as f64 - 1.0) / 2.0 * 0.8 + 0.5;
    let c = DVec3::new(bx, by, z_walk - 0.03) + fwd * 0.7;
    acc.obox(c, side, fwd, hs, 0.9, 0.03, TAG_STEEL);
    let ring: Vec<DVec3> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)].iter().map(|&(a, b)| c + side * a * hs + fwd * b * 0.9).collect();
    steel.push((ring.iter().map(|q| *q + DVec3::new(0.0, 0.0, 1.1)).collect(), 0.022));
    for q in &ring[..4] {
        steel.push((vec![*q, *q + DVec3::new(0.0, 0.0, 1.1)], 0.02));
    }
    let n = pts.len();
    let corners = [pts[0], pts[cols - 1], pts[n - 1], pts[n - cols]];
    let mid = (corners[0] + corners[1] + corners[2] + corners[3]) / 4.0;
    let mut frame = Vec::new();
    for q in corners.iter().chain(corners.iter().take(1)) {
        let d = *q - mid;
        frame.push(*q + side * if d.dot(side) > 0.0 { 0.45 } else { -0.45 } + up * if d.dot(up) > 0.0 { 0.42 } else { -0.42 } - fwd * 0.1);
    }
    steel.push((frame, 0.07));
    (steel, acc)
}

/// A grassroots pole: three telescoped sections with collars, the lamp crossbar, footing, base
/// flange, access door, control gear at the head and a feeder pillar.
pub fn pole_parts(pole: &Mast, lamps: usize) -> (Vec<(DVec3, DVec3, f64)>, MeshAcc) {
    let mut acc = MeshAcc::new();
    let (bx, by, h) = (pole.base.x, pole.base.y, pole.height);
    let mut tubes = Vec::new();
    let cuts = [0.0, 0.4 * h, 0.75 * h, h];
    let radii = [0.2, 0.165, 0.13];
    for k in 0..3 {
        tubes.push((DVec3::new(bx, by, cuts[k]), DVec3::new(bx, by, cuts[k + 1]), radii[k]));
    }
    for k in [1, 2] {
        tubes.push((DVec3::new(bx, by, cuts[k] - 0.1), DVec3::new(bx, by, cuts[k] + 0.06), radii[k - 1] + 0.012));
    }
    let bar = 1.2f64.max(0.4 * (lamps as f64 - 1.0) + 0.4);
    tubes.push((DVec3::new(bx - bar, by, h), DVec3::new(bx + bar, by, h), 0.05));
    tubes.push((DVec3::new(bx - bar, by, h - 0.45), DVec3::new(bx + bar, by, h - 0.45), 0.035));
    for x in [bx - bar + 0.1, bx + bar - 0.1] {
        tubes.push((DVec3::new(x, by, h - 0.45), DVec3::new(x, by, h), 0.03));
    }
    let sgn = if by >= 0.0 { 1.0 } else { -1.0 };
    acc.zbox(DVec3::new(bx - 0.6, by - 0.6, -0.04), DVec3::new(bx + 0.6, by + 0.6, 0.15), TAG_CONCRETE);
    acc.zbox(DVec3::new(bx - 0.32, by - 0.32, 0.15), DVec3::new(bx + 0.32, by + 0.32, 0.19), TAG_STEEL);
    acc.obox(DVec3::new(bx, by + sgn * 0.2, 0.95), DVec3::X, DVec3::Y, 0.1, 0.025, 0.3, TAG_DARK);
    acc.obox(DVec3::new(bx, by + sgn * 0.3, h - 1.1), DVec3::X, DVec3::Y, 0.22, 0.14, 0.3, TAG_DARK);
    acc.obox(DVec3::new(bx + 0.9, by + sgn * 0.7, 0.6), DVec3::X, DVec3::Y, 0.3, 0.2, 0.6, TAG_DARK);
    (tubes, acc)
}

/// half width, half height and forward offset (in front of the bezel, clear of the lens) of an LED lens
pub const LENS_PANEL: (f64, f64, f64) = (0.27, 0.19, 0.076);

/// The glowing diffuser panel over each LED fixture's lens (a quad just in front of the lens,
/// facing the aim): what makes a roof ring read as lit luminaires from across the ground.
/// lamps: (housing centre, aim target).
pub fn lens_panels(lamps: &[(DVec3, DVec3)], scale: f64) -> MeshAcc {
    let mut acc = MeshAcc::new();
    let (hx, hz, fy) = (LENS_PANEL.0 * scale, LENS_PANEL.1 * scale, LENS_PANEL.2 * scale);
    for (q, t) in lamps {
        let d = (*t - *q).normalize();
        let yaw = (-d.x).atan2(d.y);
        let (st, ct) = d.z.clamp(-1.0, 1.0).asin().sin_cos();
        let ax = DVec3::new(yaw.cos(), yaw.sin(), 0.0);
        let az = DVec3::new(st * yaw.sin(), -st * yaw.cos(), ct);
        let c = *q + d * fy;
        let base = acc.verts.len() as u32;
        acc.verts.extend([c - ax * hx - az * hz, c + ax * hx - az * hz, c + ax * hx + az * hz, c - ax * hx + az * hz]);
        acc.add_face(vec![base, base + 3, base + 2, base + 1], 0, Some(vec![DVec2::new(0.0, 0.0), DVec2::new(0.0, 1.0), DVec2::new(1.0, 1.0), DVec2::new(1.0, 0.0)]));
    }
    acc
}
