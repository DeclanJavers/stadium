//! Geometry kernel for the training-ground generators (port of kit/site/kernel.py): the same
//! operations in the same order, so both produce identical vertices. Generators work in a plot's
//! local frame (origin at the plot centre, x along its width, y along its depth, z up) and a
//! Frame moves the result into the site.

use crate::mesh::MeshAcc;
use crate::stands::Polyline;
use glam::{DVec2, DVec3};

/// A plot's frame: local (u, v) -> site (x, y), rotated by yaw about (x, y).
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
    pub c: f64,
    pub s: f64,
}

impl Frame {
    pub fn new(x: f64, y: f64, yaw_deg: f64) -> Self {
        let yaw = yaw_deg.to_radians();
        Frame { x, y, yaw, c: yaw.cos(), s: yaw.sin() }
    }

    /// A frame from a yaw in radians.
    pub fn rad(x: f64, y: f64, yaw: f64) -> Self {
        Frame { x, y, yaw, c: yaw.cos(), s: yaw.sin() }
    }

    /// A frame inside this one: origin at local (u, v), turned by a further local yaw (radians).
    pub fn sub(&self, u: f64, v: f64, yaw: f64) -> Frame {
        let p = self.pt(u, v);
        Frame::rad(p.x, p.y, self.yaw + yaw)
    }

    pub fn pt(&self, u: f64, v: f64) -> DVec2 {
        DVec2::new(self.x + u * self.c - v * self.s, self.y + u * self.s + v * self.c)
    }

    pub fn vec(&self, u: f64, v: f64) -> DVec2 {
        DVec2::new(u * self.c - v * self.s, u * self.s + v * self.c)
    }

    pub fn mesh(&self, m: &mut MeshAcc) {
        for p in m.verts.iter_mut() {
            *p = DVec3::new(self.x + p.x * self.c - p.y * self.s, self.y + p.x * self.s + p.y * self.c, p.z);
        }
    }

    pub fn tubes(&self, lines: &[Polyline]) -> Vec<Polyline> {
        lines.iter().map(|(pts, r)| (pts.iter().map(|p| DVec3::new(self.x + p.x * self.c - p.y * self.s, self.y + p.x * self.s + p.y * self.c, p.z)).collect(), *r)).collect()
    }
}

pub fn add_verts(m: &mut MeshAcc, pts: &[DVec3]) -> u32 {
    let base = m.verts.len() as u32;
    m.verts.extend_from_slice(pts);
    base
}

/// Axis-aligned box, outward quads, metric UVs on the two axes each face spans.
#[allow(clippy::too_many_arguments)]
pub fn bx(m: &mut MeshAcc, x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64, tag: i32) {
    let base = m.verts.len() as u32;
    for z in [z0, z1] {
        for y in [y0, y1] {
            for x in [x0, x1] {
                m.verts.push(DVec3::new(x, y, z));
            }
        }
    }
    for f in [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]] {
        let q: Vec<u32> = f.iter().map(|i| base + i).collect();
        let pts: Vec<DVec3> = q.iter().map(|&i| m.verts[i as usize]).collect();
        let n = (pts[1] - pts[0]).cross(pts[2] - pts[0]).abs();
        let uv = pts
            .iter()
            .map(|p| if n.x >= n.y && n.x >= n.z { DVec2::new(p.y, p.z) } else if n.y >= n.z { DVec2::new(p.x, p.z) } else { DVec2::new(p.x, p.y) })
            .collect();
        m.add_face(q, tag, Some(uv));
    }
}

/// Box of size (w, d, h) standing on z0, centred on (cx, cy), turned by yaw (radians).
#[allow(clippy::too_many_arguments)]
pub fn obox(m: &mut MeshAcc, cx: f64, cy: f64, z0: f64, w: f64, d: f64, h: f64, yaw: f64, tag: i32) {
    let mut sub = MeshAcc::new();
    bx(&mut sub, -w / 2.0, -d / 2.0, z0, w / 2.0, d / 2.0, z0 + h, tag);
    let (c, s) = (yaw.cos(), yaw.sin());
    for p in sub.verts.iter_mut() {
        *p = DVec3::new(cx + p.x * c - p.y * s, cy + p.x * s + p.y * c, p.z);
    }
    m.append(&sub, 0);
}

/// Metric UVs in a polygon's plane: u along its first edge.
pub fn planar_uv(pts: &[DVec3]) -> Vec<DVec2> {
    let a = pts[0];
    let eu = pts[1] - a;
    let lu = eu.length();
    if lu < 1e-9 {
        return vec![DVec2::ZERO; pts.len()];
    }
    let eu = eu / lu;
    let w = pts[pts.len() - 1] - a;
    let k = w.dot(eu);
    let ev = w - eu * k;
    let lv = ev.length();
    if lv < 1e-9 {
        return pts.iter().map(|p| DVec2::new((*p - a).dot(eu), 0.0)).collect();
    }
    let ev = ev / lv;
    pts.iter().map(|p| DVec2::new((*p - a).dot(eu), (*p - a).dot(ev))).collect()
}

/// Planar quad a-b-c-d (counter-clockwise seen from its front); UVs: u along a->b.
pub fn quad(m: &mut MeshAcc, a: DVec3, b: DVec3, c: DVec3, d: DVec3, tag: i32) {
    let base = add_verts(m, &[a, b, c, d]);
    m.add_face(vec![base, base + 1, base + 2, base + 3], tag, Some(planar_uv(&[a, b, c, d])));
}

pub fn tri(m: &mut MeshAcc, a: DVec3, b: DVec3, c: DVec3, tag: i32) {
    let base = add_verts(m, &[a, b, c]);
    m.add_face(vec![base, base + 1, base + 2], tag, Some(planar_uv(&[a, b, c])));
}

/// Twice the signed area of a 2D polygon (positive = counter-clockwise).
pub fn area2(pts: &[DVec2]) -> f64 {
    let mut s = 0.0;
    let n = pts.len();
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        s += a.x * b.y - b.x * a.y;
    }
    s
}

pub fn ccw(pts: &[DVec2]) -> Vec<DVec2> {
    if area2(pts) >= 0.0 {
        pts.to_vec()
    } else {
        pts.iter().rev().cloned().collect()
    }
}

/// One flat polygon facing up (any simple polygon), UVs = site metres.
pub fn ground(m: &mut MeshAcc, pts: &[DVec2], z: f64, tag: i32) {
    let pts = ccw(pts);
    let base = add_verts(m, &pts.iter().map(|p| DVec3::new(p.x, p.y, z)).collect::<Vec<_>>());
    m.add_face((base..base + pts.len() as u32).collect(), tag, Some(pts.clone()));
}

/// Vertical extrusion of a 2D polygon: walls, a top, optionally a bottom.
pub fn prism(m: &mut MeshAcc, pts: &[DVec2], z0: f64, z1: f64, side_tag: i32, top_tag: Option<i32>, bottom_tag: Option<i32>) {
    let pts = ccw(pts);
    let n = pts.len();
    let mut s = 0.0;
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        let ln = (b.x - a.x).hypot(b.y - a.y);
        let base = add_verts(m, &[DVec3::new(a.x, a.y, z0), DVec3::new(b.x, b.y, z0), DVec3::new(b.x, b.y, z1), DVec3::new(a.x, a.y, z1)]);
        m.add_face(vec![base, base + 1, base + 2, base + 3], side_tag, Some(vec![DVec2::new(s, z0), DVec2::new(s + ln, z0), DVec2::new(s + ln, z1), DVec2::new(s, z1)]));
        s += ln;
    }
    if let Some(t) = top_tag {
        let base = add_verts(m, &pts.iter().map(|p| DVec3::new(p.x, p.y, z1)).collect::<Vec<_>>());
        m.add_face((base..base + n as u32).collect(), t, Some(pts.clone()));
    }
    if let Some(t) = bottom_tag {
        let rev: Vec<DVec2> = pts.iter().rev().cloned().collect();
        let base = add_verts(m, &rev.iter().map(|p| DVec3::new(p.x, p.y, z0)).collect::<Vec<_>>());
        m.add_face((base..base + n as u32).collect(), t, Some(rev));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn cylinder(m: &mut MeshAcc, cx: f64, cy: f64, z0: f64, z1: f64, r: f64, n: usize, tag: i32, cap: bool) {
    let pi = std::f64::consts::PI;
    let mut s = 0.0;
    let step = 2.0 * pi * r / n as f64;
    for k in 0..n {
        let (a0, a1) = (2.0 * pi * k as f64 / n as f64, 2.0 * pi * (k + 1) as f64 / n as f64);
        let p0 = DVec2::new(cx + r * a0.cos(), cy + r * a0.sin());
        let p1 = DVec2::new(cx + r * a1.cos(), cy + r * a1.sin());
        let base = add_verts(m, &[DVec3::new(p0.x, p0.y, z0), DVec3::new(p1.x, p1.y, z0), DVec3::new(p1.x, p1.y, z1), DVec3::new(p0.x, p0.y, z1)]);
        m.add_face(vec![base, base + 1, base + 2, base + 3], tag, Some(vec![DVec2::new(s, z0), DVec2::new(s + step, z0), DVec2::new(s + step, z1), DVec2::new(s, z1)]));
        s += step;
    }
    if cap {
        let pts: Vec<DVec2> = (0..n).map(|k| DVec2::new(cx + r * (2.0 * pi * k as f64 / n as f64).cos(), cy + r * (2.0 * pi * k as f64 / n as f64).sin())).collect();
        let base = add_verts(m, &pts.iter().map(|p| DVec3::new(p.x, p.y, z1)).collect::<Vec<_>>());
        m.add_face((base..base + n as u32).collect(), tag, Some(pts));
    }
}

/// Flat ribbon along an open 2D polyline (mitred joins), UVs (along, across) in metres.
/// `z_of(x, y)` overrides the height per vertex.
pub fn strip(m: &mut MeshAcc, path: &[DVec2], width: f64, z: f64, tag: i32, z_of: Option<&dyn Fn(f64, f64) -> f64>) {
    let n = path.len();
    if n < 2 {
        return;
    }
    let ns: Vec<DVec2> = (0..n - 1)
        .map(|i| {
            let (dx, dy) = (path[i + 1].x - path[i].x, path[i + 1].y - path[i].y);
            let ln = dx.hypot(dy);
            if ln > 1e-12 {
                DVec2::new(-dy / ln, dx / ln)
            } else {
                DVec2::new(0.0, 1.0)
            }
        })
        .collect();
    let mut nrm = vec![ns[0]; n];
    nrm[n - 1] = ns[n - 2];
    for i in 1..n - 1 {
        let (mut mx, mut my) = (ns[i - 1].x + ns[i].x, ns[i - 1].y + ns[i].y);
        let ml = mx.hypot(my);
        if ml > 1e-12 {
            mx /= ml;
            my /= ml;
        } else {
            mx = ns[i].x;
            my = ns[i].y;
        }
        let k = (mx * ns[i].x + my * ns[i].y).max(0.3);
        nrm[i] = DVec2::new(mx / k, my / k);
    }
    let h = width / 2.0;
    let base = m.verts.len() as u32;
    for i in 0..n {
        let p = path[i];
        for sgn in [-1.0, 1.0] {
            let (x, y) = (p.x + nrm[i].x * h * sgn, p.y + nrm[i].y * h * sgn);
            m.verts.push(DVec3::new(x, y, z_of.map_or(z, |f| f(x, y))));
        }
    }
    let mut s = 0.0;
    for i in 0..n - 1 {
        let ln = (path[i + 1].x - path[i].x).hypot(path[i + 1].y - path[i].y);
        let a = base + 2 * i as u32;
        m.add_face(vec![a, a + 2, a + 3, a + 1], tag, Some(vec![DVec2::new(s, 0.0), DVec2::new(s + ln, 0.0), DVec2::new(s + ln, width), DVec2::new(s, width)]));
        s += ln;
    }
}

pub fn polyline_length(pts: &[DVec2]) -> f64 {
    (0..pts.len().saturating_sub(1)).map(|i| (pts[i + 1].x - pts[i].x).hypot(pts[i + 1].y - pts[i].y)).sum()
}

/// Points every ~spacing metres along a 2D polyline (evenly divided) with the unit direction at
/// each: (point, direction). offset 0.5 = centred in each division, 0 = from the start (and the end).
pub fn along(pts: &[DVec2], spacing: f64, offset: f64) -> Vec<(DVec2, DVec2)> {
    let total = polyline_length(pts);
    if total <= 1e-9 {
        return vec![];
    }
    let n = 1usize.max((total / spacing) as i64 as usize);
    let mut out = Vec::new();
    for k in 0..n + usize::from(offset == 0.0) {
        let t = (k as f64 + offset) * total / n as f64;
        let mut acc = 0.0;
        for i in 0..pts.len() - 1 {
            let (a, b) = (pts[i], pts[i + 1]);
            let ln = (b.x - a.x).hypot(b.y - a.y);
            if t <= acc + ln || i == pts.len() - 2 {
                let f = if ln <= 1e-12 { 0.0 } else { ((t - acc) / ln).clamp(0.0, 1.0) };
                let d = if ln > 1e-12 { DVec2::new((b.x - a.x) / ln, (b.y - a.y) / ln) } else { DVec2::new(1.0, 0.0) };
                out.push((DVec2::new(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f), d));
                break;
            }
            acc += ln;
        }
    }
    out
}

/// Counter-clockwise rectangle corners.
pub fn rect(w: f64, d: f64, cx: f64, cy: f64) -> Vec<DVec2> {
    vec![DVec2::new(cx - w / 2.0, cy - d / 2.0), DVec2::new(cx + w / 2.0, cy - d / 2.0), DVec2::new(cx + w / 2.0, cy + d / 2.0), DVec2::new(cx - w / 2.0, cy + d / 2.0)]
}

/// Point in polygon (even-odd).
pub fn inside(pts: &[DVec2], x: f64, y: f64) -> bool {
    let mut c = false;
    let n = pts.len();
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        if (a.y > y) != (b.y > y) {
            let xi = a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y);
            if x < xi {
                c = !c;
            }
        }
    }
    c
}

/// Ray o + t d against segment a-b: t >= 0.
pub fn seg_hit(o: DVec2, d: DVec2, a: DVec2, b: DVec2) -> Option<f64> {
    let (ex, ey) = (b.x - a.x, b.y - a.y);
    let den = d.x * ey - d.y * ex;
    if den.abs() < 1e-12 {
        return None;
    }
    let (wx, wy) = (a.x - o.x, a.y - o.y);
    let t = (wx * ey - wy * ex) / den;
    let u = (wx * d.y - wy * d.x) / den;
    (t >= 0.0 && (-1e-9..=1.0 + 1e-9).contains(&u)).then_some(t)
}

pub fn dist_to_polyline(p: DVec2, pts: &[DVec2]) -> f64 {
    let mut best = f64::INFINITY;
    for i in 0..pts.len().saturating_sub(1) {
        let (a, b) = (pts[i], pts[i + 1]);
        let (ex, ey) = (b.x - a.x, b.y - a.y);
        let l2 = ex * ex + ey * ey;
        let t = if l2 <= 1e-12 { 0.0 } else { (((p.x - a.x) * ex + (p.y - a.y) * ey) / l2).clamp(0.0, 1.0) };
        best = best.min((a.x + ex * t - p.x).hypot(a.y + ey * t - p.y));
    }
    best
}
