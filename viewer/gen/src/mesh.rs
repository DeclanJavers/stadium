//! Polygon accumulation (port of geometry.MeshAcc), tubes, and conversion to triangle meshes with
//! normals for rendering.

use crate::py::linspace;
use crate::runs::Run;
use glam::{DVec2, DVec3, Vec2, Vec3};

/// A closed section polygon in (s, z) with one tag per edge.
#[derive(Clone, Debug)]
pub struct Section {
    pub pts: Vec<DVec2>,
    pub tags: Vec<i32>,
}

impl Section {
    pub fn new(pts: Vec<DVec2>, tags: Vec<i32>) -> Self {
        Section { pts, tags }
    }
    pub fn uniform(pts: Vec<DVec2>, tag: i32) -> Self {
        let n = pts.len();
        Section { pts, tags: vec![tag; n] }
    }
}

/// Mixed-size polygons, per-face tags and per-loop UVs.
#[derive(Clone, Debug, Default)]
pub struct MeshAcc {
    pub verts: Vec<DVec3>,
    pub faces: Vec<Vec<u32>>,
    pub tags: Vec<i32>,
    pub uvs: Vec<DVec2>,
    /// Faces whose winding points inward (counter-clockwise sections); Blender fixes these with a
    /// normals recalculation, the port flips them when triangulating.
    pub flip: Vec<bool>,
}

impl MeshAcc {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n(&self) -> usize {
        self.verts.len()
    }

    pub fn add_prism(&mut self, run: &Run, ta: f64, tb: f64, sec: &Section, cap_tag: i32) {
        let poly = &sec.pts;
        let k = poly.len();
        let smax = poly.iter().map(|p| p.x).fold(f64::MIN, f64::max);
        let ts = run.samples(ta, tb, smax);
        let area: f64 = (0..k).map(|i| poly[i].perp_dot(poly[(i + 1) % k])).sum();
        let inward = area > 0.0;
        let j = ts.len();
        let base = self.verts.len() as u32;
        let mut per = vec![0.0];
        for i in 0..k {
            let d = poly[(i + 1) % k] - poly[i];
            per.push(per[i] + d.length());
        }
        for &t in &ts {
            let (p, n) = run.frame(t);
            for q in poly {
                let xy = p + n * q.x;
                self.verts.push(DVec3::new(xy.x, xy.y, q.y));
            }
        }
        let m0 = run.metric(0.0);
        let along: Vec<f64> = ts.iter().map(|t| (t - ts[0]) * m0).collect();
        let ku = k as u32;
        for jj in 0..j.saturating_sub(1) {
            for i in 0..k {
                let a = base + jj as u32 * ku + i as u32;
                let b = base + jj as u32 * ku + ((i + 1) % k) as u32;
                self.faces.push(vec![a, b, b + ku, a + ku]);
                self.tags.push(sec.tags[i]);
                self.flip.push(inward);
                self.uvs.extend_from_slice(&[
                    DVec2::new(along[jj], per[i]),
                    DVec2::new(along[jj], per[i + 1]),
                    DVec2::new(along[jj + 1], per[i + 1]),
                    DVec2::new(along[jj + 1], per[i]),
                ]);
            }
        }
        for (jj, flip) in [(0usize, true), (j - 1, false)] {
            let mut idx: Vec<u32> = (0..k).map(|i| base + jj as u32 * ku + i as u32).collect();
            let mut uv: Vec<DVec2> = poly.clone();
            if flip {
                idx.reverse();
                uv.reverse();
            }
            self.faces.push(idx);
            self.tags.push(cap_tag);
            self.flip.push(inward);
            self.uvs.extend(uv);
        }
    }

    /// Append a quad grid / arbitrary polygons with zero UVs.
    pub fn add_face(&mut self, idx: Vec<u32>, tag: i32, uv: Option<Vec<DVec2>>) {
        let n = idx.len();
        self.uvs.extend(uv.unwrap_or_else(|| vec![DVec2::ZERO; n]));
        self.faces.push(idx);
        self.tags.push(tag);
        self.flip.push(false);
    }

    pub fn append(&mut self, other: &MeshAcc, tag_offset: i32) {
        let base = self.verts.len() as u32;
        self.verts.extend_from_slice(&other.verts);
        for f in &other.faces {
            self.faces.push(f.iter().map(|i| i + base).collect());
        }
        self.tags.extend(other.tags.iter().map(|t| t + tag_offset));
        self.flip.extend_from_slice(&other.flip);
        self.uvs.extend_from_slice(&other.uvs);
    }
}

/// Horizontal unit normal at every vertex of a polyline (central differences), pointing away from
/// the pitch centre (port of shapes.outward).
pub fn outward(line: &[DVec3]) -> Vec<DVec2> {
    let n = line.len();
    (0..n)
        .map(|i| {
            let (a, b) = (line[i.saturating_sub(1)], line[(i + 1).min(n - 1)]);
            let (mut tx, mut ty) = (b.x - a.x, b.y - a.y);
            let l = tx.hypot(ty);
            if l > 1e-12 {
                tx /= l;
                ty /= l;
            } else {
                (tx, ty) = (1.0, 0.0);
            }
            let o = DVec2::new(ty, -tx);
            if o.x * line[i].x + o.y * line[i].y < 0.0 { -o } else { o }
        })
        .collect()
}

/// The polyline moved `o` outward and `dz` up (shapes.offset).
pub fn offset_line(line: &[DVec3], normals: &[DVec2], o: f64, dz: f64) -> Vec<DVec3> {
    line.iter().zip(normals).map(|(p, n)| DVec3::new(p.x + n.x * o, p.y + n.y * o, p.z + dz)).collect()
}

const BOX_FACES: [[usize; 4]; 6] = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];

impl MeshAcc {
    /// Closed section [(o outward, dz up)] swept along a polyline, one ring per vertex
    /// (shapes.sweep).
    pub fn sweep(&mut self, line: &[DVec3], section: &[DVec2], tag: i32, normals: Option<&[DVec2]>, caps: bool) {
        let j = line.len();
        if j < 2 {
            return;
        }
        let own;
        let o = match normals {
            Some(n) => n,
            None => {
                own = outward(line);
                &own[..]
            }
        };
        let k = section.len();
        let base = self.verts.len() as u32;
        for (p, n) in line.iter().zip(o) {
            for q in section {
                self.verts.push(DVec3::new(p.x + n.x * q.x, p.y + n.y * q.x, p.z + q.y));
            }
        }
        let mut along = vec![0.0];
        for w in line.windows(2) {
            along.push(along[along.len() - 1] + (w[1].x - w[0].x).hypot(w[1].y - w[0].y));
        }
        let mut per = vec![0.0];
        for i in 0..k {
            per.push(per[i] + (section[(i + 1) % k] - section[i]).length());
        }
        // winding as add_prism's (whose frame has t x n pointing down), mirrored for a line that
        // runs the other way round
        let area: f64 = (0..k).map(|i| section[i].perp_dot(section[(i + 1) % k])).sum();
        let t0 = (line[1] - line[0]).truncate();
        let inward = (area > 0.0) == (t0.x * o[0].y - t0.y * o[0].x <= 0.0);
        let ku = k as u32;
        for jj in 0..j - 1 {
            for i in 0..k {
                let a = base + jj as u32 * ku + i as u32;
                let b = base + jj as u32 * ku + ((i + 1) % k) as u32;
                self.faces.push(vec![a, b, b + ku, a + ku]);
                self.tags.push(tag);
                self.flip.push(inward);
                self.uvs.extend_from_slice(&[
                    DVec2::new(along[jj], per[i]),
                    DVec2::new(along[jj], per[i + 1]),
                    DVec2::new(along[jj + 1], per[i + 1]),
                    DVec2::new(along[jj + 1], per[i]),
                ]);
            }
        }
        if caps {
            for (jj, flip) in [(0usize, true), (j - 1, false)] {
                let mut idx: Vec<u32> = (0..k).map(|i| base + jj as u32 * ku + i as u32).collect();
                let mut uv: Vec<DVec2> = section.to_vec();
                if flip {
                    idx.reverse();
                    uv.reverse();
                }
                self.faces.push(idx);
                self.tags.push(tag);
                self.flip.push(inward);
                self.uvs.extend(uv);
            }
        }
    }

    /// Box centred at c with half extents along ax, ay and az = ax x ay; faces wind outward
    /// (shapes.obox).
    #[allow(clippy::too_many_arguments)]
    pub fn obox(&mut self, c: DVec3, ax: DVec3, ay: DVec3, hx: f64, hy: f64, hz: f64, tag: i32) {
        let az = ax.cross(ay);
        let base = self.verts.len() as u32;
        let mut loc = Vec::with_capacity(8);
        for z in [-hz, hz] {
            for y in [-hy, hy] {
                for x in [-hx, hx] {
                    loc.push(DVec3::new(x, y, z));
                }
            }
        }
        for l in &loc {
            self.verts.push(c + ax * l.x + ay * l.y + az * l.z);
        }
        for f in BOX_FACES {
            let q: Vec<DVec3> = f.iter().map(|&i| loc[i]).collect();
            let n = (q[1] - q[0]).cross(q[2] - q[0]).abs();
            let uv = q.iter().map(|v| if n.x >= n.y && n.x >= n.z { DVec2::new(v.y, v.z) } else if n.y >= n.z { DVec2::new(v.x, v.z) } else { DVec2::new(v.x, v.y) }).collect();
            self.add_face(f.iter().map(|&i| base + i as u32).collect(), tag, Some(uv));
        }
    }

    /// Axis-aligned box between two corners (shapes.zbox).
    pub fn zbox(&mut self, lo: DVec3, hi: DVec3, tag: i32) {
        let (c, h) = ((lo + hi) / 2.0, (hi - lo) / 2.0);
        self.obox(c, DVec3::X, DVec3::Y, h.x, h.y, h.z, tag);
    }
}

/// Independent prisms along segments (port of mesh.tubes): verts, quads.
pub fn tubes(p0: &[DVec3], p1: &[DVec3], radius: f64, sides: usize, extend: Option<f64>) -> (Vec<DVec3>, Vec<[u32; 4]>) {
    let ext = extend.unwrap_or(radius);
    let ang = linspace(0.0, 2.0 * std::f64::consts::PI, sides + 1);
    let mut v = Vec::new();
    let mut q = Vec::new();
    for (a, b) in p0.iter().zip(p1) {
        let d = *b - *a;
        let l = d.length();
        if l <= 1e-6 {
            continue;
        }
        let d = d / l;
        let a = *a - d * ext;
        let b = *b + d * ext;
        let rf = if d.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
        let n1 = d.cross(rf).normalize();
        let n2 = d.cross(n1);
        let base = v.len() as u32;
        for p in [a, b] {
            for k in 0..sides {
                v.push(p + radius * (ang[k].cos() * n1 + ang[k].sin() * n2));
            }
        }
        let s = sides as u32;
        for j in 0..s {
            let aa = base + j;
            let bb = base + (j + 1) % s;
            q.push([aa, bb, bb + s, aa + s]);
        }
    }
    (v, q)
}

/// Polylines with radii -> one tube mesh (rails, trusses).
pub fn tube_lines(lines: &[(Vec<DVec3>, f64)], sides: usize) -> MeshAcc {
    let mut acc = MeshAcc::new();
    for (pts, r) in lines {
        if pts.len() < 2 {
            continue;
        }
        let (v, q) = tubes(&pts[..pts.len() - 1], &pts[1..], *r, sides, Some(r * 0.5));
        let base = acc.verts.len() as u32;
        acc.verts.extend(v);
        for f in q {
            acc.add_face(f.iter().map(|i| i + base).collect(), 0, None);
        }
    }
    acc
}

/// A triangle mesh ready for the GPU (f32), one index list per tag.
#[derive(Clone, Debug, Default)]
pub struct TriMesh {
    pub pos: Vec<Vec3>,
    pub nrm: Vec<Vec3>,
    pub uv: Vec<Vec2>,
    pub groups: Vec<(i32, Vec<u32>)>,
}

impl TriMesh {
    pub fn tri_count(&self) -> usize {
        self.groups.iter().map(|g| g.1.len() / 3).sum()
    }
}

fn newell(v: &[DVec3]) -> DVec3 {
    let mut n = DVec3::ZERO;
    for i in 0..v.len() {
        let a = v[i];
        let b = v[(i + 1) % v.len()];
        n += DVec3::new((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
    }
    n
}

/// Ear-clip a planar polygon; returns local index triples.
pub fn triangulate(v: &[DVec3]) -> Vec<[usize; 3]> {
    let n = v.len();
    if n < 3 {
        return vec![];
    }
    if n == 3 {
        return vec![[0, 1, 2]];
    }
    let nrm = newell(v);
    let ax = nrm.abs();
    // project to the plane's dominant axes, keeping orientation counter-clockwise
    let (i0, i1) = if ax.z >= ax.x && ax.z >= ax.y { (0, 1) } else if ax.x >= ax.y { (1, 2) } else { (2, 0) };
    let sign = if [nrm.x, nrm.y, nrm.z][3 - i0 - i1] >= 0.0 { 1.0 } else { -1.0 };
    let p: Vec<DVec2> = v.iter().map(|q| DVec2::new(q[i0], q[i1] * sign)).collect();
    let cross = |a: DVec2, b: DVec2, c: DVec2| (b - a).perp_dot(c - a);
    let convex = (0..n).all(|i| cross(p[i], p[(i + 1) % n], p[(i + 2) % n]) >= -1e-12);
    if convex {
        return (1..n - 1).map(|i| [0, i, i + 1]).collect();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(n - 2);
    let mut guard = 0;
    while idx.len() > 3 && guard < 10 * n * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for k in 0..m {
            let (a, b, c) = (idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]);
            if cross(p[a], p[b], p[c]) <= 1e-12 {
                continue;
            }
            let inside = idx.iter().any(|&o| {
                if o == a || o == b || o == c {
                    return false;
                }
                let q = p[o];
                cross(p[a], p[b], q) >= -1e-12 && cross(p[b], p[c], q) >= -1e-12 && cross(p[c], p[a], q) >= -1e-12
                    && (q - p[a]).length_squared() > 1e-18
                    && (q - p[b]).length_squared() > 1e-18
                    && (q - p[c]).length_squared() > 1e-18
            });
            if inside {
                continue;
            }
            out.push([a, b, c]);
            idx.remove(k);
            clipped = true;
            break;
        }
        if !clipped {
            // degenerate: fan the rest
            for k in 1..idx.len() - 1 {
                out.push([idx[0], idx[k], idx[k + 1]]);
            }
            return out;
        }
    }
    if idx.len() == 3 {
        out.push([idx[0], idx[1], idx[2]]);
    }
    out
}

/// Triangulate with normals smoothed across edges below `smooth_deg` (0 = flat shading).
pub fn to_trimesh(acc: &MeshAcc, smooth_deg: f64, uv_scale: f64) -> TriMesh {
    let fnorm: Vec<DVec3> = acc
        .faces
        .iter()
        .enumerate()
        .map(|(fi, f)| {
            let v: Vec<DVec3> = f.iter().map(|&i| acc.verts[i as usize]).collect();
            let n = newell(&v);
            if acc.flip.get(fi).copied().unwrap_or(false) { -n } else { n }
        })
        .collect();
    let cos_lim = smooth_deg.to_radians().cos();
    let mut incident: Vec<Vec<u32>> = Vec::new();
    if smooth_deg > 0.0 {
        incident = vec![Vec::new(); acc.verts.len()];
        for (fi, f) in acc.faces.iter().enumerate() {
            for &i in f {
                incident[i as usize].push(fi as u32);
            }
        }
    }
    let mut out = TriMesh::default();
    let mut groups: std::collections::BTreeMap<i32, Vec<u32>> = Default::default();
    let mut loop_i = 0usize;
    for (fi, f) in acc.faces.iter().enumerate() {
        let nf = fnorm[fi].normalize_or_zero();
        let base = out.pos.len() as u32;
        for (k, &vi) in f.iter().enumerate() {
            let mut n = nf;
            if smooth_deg > 0.0 {
                let mut sum = DVec3::ZERO;
                for &g in &incident[vi as usize] {
                    let ng = fnorm[g as usize];
                    if ng.normalize_or_zero().dot(nf) >= cos_lim {
                        sum += ng;
                    }
                }
                n = sum.normalize_or(nf);
            }
            out.pos.push(acc.verts[vi as usize].as_vec3());
            out.nrm.push(n.as_vec3());
            let uv = acc.uvs.get(loop_i + k).copied().unwrap_or(DVec2::ZERO) * uv_scale;
            out.uv.push(uv.as_vec2());
        }
        loop_i += f.len();
        let v: Vec<DVec3> = f.iter().map(|&i| acc.verts[i as usize]).collect();
        let g = groups.entry(acc.tags[fi]).or_default();
        let flip = acc.flip.get(fi).copied().unwrap_or(false);
        for t in triangulate(&v) {
            if flip {
                g.extend([base + t[0] as u32, base + t[2] as u32, base + t[1] as u32]);
            } else {
                g.extend(t.iter().map(|&i| base + i as u32));
            }
        }
    }
    out.groups = groups.into_iter().collect();
    out
}
