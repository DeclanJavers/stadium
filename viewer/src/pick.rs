//! Click-to-select: cast a ray from the camera through the cursor against the generated parts and
//! the props (seats excluded: a click on the seats lands on the stand under them).

use crate::glb::Scene;
use glam::{Mat4, Vec3};

fn ray_aabb(o: Vec3, d: Vec3, lo: Vec3, hi: Vec3) -> Option<f32> {
    let inv = d.recip();
    let t0 = (lo - o) * inv;
    let t1 = (hi - o) * inv;
    let tmin = t0.min(t1).max_element().max(0.0);
    let tmax = t0.max(t1).min_element();
    (tmax >= tmin).then_some(tmin)
}

fn ray_tri(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let e1 = b - a;
    let e2 = c - a;
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 1e-4).then_some(t)
}

/// Nearest hit along the ray in mesh-local space (positions as stored).
fn hit_mesh(scene: &Scene, mesh: usize, o: Vec3, d: Vec3, best: f32) -> Option<f32> {
    let mut nearest = best;
    let mut found = None;
    for p in &scene.meshes[mesh] {
        if p.positions.is_empty() {
            continue;
        }
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in &p.positions {
            lo = lo.min(Vec3::from(*v));
            hi = hi.max(Vec3::from(*v));
        }
        match ray_aabb(o, d, lo - 0.01, hi + 0.01) {
            Some(t) if t < nearest => {}
            _ => continue,
        }
        for tri in p.indices.chunks_exact(3) {
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| Vec3::from(p.positions[i as usize]));
            if let Some(t) = ray_tri(o, d, a, b, c) {
                if t < nearest {
                    nearest = t;
                    found = Some(t);
                }
            }
        }
    }
    found
}

/// The name of the nearest part or prop under the ray (world/glTF space), with its distance.
pub fn pick(scene: &Scene, origin: Vec3, dir: Vec3) -> Option<(String, f32)> {
    let mut best: Option<(String, f32)> = None;
    let limit = |b: &Option<(String, f32)>| b.as_ref().map(|x| x.1).unwrap_or(f32::MAX);
    for (name, kind, mesh, _) in &scene.nodes {
        if kind == "part" {
            if let Some(t) = hit_mesh(scene, *mesh, origin, dir, limit(&best)) {
                best = Some((name.clone(), t));
            }
        } else if kind == "prop" && !name.starts_with("seat.") {
            for inst in &scene.instances[*mesh] {
                let inv: Mat4 = inst.model.inverse();
                let o = inv.transform_point3(origin);
                let d = inv.transform_vector3(dir);
                let scale = d.length();
                if scale <= 0.0 {
                    continue;
                }
                // distances in local space scale with the transform; compare in world units
                if let Some(t) = hit_mesh(scene, *mesh, o, d / scale, limit(&best) * scale) {
                    let tw = t / scale;
                    if tw < limit(&best) {
                        best = Some((name.clone(), tw));
                    }
                }
            }
        }
    }
    best
}
