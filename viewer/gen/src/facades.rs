//! Facades (port of kit/stands/facades.py): the outside of a stand run, its entrances, stair cores,
//! gable walls and sign sockets.

use crate::catalog::{num, text};
use crate::mesh::{MeshAcc, Section};
use crate::py::{arange, clip, linspace, round_half_even, round_to};
use crate::runs::Run;
use crate::sightlines::Tier;
use crate::stands::{convex_hull, section_hull, stand_depth, Polyline, Socket};
use glam::{DVec2, DVec3};
use serde_json::Value;

pub const TAG_MAIN: i32 = 0;
pub const TAG_ACCENT: i32 = 1;
pub const TAG_DARK: i32 = 2;
pub const TAG_TRIM: i32 = 3;
pub const TAG_ETFE: i32 = 4;
pub const TAG_CORE: i32 = 5;
pub const LEVEL: f64 = 4.5;

#[derive(Clone, Copy, Debug)]
pub struct RoofInfo {
    pub tail_s: f64,
    pub tail_top: f64,
}

pub fn line(tiers: &[Tier], roof: Option<RoofInfo>, offset: f64) -> (f64, f64) {
    let mut s = stand_depth(tiers);
    let mut top = tiers.iter().map(|t| t.h_last()).fold(f64::MIN, f64::max) + 1.6;
    if let Some(r) = roof {
        s = s.max(r.tail_s);
        top = top.max(r.tail_top);
    }
    (s + offset, top)
}

fn rect(s0: f64, s1: f64, z0: f64, z1: f64) -> Vec<DVec2> {
    vec![DVec2::new(s0, z0), DVec2::new(s1, z0), DVec2::new(s1, z1), DVec2::new(s0, z1)]
}

#[allow(clippy::too_many_arguments)]
fn prism(acc: &mut MeshAcc, run: &Run, ta: f64, tb: f64, s0: f64, s1: f64, z0: f64, z1: f64, tag: i32) {
    acc.add_prism(run, ta, tb, &Section::uniform(rect(s0, s1, z0, z1), tag), tag);
}

fn repeat(run: &Run, s: f64, spacing: f64, t0: f64, t1: f64, offset: f64) -> Vec<f64> {
    let m = run.metric(s);
    let n = 1usize.max(((t1 - t0) * m / spacing) as i64 as usize);
    let step = (t1 - t0) / n as f64;
    (0..n).map(|k| t0 + (k as f64 + offset) * step).collect()
}

#[derive(Clone, Debug, Default)]
pub struct FacadeOut {
    pub mesh: MeshAcc,
    pub tubes: Vec<Polyline>,
    pub sockets: Vec<Socket>,
    pub line_s: f64,
    pub height: f64,
}

pub fn build(run: &Run, tiers: &[Tier], bay_list: &[(f64, f64)], spec: &Value, roof: Option<RoofInfo>, ends: (bool, bool)) -> FacadeOut {
    let kind = text(spec, "type", "concrete_frame");
    let offset = num(spec, "offset", 0.6);
    let cores_every = num(spec, "cores_every", 4.0) as usize;
    let entrance_every = (num(spec, "entrance_every", 2.0) as usize).max(1);
    let mut acc = MeshAcc::new();
    let mut tubes = Vec::new();
    let mut sockets = Vec::new();
    let (sw, h) = line(tiers, roof, offset);
    let (t0, t1) = (run.t0, run.t1);
    let levels = arange(LEVEL, h - 1.0, LEVEL);
    let hw = |width: f64, s: f64| 0.5 * width / run.metric(s);
    let backing = if matches!(kind, "concrete_frame" | "fins" | "glass" | "etfe") { TAG_DARK } else { TAG_MAIN };
    prism(&mut acc, run, t0, t1, sw, sw + 0.3, 0.0, h, backing);
    prism(&mut acc, run, t0, t1, sw - 0.1, sw + 0.5, h - 0.6, h + 0.3, TAG_TRIM);
    let out_s = sw + 0.3;
    match kind {
        "concrete_frame" => {
            let mut cols: Vec<f64> = bay_list.iter().map(|b| b.0).collect();
            cols.push(bay_list[bay_list.len() - 1].1);
            for ta in cols {
                prism(&mut acc, run, ta - hw(0.7, out_s), ta + hw(0.7, out_s), sw, sw + 1.1, 0.0, h, TAG_MAIN);
            }
            for &z in &levels {
                prism(&mut acc, run, t0, t1, sw, sw + 0.85, z - 0.25, z + 0.25, TAG_MAIN);
            }
        }
        "brick" => {
            for &z in &levels {
                prism(&mut acc, run, t0, t1, sw, sw + 0.38, z - 0.15, z + 0.12, TAG_TRIM);
            }
            for &(ta, tb) in bay_list {
                let tm = 0.5 * (ta + tb);
                for &z in &levels {
                    if z + 3.0 > h - 0.8 {
                        continue;
                    }
                    for dx in [-1.6, 1.6] {
                        let tc = tm + dx / run.metric(out_s);
                        prism(&mut acc, run, tc - hw(1.3, out_s), tc + hw(1.3, out_s), sw + 0.3, sw + 0.33, z + 0.5, z + 2.9, TAG_DARK);
                        prism(&mut acc, run, tc - hw(1.6, out_s), tc + hw(1.6, out_s), sw + 0.3, sw + 0.4, z + 2.9, z + 3.2, TAG_TRIM);
                        prism(&mut acc, run, tc - hw(1.5, out_s), tc + hw(1.5, out_s), sw + 0.3, sw + 0.42, z + 0.38, z + 0.5, TAG_TRIM);
                    }
                }
            }
            prism(&mut acc, run, t0, t1, sw, sw + 0.42, 0.0, 0.6, TAG_TRIM);
        }
        "metal_bands" => {
            let (mut z, mut k) = (0.6, 0);
            while z < h - 0.7 {
                let hh = 1.2f64.min(h - 0.7 - z);
                prism(&mut acc, run, t0, t1, sw + 0.3, sw + 0.42, z + 0.03, z + hh - 0.03, if k % 2 == 0 { TAG_MAIN } else { TAG_ACCENT });
                z += hh;
                k += 1;
            }
        }
        "glass" => {
            prism(&mut acc, run, t0, t1, sw + 0.3, sw + 0.34, 0.0, h - 0.6, TAG_DARK);
            let mut ts = repeat(run, out_s, 1.6, t0, t1, 0.0);
            ts.push(t1);
            for t in ts {
                prism(&mut acc, run, t - hw(0.08, out_s), t + hw(0.08, out_s), sw + 0.3, sw + 0.5, 0.0, h - 0.6, TAG_TRIM);
            }
            for z in arange(3.6, h - 0.6, 3.6) {
                prism(&mut acc, run, t0, t1, sw + 0.3, sw + 0.48, z - 0.06, z + 0.06, TAG_TRIM);
            }
        }
        "fins" => {
            for t in repeat(run, out_s, num(spec, "fin_spacing", 1.1), t0, t1, 0.5) {
                prism(&mut acc, run, t - hw(0.07, out_s), t + hw(0.07, out_s), sw + 0.35, sw + 1.35, 0.5, h - 0.3, TAG_MAIN);
            }
        }
        "etfe" => etfe(&mut acc, &mut tubes, run, sw + 1.2, h, spec),
        _ => {}
    }
    for (b, &(ta, tb)) in bay_list.iter().enumerate() {
        let tm = 0.5 * (ta + tb);
        if b % entrance_every == 0 {
            prism(&mut acc, run, tm - hw(4.2, out_s), tm + hw(4.2, out_s), sw + 0.3, sw + 0.36, 0.0, 3.2, TAG_DARK);
            prism(&mut acc, run, tm - hw(4.8, out_s), tm + hw(4.8, out_s), sw + 0.3, sw + 1.4, 3.2, 3.55, TAG_TRIM);
        }
        if cores_every > 0 && b % cores_every == cores_every / 2 && h > 9.0 {
            core(&mut acc, run, tm, sw, h);
        }
        let (p, n) = run.frame(tm);
        let q = p + n * (sw + 1.5);
        sockets.push(Socket {
            kind: "facade_sign",
            id: format!("{}_B{}", run.name, b),
            loc: DVec3::new(q.x, q.y, round_to(0.62 * h, 3)),
            n: -n,
            props: vec![("width", round_to((tb - ta) * run.metric(sw), 3)), ("max_height", round_to(0.3 * h, 3))],
        });
    }
    for (at_start, flag) in [(true, ends.0), (false, ends.1)] {
        if flag {
            gable(&mut acc, run, tiers, sw, h, at_start, kind);
        }
    }
    FacadeOut { mesh: acc, tubes, sockets, line_s: sw, height: h }
}

fn core(acc: &mut MeshAcc, run: &Run, tm: f64, sw: f64, h: f64) {
    let s1 = sw + 6.0;
    let hw = |width: f64, s: f64| 0.5 * width / run.metric(s);
    prism(acc, run, tm - hw(7.0, s1), tm + hw(7.0, s1), sw, s1, 0.0, h + 1.5, TAG_CORE);
    for dx in [-2.0, 0.0, 2.0] {
        let tc = tm + dx / run.metric(s1);
        prism(acc, run, tc - hw(0.35, s1), tc + hw(0.35, s1), s1, s1 + 0.03, 1.0, h - 0.5, TAG_DARK);
    }
}

fn gable(acc: &mut MeshAcc, run: &Run, tiers: &[Tier], sw: f64, h: f64, at_start: bool, kind: &str) {
    let mut hull = section_hull(tiers);
    hull.extend([DVec2::new(sw + 0.3, 0.0), DVec2::new(sw + 0.3, h)]);
    let poly = convex_hull(hull);
    let tag = if matches!(kind, "brick" | "metal_bands" | "fins") { TAG_MAIN } else { TAG_CORE };
    let d = 0.3 / run.metric(0.0);
    let (ta, tb) = if at_start { (run.t0 - d, run.t0) } else { (run.t1, run.t1 + d) };
    acc.add_prism(run, ta, tb, &Section::uniform(poly, tag), tag);
}

fn etfe(acc: &mut MeshAcc, tubes: &mut Vec<Polyline>, run: &Run, s: f64, h: f64, spec: &Value) {
    let (mut wd, hd) = match spec.get("cushion").and_then(|v| v.as_array()) {
        Some(a) if a.len() == 2 => (a[0].as_f64().unwrap_or(4.0), a[1].as_f64().unwrap_or(4.0)),
        _ => (4.0, 4.0),
    };
    let bulge = num(spec, "bulge", 0.35);
    let m = run.metric(s);
    let length = (run.t1 - run.t0) * m;
    let n_u = 1i64.max(round_half_even(length / wd) as i64);
    wd = length / n_u as f64;
    let n_z = 1i64.max(((h - 1.0) / hd) as i64);
    let z0 = 0.5;
    let k = 5usize;
    let grid = linspace(0.0, 1.0, k + 1);
    let pi = std::f64::consts::PI;
    let zmax = z0 + n_z as f64 * hd;
    for j in -1..2 * n_z {
        let cz = z0 + hd / 2.0 + j as f64 * hd / 2.0;
        for i in -1..n_u + 1 {
            let cu = if j.rem_euclid(2) == 0 { (i as f64 + 0.5) * wd } else { (i as f64 + 1.0) * wd };
            let v0 = DVec2::new(cu, cz - hd / 2.0);
            let v1 = DVec2::new(cu + wd / 2.0, cz);
            let v3 = DVec2::new(cu - wd / 2.0, cz);
            let mut uz = vec![DVec2::ZERO; (k + 1) * (k + 1)];
            let (mut umin, mut umax, mut zmin_, mut zmax_) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
            for a in 0..=k {
                for b in 0..=k {
                    let p = v0 + grid[a] * (v1 - v0) + grid[b] * (v3 - v0);
                    umin = umin.min(p.x);
                    umax = umax.max(p.x);
                    zmin_ = zmin_.min(p.y);
                    zmax_ = zmax_.max(p.y);
                    uz[a * (k + 1) + b] = p;
                }
            }
            if umax <= 0.0 || umin >= length || zmax_ <= z0 || zmin_ >= zmax {
                continue;
            }
            let base = acc.verts.len() as u32;
            for a in 0..=k {
                for b in 0..=k {
                    let p = uz[a * (k + 1) + b];
                    let (u, z) = (clip(p.x, 0.0, length), clip(p.y, z0, zmax));
                    let out = (pi * grid[a]).sin() * (pi * grid[b]).sin() * bulge;
                    acc.verts.push(run.world(run.t0 + u / m, s + out, z));
                }
            }
            let kk = (k + 1) as u32;
            for a in 0..k as u32 {
                for b in 0..k as u32 {
                    acc.add_face(vec![base + a * kk + b, base + (a + 1) * kk + b, base + (a + 1) * kk + b + 1, base + a * kk + b + 1], TAG_ETFE, None);
                }
            }
        }
    }
    for i in -n_z..n_u + n_z + 1 {
        for sign in [1.0, -1.0] {
            let mut pts = Vec::new();
            for zz in linspace(z0, zmax, 12) {
                let u = (i as f64 + 0.5) * wd + sign * (zz - z0) * (wd / hd);
                if (0.0..=length).contains(&u) {
                    pts.push(run.world(run.t0 + u / m, s, zz));
                }
            }
            if pts.len() >= 2 {
                tubes.push((pts, 0.045));
            }
        }
    }
}
