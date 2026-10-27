//! The land round and between the plots (port of kit/site/landscape.py): the fields or river
//! outside, the site lawn, driveways and footpaths (with a stub to each plot that needs one), the
//! boundary (fence, railings, wall or hedge, with tree or berm screening), the gate, and trees.

use super::kernel::{self, along, ccw, dist_to_polyline, inside, seg_hit, strip};
use super::{buildings, mats, Out, Plot, COVER_Z, FIELD_Z, LAWN_Z, PATH_Z, ROAD_Z};
use crate::catalog::{num, text, Catalog};
use crate::mesh::MeshAcc;
use crate::prng::uniform;
use crate::stadium::Placement;
use crate::stands::Polyline;
use glam::{DVec2, DVec3};
use serde_json::Value;

pub const ROAD_W: f64 = 7.0;
pub const PATH_W: f64 = 3.0;
pub const GATE_W: f64 = 12.0;

pub fn pts(v: &Value) -> Vec<DVec2> {
    v.as_array().map(|a| a.iter().map(|p| DVec2::new(p[0].as_f64().unwrap_or(0.0), p[1].as_f64().unwrap_or(0.0))).collect()).unwrap_or_default()
}

pub fn lines(v: &Value) -> Vec<Vec<DVec2>> {
    v.as_array().map(|a| a.iter().map(pts).collect()).unwrap_or_default()
}

fn yaw_of(d: DVec2) -> f64 {
    (-d.x).atan2(d.y)
}

/// Offsets of a polyline (mitred like kernel::strip): one point per vertex `off` metres to its left.
pub fn offset_line(path: &[DVec2], off: f64, closed: bool) -> Vec<DVec2> {
    let n = path.len();
    let seg = |i: usize| -> DVec2 {
        let (a, b) = (path[i % n], path[(i + 1) % n]);
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let ln = dx.hypot(dy);
        if ln > 1e-12 {
            DVec2::new(-dy / ln, dx / ln)
        } else {
            DVec2::new(0.0, 1.0)
        }
    };
    (0..n)
        .map(|i| {
            let (prev, next) = if closed {
                (seg((i + n - 1) % n), seg(i))
            } else if i == 0 {
                (seg(0), seg(0))
            } else if i == n - 1 {
                (seg(n - 2), seg(n - 2))
            } else {
                (seg(i - 1), seg(i))
            };
            let (mut mx, mut my) = (prev.x + next.x, prev.y + next.y);
            let ml = mx.hypot(my);
            if ml > 1e-12 {
                mx /= ml;
                my /= ml;
            } else {
                mx = next.x;
                my = next.y;
            }
            let k = (mx * next.x + my * next.y).max(0.3);
            DVec2::new(path[i].x + mx / k * off, path[i].y + my / k * off)
        })
        .collect()
}

/// A ribbon between two offsets of a polyline at two heights (banks, slopes), UVs in metres.
pub fn ribbon(m: &mut MeshAcc, path: &[DVec2], (o0, z0): (f64, f64), (o1, z1): (f64, f64), tag: i32, closed: bool) {
    let a = offset_line(path, o0, closed);
    let b = offset_line(path, o1, closed);
    let n = path.len();
    let segs = if closed { n } else { n - 1 };
    let mut s = 0.0;
    for i in 0..segs {
        let j = (i + 1) % n;
        let ln = (path[j] - path[i]).length();
        let w = (b[i] - a[i]).length().hypot(z1 - z0);
        let base = kernel::add_verts(m, &[DVec3::new(a[i].x, a[i].y, z0), DVec3::new(a[j].x, a[j].y, z0), DVec3::new(b[j].x, b[j].y, z1), DVec3::new(b[i].x, b[i].y, z1)]);
        // a's left is b's side: a(i) a(j) b(j) b(i) faces up when b is left of a
        let face = if o1 > o0 { vec![base, base + 1, base + 2, base + 3] } else { vec![base + 3, base + 2, base + 1, base] };
        let uv = if o1 > o0 {
            vec![DVec2::new(s, 0.0), DVec2::new(s + ln, 0.0), DVec2::new(s + ln, w), DVec2::new(s, w)]
        } else {
            vec![DVec2::new(s, w), DVec2::new(s + ln, w), DVec2::new(s + ln, 0.0), DVec2::new(s, 0.0)]
        };
        m.add_face(face, tag, Some(uv));
        s += ln;
    }
}

/// Fields (hedgerows and trees) or a river meadow round the site, on a big plane of farm grass.
pub fn context(o: &mut Out, tpl: &Value, boundary: &[DVec2], seed: u64) {
    let (lo, hi) = boundary.iter().fold((DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let e = 340.0;
    let mut m = MeshAcc::new();
    kernel::ground(&mut m, &kernel::rect(hi.x - lo.x + 2.0 * e, hi.y - lo.y + 2.0 * e, (lo.x + hi.x) / 2.0, (lo.y + hi.y) / 2.0), FIELD_Z, 0);
    o.part("World:fields", m, vec![mats::lawn("field", DVec2::ZERO, 0.0)], 0.0);
    let river = pts(&tpl["river"]);
    // the fields: boundaries every 110-240 m each way, kept 25 m clear of the site
    let big = 300.0;
    let (ko_lo, ko_hi) = (lo - DVec2::splat(25.0), hi + DVec2::splat(25.0));
    let clear = |p: DVec2| !(ko_lo.x < p.x && p.x < ko_hi.x && ko_lo.y < p.y && p.y < ko_hi.y);
    let cuts = |a: f64, b: f64, salt: u64| {
        let (mut out, mut v, mut i) = (Vec::new(), a + 40.0 * uniform(seed + salt, 0), 0u64);
        while v < b {
            out.push(v);
            i += 1;
            v += 110.0 + 130.0 * uniform(seed + salt, i);
        }
        out
    };
    let xs = cuts(lo.x - big, hi.x + big, 60);
    let ys = cuts(lo.y - big, hi.y + big, 61);
    // each field is pasture (the plane), a hay meadow, stubble or ploughland
    let mut covers = [MeshAcc::new(), MeshAcc::new(), MeshAcc::new()];
    for j in 0..ys.len().saturating_sub(1) {
        for i in 0..xs.len().saturating_sub(1) {
            let c = (j * 64 + i) as u64;
            let (x0, x1, y0, y1) = (xs[i] + 2.0, xs[i + 1] - 2.0, ys[j] + 2.0, ys[j + 1] - 2.0);
            if x1 < ko_lo.x || x0 > ko_hi.x || y1 < ko_lo.y || y0 > ko_hi.y {
                let r = uniform(seed + 62, c);
                let kind = if (0.45..0.65).contains(&r) { Some(0) } else if (0.65..0.82).contains(&r) { Some(1) } else if r >= 0.82 { Some(2) } else { None };
                if let Some(kd) = kind {
                    kernel::ground(&mut covers[kd], &[DVec2::new(x0, y0), DVec2::new(x1, y0), DVec2::new(x1, y1), DVec2::new(x0, y1)], FIELD_Z + 0.006, 0);
                }
                // a copse in some field corners: oak and birch with an understorey
                if uniform(seed + 63, c) < 0.16 {
                    let cx = if uniform(seed + 64, c) < 0.5 { x0 + 22.0 } else { x1 - 22.0 };
                    let cy = if uniform(seed + 65, c) < 0.5 { y0 + 22.0 } else { y1 - 22.0 };
                    for t in 0..9u64 {
                        let a = uniform(seed + 66, c * 16 + t) * 6.283;
                        let rr = 16.0 * uniform(seed + 67, c * 16 + t).sqrt();
                        let q = DVec2::new(cx + rr * a.cos(), cy + rr * a.sin());
                        let u = uniform(seed + 68, c * 16 + t);
                        let key = if u < 0.5 { "prop.tree_oak" } else if u < 0.72 { "prop.tree_birch" } else { "prop.shrub" };
                        if river.is_empty() || dist_to_polyline(q, &river) > 30.0 {
                            tree(o, key, q, FIELD_Z, seed + 69, c * 16 + t);
                        }
                    }
                }
            }
        }
    }
    for (kd, m2) in covers.into_iter().enumerate() {
        if !m2.faces.is_empty() {
            let (name, mat) = match kd {
                0 => ("hay", mats::lawn("hay", DVec2::ZERO, 0.0)),
                1 => ("stubble", mats::lawn("stubble", DVec2::ZERO, 0.0)),
                _ => ("plough", mats::named("Site_Dirt")),
            };
            o.part(&format!("World:fields_{name}"), m2, vec![mat], 0.0);
        }
    }
    // hedgerows along every boundary, a little crooked, with oak standards in runs
    let mut hedges: Vec<Vec<DVec2>> = Vec::new();
    for (n, x) in xs.iter().enumerate() {
        let steps = ((hi.y - lo.y + 2.0 * big) / 100.0) as u64;
        hedges.push((0..=steps).map(|q| DVec2::new(x + 5.0 * (uniform(seed + 70, n as u64 * 16 + q) - 0.5), lo.y - big + q as f64 * 100.0)).collect());
    }
    for (n, y) in ys.iter().enumerate() {
        let steps = ((hi.x - lo.x + 2.0 * big) / 100.0) as u64;
        hedges.push((0..=steps).map(|q| DVec2::new(lo.x - big + q as f64 * 100.0, y + 5.0 * (uniform(seed + 71, n as u64 * 16 + q) - 0.5))).collect());
    }
    let mut k = 0u64;
    for h in &hedges {
        for (p, d) in along(h, 2.2, 0.5) {
            k += 1;
            // gaps (field gates), the site and the road out
            if (uniform(seed + 7, k / 9) < 0.08) || !clear(p) || roads_near(tpl, p, 9.0) || (!river.is_empty() && dist_to_polyline(p, &river) < 26.0) {
                continue;
            }
            let s = 0.85 + 0.35 * uniform(seed + 8, k);
            // a hedge segment runs along its local x (its front faces across the row)
            let mut pl = Placement::at(DVec3::new(p.x, p.y, FIELD_Z), yaw_of(d) - std::f64::consts::FRAC_PI_2 + (uniform(seed + 9, k) - 0.5) * 0.3);
            pl.scale = DVec3::new(1.0, 1.0, s);
            o.prop("prop.hedge", pl, &[]);
            let run = if uniform(seed + 72, k / 24) < 0.35 { 0.11 } else { 0.012 };
            if uniform(seed + 10, k) < run {
                let key = if uniform(seed + 11, k) < 0.8 { "prop.tree_oak" } else { "prop.tree_birch" };
                tree(o, key, p, FIELD_Z, seed + 12, k);
            }
        }
    }
    if !river.is_empty() {
        let mut w = MeshAcc::new();
        strip(&mut w, &river, 30.0, FIELD_Z + 0.02, 0, None);
        o.part("World:river", w, vec![mats::named("Site_Water")], 0.0);
        let mut b = MeshAcc::new();
        ribbon(&mut b, &river, (15.0, FIELD_Z + 0.02), (21.0, FIELD_Z + 0.05), 0, false);
        ribbon(&mut b, &river, (-21.0, FIELD_Z + 0.05), (-15.0, FIELD_Z + 0.02), 0, false);
        o.part("World:riverbanks", b, vec![mats::named("Site_Dirt")], 0.0);
        for (j, (p, d)) in along(&river, 3.0, 0.5).into_iter().enumerate() {
            let j = j as u64;
            for side in [-1.0, 1.0] {
                let n = DVec2::new(-d.y, d.x) * side;
                if uniform(seed + 13, j * 2 + (side > 0.0) as u64) < 0.55 {
                    let q = p + n * (14.5 + 3.0 * uniform(seed + 14, j));
                    o.prop("prop.reeds", Placement::at(DVec3::new(q.x, q.y, FIELD_Z + 0.02), uniform(seed + 15, j) * 6.283), &[]);
                }
                if uniform(seed + 16, j * 2 + (side > 0.0) as u64) < 0.09 {
                    let q = p + n * (25.0 + 8.0 * uniform(seed + 17, j));
                    tree(o, if uniform(seed + 18, j) < 0.5 { "prop.tree_willow" } else { "prop.tree_oak" }, q, FIELD_Z, seed + 19, j);
                }
            }
        }
    }
}

fn roads_near(tpl: &Value, p: DVec2, d: f64) -> bool {
    lines(&tpl["roads"]).iter().any(|r| dist_to_polyline(p, r) < d)
}

/// A tree prop with variation in size and turn, and its own proportions (spread and height scale
/// separately, about 12 % each way), so no two crowns of a species share a silhouette.
pub fn tree(o: &mut Out, key: &str, p: DVec2, z: f64, seed: u64, k: u64) {
    let s = 0.8 + 0.45 * uniform(seed, k);
    let mut pl = Placement::at(DVec3::new(p.x, p.y, z), uniform(seed + 1, k) * std::f64::consts::TAU);
    // its own proportions: spread and height scale separately, so no two crowns share a silhouette
    let f = |i: u64| s * (0.88 + 0.24 * uniform(seed + 501 + i, k));
    pl.scale = DVec3::new(f(0), f(1), f(2));
    o.prop(key, pl, &[]);
}

pub fn site_ground(o: &mut Out, boundary: &[DVec2]) {
    let mut m = MeshAcc::new();
    kernel::ground(&mut m, boundary, LAWN_Z, 0);
    o.part("Site:lawn", m, vec![mats::lawn("amenity", DVec2::ZERO, 0.0)], 0.0);
}

/// Where a plot's access stub starts (site xy), its direction, and its width (0 = no stub).
fn stub_of(p: &Plot) -> (DVec2, DVec2, f64, bool) {
    let st = p.state();
    let fac = text(&p.cfg, "facility", "");
    let occ = text(&p.cfg, "occupant", "");
    let start = p.frame.pt(p.front.x * p.w / 2.0, p.front.y * p.d / 2.0);
    let dir = p.frame.vec(p.front.x, p.front.y);
    let (w, old) = match st {
        "built" | "construction" => match fac {
            "car_park" | "drop_off" => (6.0, false),
            "pitch" | "artificial_pitch" | "youth_pitches" | "showpiece_pitch" => (0.0, false),
            _ => (PATH_W, false),
        },
        "occupied" | "demolition" => match occ {
            "old_car_park" | "container_yard" | "old_warehouse" | "farm_barns" | "derelict_mill" | "abandoned_offices" => (5.0, true),
            _ => (0.0, true),
        },
        _ => (0.0, false),
    };
    (start, dir, w, old)
}

/// Driveways (kerbs, centre line), footpaths and the stubs to the plots.
pub fn roads(o: &mut Out, tpl: &Value, plots: &[Plot]) {
    let roads = lines(&tpl["roads"]);
    let paths = lines(&tpl["paths"]);
    let mut asphalt = MeshAcc::new();
    let mut kerb = MeshAcc::new();
    let mut paint = MeshAcc::new();
    for r in &roads {
        strip(&mut asphalt, r, ROAD_W, ROAD_Z, 0, None);
        for side in [-1.0, 1.0] {
            let e = offset_line(r, side * (ROAD_W / 2.0 + 0.1), false);
            strip(&mut kerb, &e, 0.22, ROAD_Z + 0.09, 0, None);
        }
        for (p, d) in along(r, 9.0, 0.5) {
            let (a, b) = (p - d * 1.8, p + d * 1.8);
            strip(&mut paint, &[a, b], 0.12, ROAD_Z + 0.006, 0, None);
        }
    }
    let mut path = MeshAcc::new();
    for p in &paths {
        strip(&mut path, p, PATH_W, PATH_Z, 0, None);
    }
    // stubs: straight from each plot's frontage to the first road or path in that direction
    let mut old = MeshAcc::new();
    for p in plots {
        let (start, dir, w, worn) = stub_of(p);
        if w <= 0.0 {
            continue;
        }
        let mut best: Option<(f64, f64)> = None;
        for (l, half) in roads.iter().map(|r| (r, ROAD_W / 2.0)).chain(paths.iter().map(|r| (r, PATH_W / 2.0))) {
            for i in 0..l.len() - 1 {
                if let Some(t) = seg_hit(start, dir, l[i], l[i + 1]) {
                    if t < 90.0 && best.is_none_or(|b| t < b.0) {
                        best = Some((t, half));
                    }
                }
            }
        }
        if let Some((t, half)) = best {
            let line = [start - dir * 0.5, start + dir * (t + half * 0.5)];
            if worn {
                strip(&mut old, &line, w, PATH_Z - 0.003, 0, None);
            } else if w > PATH_W {
                strip(&mut asphalt, &line, w, ROAD_Z - 0.004, 0, None);
            } else {
                strip(&mut path, &line, w, PATH_Z, 0, None);
            }
        }
    }
    o.part("Site:roads", asphalt, vec![mats::named("Site_Asphalt")], 0.0);
    o.part("Site:kerbs", kerb, vec![mats::named("Site_Kerb")], 0.0);
    o.part("Site:road_paint", paint, vec![mats::tint("TINT_road_paint", "#ecebe4")], 0.0);
    o.part("Site:paths", path, vec![mats::named("Site_Asphalt")], 0.0);
    o.part("Site:old_tracks", old, vec![mats::named("Site_Asphalt_Old")], 0.0);
}

/// The boundary as an open polyline that starts and ends either side of the gate opening.
pub fn fence_line(boundary: &[DVec2], gate: DVec2) -> Vec<DVec2> {
    let b = ccw(boundary);
    let n = b.len();
    // the edge the gate is on
    let mut best = (f64::MAX, 0usize);
    for i in 0..n {
        let d = dist_to_polyline(gate, &[b[i], b[(i + 1) % n]]);
        if d < best.0 {
            best = (d, i);
        }
    }
    let i = best.1;
    let (a, c) = (b[i], b[(i + 1) % n]);
    let dir = (c - a).normalize();
    let mut out = vec![gate + dir * (GATE_W / 2.0)];
    for k in 1..=n {
        out.push(b[(i + k) % n]);
    }
    out.push(gate - dir * (GATE_W / 2.0));
    out
}

pub fn boundary(o: &mut Out, tpl: &Value, boundary: &[DVec2], cfg: &Value, seed: u64) {
    let gate = DVec2::new(num(&tpl["gate"], "x", 0.0), num(&tpl["gate"], "y", 0.0));
    let line = fence_line(boundary, gate);
    let kind = text(cfg, "type", "chainlink");
    match kind {
        "hedge" => {
            for (k, (p, d)) in along(&line, 2.0, 0.5).into_iter().enumerate() {
                let k = k as u64;
                let s = 0.9 + 0.25 * uniform(seed + 21, k);
                let mut pl = Placement::at(DVec3::new(p.x, p.y, LAWN_Z), yaw_of(d) - std::f64::consts::FRAC_PI_2 + (uniform(seed + 22, k) - 0.5) * 0.2);
                pl.scale = DVec3::new(1.0, 1.0, s);
                o.prop("prop.hedge", pl, &[]);
            }
            // a stock fence inside the hedge, as farms do
            chainlink(o, &offset_line(&line, 0.9, false), 1.2, "Site:boundary");
        }
        "brick_wall" => {
            let mut m = MeshAcc::new();
            for i in 0..line.len() - 1 {
                let (a, b) = (line[i], line[i + 1]);
                let d = b - a;
                let l = d.length();
                if l < 0.1 {
                    continue;
                }
                let c = (a + b) / 2.0;
                let yaw = d.y.atan2(d.x);
                kernel::obox(&mut m, c.x, c.y, LAWN_Z, l, 0.34, 2.05, yaw, 0);
                kernel::obox(&mut m, c.x, c.y, LAWN_Z + 2.05, l + 0.1, 0.44, 0.09, yaw, 1);
            }
            for (p, d) in along(&line, 3.6, 0.0) {
                let yaw = d.y.atan2(d.x);
                kernel::obox(&mut m, p.x, p.y, LAWN_Z, 0.56, 0.56, 2.25, yaw, 0);
                kernel::obox(&mut m, p.x, p.y, LAWN_Z + 2.25, 0.66, 0.66, 0.1, yaw, 1);
            }
            o.part("Site:boundary", m, vec![mats::named("Facade_Brick"), mats::named("Facade_Stone")], 0.0);
        }
        "palisade" => {
            let mut m = MeshAcc::new();
            let mut rails: Vec<Polyline> = Vec::new();
            for z in [0.45, 2.0] {
                rails.push((line.iter().map(|p| DVec3::new(p.x, p.y, LAWN_Z + z)).collect(), 0.035));
            }
            for (p, d) in along(&line, 2.75, 0.0) {
                kernel::obox(&mut m, p.x, p.y, LAWN_Z, 0.1, 0.1, 2.55, d.y.atan2(d.x), 0);
            }
            for (k, (p, d)) in along(&line, 0.16, 0.5).into_iter().enumerate() {
                let q = p + DVec2::new(-d.y, d.x) * 0.06;
                kernel::obox(&mut m, q.x, q.y, LAWN_Z, 0.068, 0.022, 2.4 + 0.04 * ((k % 2) as f64), d.y.atan2(d.x), 0);
            }
            o.part("Site:boundary", m, vec![mats::tint("TINT_fence", "#2b4433")], 0.0);
            o.tubes("Site:boundary_rails", rails, mats::tint("TINT_fence", "#2b4433"), 6);
        }
        "railings" => {
            let mut bars: Vec<Polyline> = Vec::new();
            for z in [0.12, 1.65] {
                bars.push((line.iter().map(|p| DVec3::new(p.x, p.y, LAWN_Z + z)).collect(), 0.02));
            }
            for (p, _) in along(&line, 0.13, 0.5) {
                bars.push((vec![DVec3::new(p.x, p.y, LAWN_Z), DVec3::new(p.x, p.y, LAWN_Z + 1.8)], 0.011));
            }
            let mut m = MeshAcc::new();
            for (p, d) in along(&line, 2.6, 0.0) {
                kernel::obox(&mut m, p.x, p.y, LAWN_Z, 0.08, 0.08, 1.9, d.y.atan2(d.x), 0);
            }
            o.part("Site:boundary", m, vec![mats::tint("TINT_fence", "#17191a")], 0.0);
            o.tubes("Site:boundary_rails", bars, mats::tint("TINT_fence", "#17191a"), 5);
        }
        _ => chainlink(o, &line, 2.4, "Site:boundary"),
    }
    match text(cfg, "screening", "none") {
        "trees" => {
            // a planted belt, not a row: irregular spacing, clumps, mixed species and an understorey
            let inner = offset_line(&line, 5.5, false);
            for (k, (p, d)) in along(&inner, 4.5, 0.5).into_iter().enumerate() {
                let k = k as u64;
                if (p - gate).length() < 14.0 || roads_near(tpl, p, 6.0) {
                    continue;
                }
                let dense = uniform(seed + 30, k / 5) < 0.6;
                if uniform(seed + 31, k) > if dense { 0.62 } else { 0.2 } {
                    continue;
                }
                let r = uniform(seed + 32, k);
                let key = if r < 0.3 { "prop.tree_conifer" } else if r < 0.55 { "prop.tree_birch" } else if r < 0.75 { "prop.tree_oak" } else { "prop.shrub" };
                let off = (uniform(seed + 33, k) - 0.5) * 5.0;
                tree(o, key, p + DVec2::new(-d.y, d.x) * off, LAWN_Z, seed + 34, k);
            }
        }
        "berm" => {
            let mut m = MeshAcc::new();
            ribbon(&mut m, &line, (0.6, LAWN_Z), (4.5, 2.4), 0, false);
            ribbon(&mut m, &line, (4.5, 2.4), (8.6, LAWN_Z), 0, false);
            o.part("Site:berm", m, vec![mats::lawn("berm", DVec2::ZERO, 0.0)], 0.0);
            for (k, (p, _)) in along(&offset_line(&line, 4.5, false), 11.0, 0.5).into_iter().enumerate() {
                let k = k as u64;
                if (p - gate).length() < 16.0 || roads_near(tpl, p, 8.0) || uniform(seed + 35, k) < 0.4 {
                    continue;
                }
                let key = if uniform(seed + 36, k) < 0.5 { "prop.shrub" } else { "prop.tree_birch" };
                tree(o, key, p, 2.3, seed + 37, k);
            }
        }
        _ => {}
    }
}

/// Chain-link fence along a polyline: posts every 3 m, top rail, mesh panels.
pub fn chainlink(o: &mut Out, line: &[DVec2], h: f64, name: &str) {
    let mut panels = MeshAcc::new();
    for i in 0..line.len() - 1 {
        let (a, b) = (line[i], line[i + 1]);
        if (b - a).length() < 0.05 {
            continue;
        }
        kernel::quad(&mut panels, DVec3::new(a.x, a.y, LAWN_Z + 0.03), DVec3::new(b.x, b.y, LAWN_Z + 0.03), DVec3::new(b.x, b.y, LAWN_Z + h), DVec3::new(a.x, a.y, LAWN_Z + h), 0);
    }
    let mut posts: Vec<Polyline> = vec![(line.iter().map(|p| DVec3::new(p.x, p.y, LAWN_Z + h - 0.03)).collect(), 0.022)];
    for (p, _) in along(line, 3.0, 0.0) {
        posts.push((vec![DVec3::new(p.x, p.y, LAWN_Z), DVec3::new(p.x, p.y, LAWN_Z + h + 0.05)], 0.032));
    }
    o.part(name, panels, vec![mats::chainlink()], 0.0);
    o.tubes(&format!("{name}_posts"), posts, mats::tint("TINT_fence", "#2b4433"), 6);
}

/// Farm gate, barrier and hut, or a gatehouse in the site's style.
pub fn gate(o: &mut Out, cat: &Catalog, tpl: &Value, cfg: &Value, _seed: u64) {
    let g = DVec2::new(num(&tpl["gate"], "x", 0.0), num(&tpl["gate"], "y", 0.0));
    let roads = lines(&tpl["roads"]);
    // the road's direction through the gate (into the site)
    let r = roads.first().cloned().unwrap_or_default();
    let mut dir = DVec2::new(0.0, 1.0);
    for i in 0..r.len().saturating_sub(1) {
        if dist_to_polyline(g, &[r[i], r[i + 1]]) < 1.0 {
            dir = (r[i + 1] - r[i]).normalize();
        }
    }
    let side = DVec2::new(dir.y, -dir.x); // to the right going in
    // the gate's frame: local +y into the site along the road, +x to the right
    let f = kernel::Frame::new(g.x, g.y, (-dir.x).atan2(dir.y).to_degrees());
    let tier = num(cfg, "tier", 1.0) as usize;
    let club = o.club.clone();
    // gate posts either side of the opening
    let mut m = MeshAcc::new();
    for s in [-1.0, 1.0] {
        let p = g + side * (s * (GATE_W / 2.0 + 0.2));
        kernel::obox(&mut m, p.x, p.y, LAWN_Z, 0.35, 0.35, if tier >= 3 { 2.6 } else { 1.6 }, 0.0, 0);
    }
    o.part("Gate:posts", m, vec![mats::named(if tier >= 3 { "Facade_Brick" } else { "Galvanised" })], 0.0);
    match tier {
        1 => {
            // two farm gate leaves, swung open into the site
            for s in [-1.0, 1.0] {
                // each leaf hangs on its post and stands open into the site (along +y)
                o.prop_at(&f, "prop.gate_farm", s * GATE_W / 2.0, 0.0, LAWN_Z, 0.0, &[]);
            }
            o.prop_at(&f, "prop.sign_post", GATE_W / 2.0 + 2.5, -3.0, LAWN_Z, 0.0, &[("TINT_club", &club.primary)]);
        }
        2 => {
            // one barrier on the way in (the left-hand lane): cabinet on the verge, 4 m boom to the centre line
            o.prop_at(&f, "prop.barrier_arm", -ROAD_W / 2.0 - 0.45, 9.0, LAWN_Z, 0.0, &[]);
            o.prop_at(&f, "prop.security_hut", -ROAD_W / 2.0 - 3.2, 10.0, COVER_Z, std::f64::consts::FRAC_PI_2, &[("TINT_cabin", "#d9dbd6")]);
            o.prop_at(&f, "prop.sign_post", GATE_W / 2.0 + 2.5, -3.0, LAWN_Z, 0.0, &[("TINT_club", &club.primary)]);
            o.prop_at(&f, "prop.lamp_post", ROAD_W / 2.0 + 1.0, 4.0, LAWN_Z, std::f64::consts::FRAC_PI_2, &[]);
            o.prop_at(&f, "prop.cctv_pole", -ROAD_W / 2.0 - 1.5, 3.0, LAWN_Z, std::f64::consts::PI, &[]);
        }
        _ => {
            let style = text(cfg, "style", "classic").to_string();
            buildings::gatehouse(o, cat, &f, &style);
            // a barrier per lane, cabinets on the verges, booms meeting at the centre line
            o.prop_at(&f, "prop.barrier_arm", -ROAD_W / 2.0 - 0.45, 10.0, LAWN_Z, 0.0, &[]);
            o.prop_at(&f, "prop.barrier_arm", ROAD_W / 2.0 + 0.45, 14.0, LAWN_Z, std::f64::consts::PI, &[]);
            for s in [-1.0, 1.0] {
                o.prop_at(&f, "prop.planter", s * (GATE_W / 2.0 + 2.2), 1.5, LAWN_Z, 0.0, &[]);
                o.prop_at(&f, "prop.lamp_post", s * (ROAD_W / 2.0 + 0.9), 20.0, LAWN_Z, s * std::f64::consts::FRAC_PI_2, &[]);
            }
            o.prop_at(&f, "prop.cctv_pole", ROAD_W / 2.0 + 1.6, 2.0, LAWN_Z, 0.0, &[]);
        }
    }
    o.socket_at(&f, "gate", "main", 0.0, 0.0, 0.0, DVec2::new(0.0, 1.0), vec![("tier", tier as f64)]);
}

/// Trees on the lawn between plots, roads and paths.
pub fn trees(o: &mut Out, tpl: &Value, boundary: &[DVec2], plots: &[Plot], density: f64, seed: u64) {
    if density <= 0.0 {
        return;
    }
    let roads = lines(&tpl["roads"]);
    let paths = lines(&tpl["paths"]);
    let (lo, hi) = boundary.iter().fold((DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let rects: Vec<Vec<DVec2>> = plots.iter().map(|p| kernel::rect(p.w + 6.0, p.d + 6.0, 0.0, 0.0).iter().map(|q| p.frame.pt(q.x, q.y)).collect()).collect();
    let b = ccw(boundary);
    let step = 15.0;
    let (nx, ny) = (((hi.x - lo.x) / step) as i64, ((hi.y - lo.y) / step) as i64);
    let mut k = 0u64;
    for j in 0..=ny {
        for i in 0..=nx {
            k += 1;
            let p = DVec2::new(lo.x + (i as f64 + uniform(seed + 41, k)) * step, lo.y + (j as f64 + uniform(seed + 42, k)) * step);
            if uniform(seed + 43, k) > density * 0.6 || !inside(&b, p.x, p.y) || dist_to_polyline(p, &[b.clone(), vec![b[0]]].concat()) < 5.0 {
                continue;
            }
            if rects.iter().any(|r| inside(r, p.x, p.y)) || roads.iter().any(|r| dist_to_polyline(p, r) < 8.0) || paths.iter().any(|r| dist_to_polyline(p, r) < 4.5) {
                continue;
            }
            let r = uniform(seed + 44, k);
            let key = if r < 0.4 { "prop.tree_oak" } else if r < 0.7 { "prop.tree_birch" } else if r < 0.85 { "prop.tree_conifer" } else { "prop.shrub" };
            tree(o, key, p, LAWN_Z, seed + 45, k);
        }
    }
}
