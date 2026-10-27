//! What stands on a plot before the club builds on it (port of kit/site/occupants.py): an old car
//! park, a warehouse, a mill, offices, a container yard, farm barns, scrub, old tennis courts, a
//! bowling green, a rubble lot. Each fills its plot, varies with the plot's seed, and has a
//! half-demolished form for the demolition state.

use super::buildings::{self, Spec};
use super::kernel::{self, bx, cylinder, obox, quad, strip};
use super::{landscape, mats, Out, Plot, COVER_Z};
use crate::mesh::MeshAcc;
use crate::prng::uniform;
use crate::stands::Polyline;
use glam::{DVec2, DVec3};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

pub const CAR_COLORS: [&str; 10] = ["#1d1f22", "#e8e8e4", "#8c9096", "#2b3f66", "#7a1d22", "#3a4a3c", "#c9c4b8", "#4a4f57", "#b5b9bd", "#20304a"];
const DULL: [&str; 6] = ["#5b5148", "#6d6a62", "#4f5650", "#7a6e5a", "#5c4a3f", "#6b7072"];
const CONTAINERS: [&str; 6] = ["#7a3b2a", "#2f4f7a", "#3c5a3a", "#707479", "#b0592b", "#8a2a24"];

/// The plot's ground cover (inset from its edge), at COVER_Z.
pub fn cover(o: &mut Out, p: &Plot, name: &str, inset: f64, mat: crate::stadium::Mat) {
    let mut m = MeshAcc::new();
    kernel::ground(&mut m, &kernel::rect(p.w - 2.0 * inset, p.d - 2.0 * inset, 0.0, 0.0), COVER_Z, 0);
    o.part_at(&p.frame, &format!("{}:{name}", p.id), m, vec![mat], 0.0);
}

/// A rough heap (rubble, earth, sand): a cone of rings with a lumpy edge.
#[allow(clippy::too_many_arguments)]
pub fn mound(m: &mut MeshAcc, cx: f64, cy: f64, rx: f64, ry: f64, h: f64, seed: u64, tag: i32) {
    // a slumped cone with a spread toe and a rounded crest, lumpy in broad lobes
    let n = 24usize;
    let rings = [(1.0, 0.0), (0.86, 0.17), (0.7, 0.42), (0.52, 0.66), (0.34, 0.85), (0.16, 0.97)];
    let ph: Vec<f64> = (0..4u64).map(|q| uniform(seed + 2, q) * TAU).collect();
    let mut idx: Vec<Vec<u32>> = Vec::new();
    for (ri, (rr, zz)) in rings.iter().enumerate() {
        let mut ring = Vec::new();
        for k in 0..n {
            let a = TAU * k as f64 / n as f64;
            let j = 1.0 + 0.12 * (2.0 * a + ph[0]).sin() + 0.07 * (3.0 * a + ph[1]).sin() + 0.05 * (uniform(seed, (ri * n + k) as u64) - 0.5);
            let lobe = 1.0 + 0.1 * (2.0 * a + ph[2]).sin() + 0.06 * (5.0 * a + ph[3]).sin();
            let z = if ri == 0 { COVER_Z } else { COVER_Z + h * zz * lobe * (0.95 + 0.1 * uniform(seed + 1, (ri * n + k) as u64)) };
            ring.push(m.verts.len() as u32);
            m.verts.push(DVec3::new(cx + rx * rr * j * a.cos(), cy + ry * rr * j * a.sin(), z));
        }
        idx.push(ring);
    }
    let top = m.verts.len() as u32;
    m.verts.push(DVec3::new(cx, cy, COVER_Z + h));
    for r in 0..rings.len() - 1 {
        for k in 0..n {
            let (a, b) = (idx[r][k], idx[r][(k + 1) % n]);
            let (c, d) = (idx[r + 1][(k + 1) % n], idx[r + 1][k]);
            let pts: Vec<DVec3> = [a, b, c, d].iter().map(|&i| m.verts[i as usize]).collect();
            m.add_face(vec![a, b, c, d], tag, Some(pts.iter().map(|q| DVec2::new(q.x, q.y)).collect()));
        }
    }
    let last = &idx[rings.len() - 1];
    for k in 0..n {
        let (a, b) = (last[k], last[(k + 1) % n]);
        let pts: Vec<DVec3> = [a, b, top].iter().map(|&i| m.verts[i as usize]).collect();
        m.add_face(vec![a, b, top], tag, Some(pts.iter().map(|q| DVec2::new(q.x, q.y)).collect()));
    }
}

/// A scatter point inside the plot (local), keeping `margin` from the edge.
fn spot(p: &Plot, salt: u64, i: u64, margin: f64) -> DVec2 {
    DVec2::new((uniform(p.seed + salt, i) - 0.5) * (p.w - 2.0 * margin), (uniform(p.seed + salt + 1, i) - 0.5) * (p.d - 2.0 * margin))
}

fn weeds(o: &mut Out, p: &Plot, per_m2: f64, salt: u64) {
    let n = (p.w * p.d * per_m2) as u64;
    for i in 0..n {
        let q = spot(p, salt, i, 1.5);
        let s = 0.7 + 0.8 * uniform(p.seed + salt + 2, i);
        o.prop_full(&p.frame, "prop.weeds", q.x, q.y, COVER_Z, uniform(p.seed + salt + 3, i) * TAU, 0.0, DVec3::splat(s), &[]);
    }
}

/// Bay lines in rows across a plot (used by car parks old and new): returns the bay centres and
/// their facing (local), the paint mesh, and how many bays.
pub fn bays(p_w: f64, p_d: f64, inset: f64, paint: &mut MeshAcc, z: f64, gaps: &dyn Fn(u64) -> bool) -> Vec<(DVec2, f64)> {
    let (bw, bd, aisle) = (2.5, 5.0, 6.0);
    let mut out = Vec::new();
    let usable = p_d - 2.0 * inset;
    // pattern from the near edge: bays | aisle | bays bays | aisle | bays ...
    let mut rows: Vec<(f64, bool)> = Vec::new(); // (row near edge y, facing +y?)
    let mut y = -usable / 2.0;
    let mut facing_up = true;
    loop {
        if y + bd > usable / 2.0 + 1e-9 {
            break;
        }
        rows.push((y, facing_up));
        y += bd;
        if facing_up {
            y += aisle;
        }
        facing_up = !facing_up;
    }
    let n = ((p_w - 2.0 * inset) / bw).floor() as usize;
    let x0 = -(n as f64) * bw / 2.0;
    let mut k = 0u64;
    for (y0, up) in rows {
        // the line at the bay ends (back of the bays) and between bays
        for i in 0..=n {
            let x = x0 + i as f64 * bw;
            k += 1;
            if !gaps(k) {
                strip(paint, &[DVec2::new(x, y0), DVec2::new(x, y0 + bd)], 0.1, z, 0, None);
            }
        }
        for i in 0..n {
            out.push((DVec2::new(x0 + (i as f64 + 0.5) * bw, y0 + bd / 2.0), if up { 0.0 } else { PI }));
        }
    }
    out
}

/// Parked cars in some of the bays.
pub fn park(o: &mut Out, p: &Plot, bays: &[(DVec2, f64)], fill: f64, dull: bool, salt: u64) -> usize {
    park_dir(o, p, bays, fill, dull, salt, true)
}

/// Parked cars (reverse: about half have backed in; off for a one-way lay-by, where every car faces
/// the way the traffic goes).
pub fn park_dir(o: &mut Out, p: &Plot, bays: &[(DVec2, f64)], fill: f64, dull: bool, salt: u64, reverse: bool) -> usize {
    let mut n = 0;
    for (i, (c, yaw)) in bays.iter().enumerate() {
        let i = i as u64;
        if uniform(p.seed + salt, i) >= fill {
            continue;
        }
        n += 1;
        let r = uniform(p.seed + salt + 1, i);
        let model = if r < 0.45 { "prop.car_hatch" } else if r < 0.8 { "prop.car_saloon" } else { "prop.car_suv" };
        // one of five registrations per model, so no two cars in a row share a plate
        let v = (uniform(p.seed + salt + 7, i) * 5.0) as u64;
        let key = if v > 0 { format!("{model}_{}", v + 1) } else { model.to_string() };
        let key = key.as_str();
        let col = if dull { DULL[(uniform(p.seed + salt + 2, i) * DULL.len() as f64) as usize % DULL.len()] } else { CAR_COLORS[(uniform(p.seed + salt + 2, i) * CAR_COLORS.len() as f64) as usize % CAR_COLORS.len()] };
        // cars reverse in about half the time; a little slop in the bay
        let flip = if reverse && uniform(p.seed + salt + 3, i) < 0.45 { PI } else { 0.0 };
        let jitter = DVec2::new((uniform(p.seed + salt + 4, i) - 0.5) * 0.3, (uniform(p.seed + salt + 5, i) - 0.5) * 0.4);
        o.prop_at(&p.frame, key, c.x + jitter.x, c.y + jitter.y, COVER_Z, yaw + flip + (uniform(p.seed + salt + 6, i) - 0.5) * 0.06, &[("TINT_car_paint", col)]);
    }
    n
}

pub fn build(o: &mut Out, p: &Plot, occ: &str, demolished: bool) {
    match occ {
        "old_car_park" => old_car_park(o, p, demolished),
        "old_warehouse" => warehouse(o, p, demolished),
        "derelict_mill" => mill(o, p, demolished),
        "abandoned_offices" => offices(o, p, demolished),
        "container_yard" => container_yard(o, p, demolished),
        "farm_barns" => farm(o, p, demolished),
        "disused_courts" => courts(o, p, demolished),
        "bowling_green" => bowling_green(o, p, demolished),
        "rubble_lot" => rubble(o, p, demolished),
        _ => scrub(o, p, demolished),
    }
}

fn old_car_park(o: &mut Out, p: &Plot, demolished: bool) {
    cover(o, p, "ground", 1.0, mats::named("Site_Asphalt_Old"));
    let mut paint = MeshAcc::new();
    let seed = p.seed;
    let b = bays(p.w, p.d, 2.0, &mut paint, COVER_Z + 0.004, &|k| uniform(seed + 51, k) < 0.35);
    o.part_at(&p.frame, &format!("{}:paint", p.id), paint, vec![mats::tint("TINT_road_paint", "#a9a69c")], 0.0);
    if !demolished {
        park(o, p, &b, 0.1, true, 60);
        if uniform(p.seed, 70) < 0.6 {
            let q = spot(p, 71, 0, 6.0);
            o.prop_at(&p.frame, "prop.scrap_car", q.x, q.y, COVER_Z, uniform(p.seed + 72, 0) * TAU, &[]);
        }
    } else {
        let mut m = MeshAcc::new();
        for i in 0..6 {
            let q = spot(p, 73, i, 8.0);
            mound(&mut m, q.x, q.y, 3.0 + 2.0 * uniform(p.seed + 74, i), 2.5 + 2.0 * uniform(p.seed + 75, i), 1.0 + uniform(p.seed + 76, i), p.seed + 77 + i, 0);
        }
        o.part_at(&p.frame, &format!("{}:rubble", p.id), m, vec![mats::named("Site_Rubble")], 75.0);
    }
    // lamp posts, one leaning
    let n = (p.w / 30.0).floor().max(1.0) as usize;
    for i in 0..n {
        let u = -p.w / 2.0 + (i as f64 + 0.5) * p.w / n as f64;
        let tilt = if i == 0 { 0.12 } else { 0.0 };
        o.prop_full(&p.frame, "prop.lamp_post", u, -p.d / 2.0 + 1.2, COVER_Z, 0.0, tilt, DVec3::ONE, &[]);
    }
    weeds(o, p, 1.0 / 60.0, 80);
}

/// A building's footprint inside a plot, front to the frontage, set back behind a yard.
fn place(p: &Plot, bw: f64, bd: f64) -> (super::kernel::Frame, f64, f64) {
    let pl = super::facilities::place(p, bw, bd, 12.0);
    (pl.f, pl.w, pl.d)
}

fn derelict(w: f64, d: f64, storeys: usize, sh: f64, wall: &str, wall_color: &str, roof: &str, roof_mat: &str, window: &str, seed: u64) -> Spec {
    Spec {
        w,
        d,
        storeys,
        storey_h: sh,
        roof: roof.into(),
        pitch_deg: 32.0,
        eaves: 0.3,
        parapet: 0.7,
        window: window.into(),
        door: "none".into(),
        roller_doors: 0,
        canopy: "none".into(),
        quoins: false,
        clock: false,
        fins: false,
        led_band: false,
        arcade: false,
        solar: false,
        frame: false,
        decay: 0.8,
        sign: None,
        lit: 0.0,
        seed,
        mats: buildings::materials(wall, wall_color, "Facade_Concrete", roof_mat, None, "#4a4c4b", "#5f6360", false),
    }
}

fn warehouse(o: &mut Out, p: &Plot, demolished: bool) {
    cover(o, p, "yard", 1.0, mats::named("Apron_Concrete"));
    let (f, w, d) = place(p, 64.0, 42.0);
    if demolished {
        buildings::ruin(o, &f, &format!("{}:ruin", p.id), w, d, 4.5, mats::named("Bldg_Rusted"), p.seed);
        let mut cols: Vec<Polyline> = Vec::new();
        for i in 0..((w / 7.0) as usize) {
            let x = -w / 2.0 + 3.5 + i as f64 * 7.0;
            if uniform(p.seed, 300 + i as u64) < 0.5 {
                for sy in [-1.0, 1.0] {
                    cols.push((vec![DVec3::new(x, sy * (d / 2.0 - 0.3), 0.0), DVec3::new(x, sy * (d / 2.0 - 0.3), 8.0)], 0.12));
                }
                cols.push((vec![DVec3::new(x, -d / 2.0 + 0.3, 8.0), DVec3::new(x, 0.0, 9.5), DVec3::new(x, d / 2.0 - 0.3, 8.0)], 0.1));
            }
        }
        o.tubes_at(&f, &format!("{}:frames", p.id), cols, mats::named("Site_Rust"), 6);
    } else {
        let mut sp = derelict(w, d, 1, 8.4, "Bldg_Rusted", "#7c5a46", "barrel", "Roof_Cement", "strip", p.seed);
        sp.roller_doors = ((w / 18.0) as usize).max(1);
        sp.mats[buildings::T_DOOR as usize] = mats::tint("TINT_door", "#6b6f6a");
        sp.mats[buildings::T_BASE as usize] = mats::named("Facade_Brick");
        buildings::build(o, &f, &format!("{}:warehouse", p.id), &sp);
    }
    // yard clutter at the front
    let front = -p.front * 0.0 + p.front * ((p.d.min(p.w)) / 2.0 - 6.0);
    for i in 0..4u64 {
        let side = DVec2::new(p.front.y, -p.front.x) * ((i as f64 - 1.5) * 7.0 + 3.0 * (uniform(p.seed + 90, i) - 0.5));
        let q = front + side;
        let key = ["prop.pallet_stack", "prop.oil_drum", "prop.skip", "prop.tyre_stack"][i as usize];
        let tint: &[(&str, &str)] = if key == "prop.skip" { &[("TINT_skip", "#a5562a")] } else if key == "prop.oil_drum" { &[("TINT_drum", "#2d4a6a")] } else { &[] };
        o.prop_at(&p.frame, key, q.x, q.y, COVER_Z, uniform(p.seed + 91, i) * TAU, tint);
    }
    weeds(o, p, 1.0 / 70.0, 92);
}

fn mill(o: &mut Out, p: &Plot, demolished: bool) {
    cover(o, p, "ground", 1.0, mats::named("Site_Gravel"));
    let (f, w, d) = place(p, 62.0, 18.0);
    let chimney_x = w / 2.0 + 4.0;
    let mut ch = MeshAcc::new();
    let ch_h = if demolished { 9.0 } else { 32.0 };
    cylinder(&mut ch, chimney_x.min(p.w), d / 2.0 - 2.0, 0.0, ch_h, 1.6, 16, 0, true);
    cylinder(&mut ch, chimney_x.min(p.w), d / 2.0 - 2.0, ch_h - 1.0, ch_h - 0.3, 1.85, 16, 0, true);
    o.part_at(&f, &format!("{}:chimney", p.id), ch, vec![mats::tint("Bldg_Brick_Old", "#6a3a2c")], 0.0);
    if demolished {
        buildings::ruin(o, &f, &format!("{}:ruin", p.id), w, d, 7.0, mats::tint("Bldg_Brick_Old", "#6a3a2c"), p.seed);
    } else {
        let mut sp = derelict(w, d, 4, 3.6, "Bldg_Brick_Old", "#6a3a2c", "gable", "Roof_Slate", "punched", p.seed);
        sp.door = "timber".into();
        sp.mats[buildings::T_DOOR as usize] = mats::named("Site_Plywood");
        sp.mats[buildings::T_TRIM as usize] = mats::tint("TINT_trim", "#3b3a36");
        buildings::build(o, &f, &format!("{}:mill", p.id), &sp);
    }
    for i in 0..((p.w * p.d / 400.0) as u64) {
        let q = spot(p, 95, i, 3.0);
        o.prop_at(&p.frame, "prop.bramble", q.x, q.y, COVER_Z, uniform(p.seed + 96, i) * TAU, &[]);
    }
    weeds(o, p, 1.0 / 50.0, 97);
}

fn offices(o: &mut Out, p: &Plot, demolished: bool) {
    cover(o, p, "ground", 1.0, mats::named("Site_Asphalt_Old"));
    let (f, w, d) = place(p, 46.0, 15.0);
    if demolished {
        buildings::ruin(o, &f, &format!("{}:ruin", p.id), w, d, 5.0, mats::tint("Facade_Concrete", "#9a958b"), p.seed);
    } else {
        let mut sp = derelict(w, d, 3, 3.4, "Facade_Concrete", "#9a958b", "flat", "Roof_Flat", "ribbon", p.seed);
        sp.door = "glass".into();
        sp.canopy = "steel".into();
        buildings::build(o, &f, &format!("{}:offices", p.id), &sp);
    }
    weeds(o, p, 1.0 / 45.0, 98);
}

fn container_yard(o: &mut Out, p: &Plot, demolished: bool) {
    cover(o, p, "ground", 1.0, mats::named("Site_Gravel"));
    // a fence round the yard with an opening at the front
    let (hw, hd) = (p.w / 2.0 - 1.2, p.d / 2.0 - 1.2);
    let corners = [DVec2::new(-hw, -hd), DVec2::new(hw, -hd), DVec2::new(hw, hd), DVec2::new(-hw, hd)];
    let mut ring: Vec<DVec2> = Vec::new();
    // start at the frontage's middle so the opening is there
    let mid = DVec2::new(p.front.x * hw, p.front.y * hd);
    let tangent = DVec2::new(-p.front.y, p.front.x);
    ring.push(mid + tangent * 4.0);
    let start = corners.iter().position(|c| (*c - mid).dot(tangent) > 0.0 && (*c - mid).dot(p.front) > -0.1).unwrap_or(0);
    for k in 0..4 {
        ring.push(corners[(start + k) % 4]);
    }
    ring.push(mid - tangent * 4.0);
    if !demolished {
        let world: Vec<DVec2> = ring.iter().map(|q| p.frame.pt(q.x, q.y)).collect();
        landscape::chainlink(o, &world, 2.4, &format!("{}:fence", p.id));
    }
    // container rows along the plot's long side, at the back
    let long_x = p.w >= p.d;
    let (la, lb) = if long_x { (p.w, p.d) } else { (p.d, p.w) };
    let rows = ((lb - 14.0) / 6.0).floor().max(1.0) as usize;
    let mut k = 0u64;
    for r in 0..rows {
        let across = lb / 2.0 - 5.0 - r as f64 * 6.0;
        let mut a = -la / 2.0 + 4.0;
        while a < la / 2.0 - 8.0 {
            k += 1;
            let big = uniform(p.seed + 110, k) < 0.4;
            let len = if big { 12.19 } else { 6.06 };
            if a + len > la / 2.0 - 3.0 {
                break;
            }
            if uniform(p.seed + 111, k) < if demolished { 0.6 } else { 0.18 } {
                a += len + 1.0;
                continue;
            }
            let c = a + len / 2.0;
            let (u, v) = if long_x { (c, across) } else { (across, c) };
            let yaw = if long_x { FRAC_PI_2 } else { 0.0 };
            let col = CONTAINERS[(uniform(p.seed + 112, k) * CONTAINERS.len() as f64) as usize % CONTAINERS.len()];
            let key = if big { "prop.container_40" } else { "prop.storage_container" };
            o.prop_at(&p.frame, key, u, v, COVER_Z, yaw, &[("TINT_container", col)]);
            if !demolished && uniform(p.seed + 113, k) < 0.35 {
                let col2 = CONTAINERS[(uniform(p.seed + 114, k) * CONTAINERS.len() as f64) as usize % CONTAINERS.len()];
                o.prop_at(&p.frame, key, u, v, COVER_Z + 2.59, yaw + (uniform(p.seed + 115, k) - 0.5) * 0.05, &[("TINT_container", col2)]);
            }
            a += len + 0.6 + 2.0 * uniform(p.seed + 116, k);
        }
    }
    // scrap at the front
    let mut m = MeshAcc::new();
    let front = p.front * ((if long_x { p.d } else { p.w }) / 2.0 - 9.0);
    let tangent = DVec2::new(-p.front.y, p.front.x);
    for i in 0..3u64 {
        let q = front + tangent * ((i as f64 - 1.0) * 12.0);
        mound(&mut m, q.x, q.y, 3.5, 2.8, 1.6 + uniform(p.seed + 117, i), p.seed + 118 + i, 0);
    }
    o.part_at(&p.frame, &format!("{}:scrap", p.id), m, vec![mats::named("Site_Scrap")], 0.0);
    for i in 0..3u64 {
        let q = front + tangent * ((i as f64 - 1.0) * 12.0 + 6.0);
        let key = if i == 1 { "prop.tyre_stack" } else { "prop.scrap_car" };
        o.prop_at(&p.frame, key, q.x, q.y, COVER_Z, uniform(p.seed + 119, i) * TAU, &[]);
    }
    weeds(o, p, 1.0 / 90.0, 120);
}

fn farm(o: &mut Out, p: &Plot, demolished: bool) {
    cover(o, p, "yard", 1.0, mats::named("Site_Dirt"));
    let (f, w, d) = place(p, 30.0, 14.0);
    let fa = f.sub(-w * 0.3, 0.0, 0.0);
    if demolished {
        buildings::ruin(o, &fa, &format!("{}:ruin", p.id), w * 0.6, d, 3.0, mats::tint("TINT_timber", "#3b3029"), p.seed);
    } else {
        let mut sp = derelict(w * 0.6, d, 1, 5.5, "TINT_timber", "#3b3029", "gable", "Roof_Cement", "none", p.seed);
        sp.decay = 0.5;
        sp.roller_doors = 1;
        sp.mats[buildings::T_DOOR as usize] = mats::tint("TINT_timber", "#4a3a30");
        buildings::build(o, &fa, &format!("{}:barn", p.id), &sp);
    }
    // a Dutch barn: open sides, curved roof on steel columns
    let fb = f.sub(w * 0.33, 0.0, 0.0);
    let (bw, bd, bh) = (w * 0.32, d, 6.0);
    let mut cols: Vec<Polyline> = Vec::new();
    let nx = (bw / 4.5).round().max(1.0) as usize;
    for i in 0..=nx {
        let x = -bw / 2.0 + bw * i as f64 / nx as f64;
        for sy in [-1.0, 1.0] {
            cols.push((vec![DVec3::new(x, sy * bd / 2.0, 0.0), DVec3::new(x, sy * bd / 2.0, bh)], 0.1));
        }
    }
    o.tubes_at(&fb, &format!("{}:dutch_frame", p.id), cols, mats::named("Site_Rust"), 6);
    if !demolished {
        let mut m = MeshAcc::new();
        let n = 10;
        let rise = 2.6;
        let z_at = |y: f64| bh + rise * (1.0 - (y / (bd / 2.0)).powi(2));
        for i in 0..n {
            let (ya, yb) = (-bd / 2.0 - 0.4 + (bd + 0.8) * i as f64 / n as f64, -bd / 2.0 - 0.4 + (bd + 0.8) * (i + 1) as f64 / n as f64);
            quad(&mut m, DVec3::new(-bw / 2.0 - 0.4, ya, z_at(ya)), DVec3::new(bw / 2.0 + 0.4, ya, z_at(ya)), DVec3::new(bw / 2.0 + 0.4, yb, z_at(yb)), DVec3::new(-bw / 2.0 - 0.4, yb, z_at(yb)), 0);
        }
        o.part_at(&fb, &format!("{}:dutch_roof", p.id), m, vec![mats::named("Bldg_Rusted")], 0.0);
        // a feed silo and an old tractor
        let mut s = MeshAcc::new();
        cylinder(&mut s, 0.0, 0.0, 0.0, 9.0, 1.6, 16, 0, true);
        let sf = f.sub(w / 2.0 + 3.0, d / 2.0 - 2.0, 0.0);
        o.part_at(&sf, &format!("{}:silo", p.id), s, vec![mats::named("Galvanised")], 0.0);
        let q = p.front * 8.0;
        o.prop_at(&p.frame, "prop.tractor", q.x, q.y, COVER_Z, uniform(p.seed + 130, 0) * TAU, &[("TINT_tractor", "#6d2a22")]);
    }
    for i in 0..((p.w * p.d / 300.0) as u64) {
        let q = spot(p, 131, i, 2.0);
        o.prop_at(&p.frame, if i % 3 == 0 { "prop.bramble" } else { "prop.grass_tuft" }, q.x, q.y, COVER_Z, uniform(p.seed + 132, i) * TAU, &[]);
    }
}

fn scrub(o: &mut Out, p: &Plot, demolished: bool) {
    cover(o, p, "rough", 0.5, mats::lawn("rough", p.frame.pt(0.0, 0.0), 0.0));
    let area = p.w * p.d;
    let k_bramble = if demolished { 600.0 } else { 160.0 };
    for i in 0..((area / k_bramble) as u64) {
        let q = spot(p, 140, i, 1.5);
        let s = 0.7 + 0.7 * uniform(p.seed + 141, i);
        o.prop_full(&p.frame, "prop.bramble", q.x, q.y, COVER_Z, uniform(p.seed + 142, i) * TAU, 0.0, DVec3::splat(s), &[]);
    }
    for i in 0..((area / 45.0) as u64) {
        let q = spot(p, 143, i, 1.0);
        o.prop_full(&p.frame, "prop.grass_tuft", q.x, q.y, COVER_Z, uniform(p.seed + 144, i) * TAU, 0.0, DVec3::splat(0.8 + 0.6 * uniform(p.seed + 145, i)), &[]);
    }
    if !demolished {
        for i in 0..((area / 1400.0) as u64 + 1) {
            let q = spot(p, 146, i, 4.0);
            let key = if uniform(p.seed + 147, i) < 0.25 { "prop.tree_dead" } else if uniform(p.seed + 148, i) < 0.6 { "prop.tree_birch" } else { "prop.shrub" };
            landscape::tree(o, key, p.frame.pt(q.x, q.y), COVER_Z, p.seed + 149, i);
        }
        let q = spot(p, 150, 0, 5.0);
        o.prop_at(&p.frame, "prop.tyre_stack", q.x, q.y, COVER_Z, 0.4, &[]);
    } else {
        let mut m = MeshAcc::new();
        for i in 0..4 {
            let q = spot(p, 151, i, 6.0);
            mound(&mut m, q.x, q.y, 2.5, 2.0, 1.2, p.seed + 152 + i, 0);
        }
        o.part_at(&p.frame, &format!("{}:cuttings", p.id), m, vec![mats::named("Site_Dirt")], 0.0);
    }
}

fn courts(o: &mut Out, p: &Plot, demolished: bool) {
    cover(o, p, "rough", 0.5, mats::lawn("rough", p.frame.pt(0.0, 0.0), 0.0));
    // courts side by side: 18.3 x 36.6 m each including run-off, long axis along the plot's depth
    let (cw, cd) = (18.3, 36.6);
    let along_y = p.d >= cd + 2.0;
    let (span, depth) = if along_y { (p.w, p.d) } else { (p.d, p.w) };
    let n = (((span - 4.0) / cw).floor() as usize).clamp(1, 3);
    let total = n as f64 * cw;
    let mut surf = MeshAcc::new();
    let mut lines = MeshAcc::new();
    let mut nets = MeshAcc::new();
    let mut posts: Vec<Polyline> = Vec::new();
    let to = |a: f64, b: f64| if along_y { DVec2::new(a, b) } else { DVec2::new(b, a) };
    let poly = |pts: Vec<DVec2>| -> Vec<DVec2> { pts };
    let q0 = [to(-total / 2.0, -cd / 2.0), to(total / 2.0, -cd / 2.0), to(total / 2.0, cd / 2.0), to(-total / 2.0, cd / 2.0)];
    kernel::ground(&mut surf, &poly(q0.to_vec()), COVER_Z + 0.01, 0);
    for c in 0..n {
        let a0 = -total / 2.0 + c as f64 * cw + cw / 2.0;
        let (hl, hs, hd) = (23.77 / 2.0, 10.97 / 2.0, 8.23 / 2.0);
        let l = |pts: [DVec2; 2], m: &mut MeshAcc| strip(m, &pts, 0.05, COVER_Z + 0.016, 0, None);
        for sx in [-1.0, 1.0] {
            l([to(a0 + sx * hs, -hl), to(a0 + sx * hs, hl)], &mut lines);
            l([to(a0 + sx * hd, -hl), to(a0 + sx * hd, hl)], &mut lines);
            l([to(a0 - hs, sx * hl), to(a0 + hs, sx * hl)], &mut lines);
            l([to(a0 - hd, sx * 6.4), to(a0 + hd, sx * 6.4)], &mut lines);
        }
        l([to(a0, -6.4), to(a0, 6.4)], &mut lines);
        // net posts and a sagging net
        let (pa, pb) = (to(a0 - hs - 0.9, 0.0), to(a0 + hs + 0.9, 0.0));
        for q in [pa, pb] {
            posts.push((vec![DVec3::new(q.x, q.y, COVER_Z), DVec3::new(q.x, q.y, COVER_Z + 1.07)], 0.04));
        }
        if !demolished {
            let sag = 0.25 + 0.3 * uniform(p.seed + 160, c as u64);
            let mid = (pa + pb) / 2.0;
            for (a, b) in [(pa, mid), (mid, pb)] {
                let (za, zb) = if a == pa { (1.0, 1.0 - sag) } else { (1.0 - sag, 1.0) };
                quad(&mut nets, DVec3::new(a.x, a.y, COVER_Z + 0.05), DVec3::new(b.x, b.y, COVER_Z + 0.05), DVec3::new(b.x, b.y, COVER_Z + zb), DVec3::new(a.x, a.y, COVER_Z + za), 0);
            }
        }
    }
    let _ = depth;
    o.part_at(&p.frame, &format!("{}:courts", p.id), surf, vec![mats::tint("TINT_court", "#5e7356")], 0.0);
    o.part_at(&p.frame, &format!("{}:court_lines", p.id), lines, vec![mats::tint("TINT_road_paint", "#cfccc2")], 0.0);
    o.part_at(&p.frame, &format!("{}:court_nets", p.id), nets, vec![mats::net("square", 45, 3.0, "#26282a")], 0.0);
    o.tubes_at(&p.frame, &format!("{}:court_posts", p.id), posts, mats::named("Site_Rust"), 6);
    if !demolished {
        // the old fence, panels missing
        let fl: Vec<DVec2> = vec![q0[0] - DVec2::splat(0.0), q0[1], q0[2], q0[3], q0[0]];
        let mut segs: Vec<Vec<DVec2>> = Vec::new();
        for i in 0..4 {
            let (a, b) = (fl[i], fl[i + 1]);
            let len = (b - a).length();
            let n = (len / 3.0).round().max(1.0) as usize;
            for j in 0..n {
                if uniform(p.seed + 161, (i * 100 + j) as u64) < 0.15 {
                    continue;
                }
                segs.push(vec![a + (b - a) * (j as f64 / n as f64), a + (b - a) * ((j + 1) as f64 / n as f64)]);
            }
        }
        for (j, s) in segs.iter().enumerate() {
            let world: Vec<DVec2> = s.iter().map(|q| p.frame.pt(q.x, q.y)).collect();
            landscape::chainlink(o, &world, 2.75, &format!("{}:fence{j}", p.id));
        }
    }
    weeds(o, p, 1.0 / 40.0, 162);
}

fn bowling_green(o: &mut Out, p: &Plot, demolished: bool) {
    cover(o, p, "rough", 0.5, mats::lawn("rough", p.frame.pt(0.0, 0.0), 0.0));
    let g = (p.w.min(p.d) - 10.0).min(34.0);
    let c = DVec2::new(0.0, -2.0);
    let mut m = MeshAcc::new();
    kernel::ground(&mut m, &kernel::rect(g, g, c.x, c.y), COVER_Z + 0.03, 0);
    o.part_at(&p.frame, &format!("{}:green", p.id), m, vec![mats::lawn("fine", p.frame.pt(c.x, c.y), p.frame.yaw)], 0.0);
    // ditch and timber bank
    let mut d = MeshAcc::new();
    let mut b = MeshAcc::new();
    let r = g / 2.0;
    for (a0, a1) in [(DVec2::new(-r, -r), DVec2::new(r, -r)), (DVec2::new(r, -r), DVec2::new(r, r)), (DVec2::new(r, r), DVec2::new(-r, r)), (DVec2::new(-r, r), DVec2::new(-r, -r))] {
        let n = (a1 - a0).normalize();
        let out = DVec2::new(n.y, -n.x);
        strip(&mut d, &[c + a0 + out * 0.15, c + a1 + out * 0.15], 0.3, COVER_Z + 0.01, 0, None);
        let (p0, p1) = (c + a0 + out * 0.4, c + a1 + out * 0.4);
        let mid = (p0 + p1) / 2.0;
        obox(&mut b, mid.x, mid.y, COVER_Z, (p1 - p0).length() + 0.8, 0.12, 0.3, n.y.atan2(n.x), 0);
    }
    o.part_at(&p.frame, &format!("{}:ditch", p.id), d, vec![mats::named("Site_Gravel")], 0.0);
    o.part_at(&p.frame, &format!("{}:bank", p.id), b, vec![mats::named("Site_Wood")], 0.0);
    // the pavilion at the back
    let pf = p.frame.sub(0.0, p.d / 2.0 - 5.0, PI);
    if demolished {
        buildings::ruin(o, &pf, &format!("{}:ruin", p.id), 10.0, 5.5, 2.4, mats::tint("TINT_timber", "#5b4b3c"), p.seed);
    } else {
        let mut sp = derelict(10.0, 5.5, 1, 3.0, "TINT_timber", "#5b4b3c", "gable", "Roof_Slate", "large", p.seed);
        sp.decay = 0.5;
        sp.door = "timber".into();
        sp.canopy = "timber".into();
        sp.mats[buildings::T_DOOR as usize] = mats::tint("TINT_door", "#3f5a3f");
        buildings::build(o, &pf, &format!("{}:pavilion", p.id), &sp);
        for i in 0..4u64 {
            let x = (i as f64 - 1.5) * 6.0;
            o.prop_at(&p.frame, "prop.bench", x, c.y - r - 1.6, COVER_Z, 0.0, &[]);
        }
    }
    weeds(o, p, 1.0 / 120.0, 170);
}

fn rubble(o: &mut Out, p: &Plot, demolished: bool) {
    cover(o, p, "ground", 0.5, mats::named("Site_Rubble"));
    let mut m = MeshAcc::new();
    let n = ((p.w * p.d / 350.0) as u64).clamp(3, 18) / if demolished { 2 } else { 1 };
    for i in 0..n {
        let q = spot(p, 180, i, 5.0);
        mound(&mut m, q.x, q.y, 2.0 + 3.0 * uniform(p.seed + 181, i), 2.0 + 2.5 * uniform(p.seed + 182, i), 0.8 + 1.4 * uniform(p.seed + 183, i), p.seed + 184 + i, 0);
    }
    o.part_at(&p.frame, &format!("{}:heaps", p.id), m, vec![mats::named("Site_Rubble")], 75.0);
    let mut slabs = MeshAcc::new();
    let mut rebar: Vec<Polyline> = Vec::new();
    for i in 0..(n * 2) {
        let q = spot(p, 185, i, 4.0);
        obox(&mut slabs, q.x, q.y, COVER_Z, 1.5 + 2.0 * uniform(p.seed + 186, i), 1.0 + 1.5 * uniform(p.seed + 187, i), 0.25, uniform(p.seed + 188, i) * PI, 0);
        if i % 3 == 0 {
            rebar.push((vec![DVec3::new(q.x, q.y, COVER_Z + 0.2), DVec3::new(q.x + 0.4, q.y + 0.2, COVER_Z + 1.3)], 0.012));
        }
    }
    o.part_at(&p.frame, &format!("{}:slabs", p.id), slabs, vec![mats::named("Facade_Concrete")], 0.0);
    o.tubes_at(&p.frame, &format!("{}:rebar", p.id), rebar, mats::named("Site_Rust"), 5);
    let _ = bx;
    weeds(o, p, 1.0 / 35.0, 189);
}
