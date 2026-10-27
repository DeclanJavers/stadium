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
use std::f64::consts::{FRAC_PI_2, PI};

pub const TAG_MAIN: i32 = 0;
pub const TAG_ACCENT: i32 = 1;
pub const TAG_DARK: i32 = 2;
pub const TAG_TRIM: i32 = 3;
pub const TAG_ETFE: i32 = 4;
pub const TAG_CORE: i32 = 5;
pub const LEVEL: f64 = 4.5;
/// Entrance openings: the cladding is cut this wide (m) up to DOOR_TOP (above the door canopy).
pub const DOOR_CLEAR: f64 = 9.6;
pub const DOOR_TOP: f64 = 3.55;
/// ETFE cushions start above a glazed ground floor this tall.
pub const ETFE_BASE: f64 = 4.0;
/// Metal bands and fins start above a glazed ground floor this tall.
pub const GROUND: f64 = 4.5;

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

/// [t0, t1] cut at the door intervals: (ta, tb, inside a door).
fn split(t0: f64, t1: f64, doors: &[(f64, f64)]) -> Vec<(f64, f64, bool)> {
    let mut out = Vec::new();
    let mut t = t0;
    for &(a, b) in doors {
        let (a, b) = (a.max(t0), b.min(t1));
        if b <= a || a < t {
            continue;
        }
        if a > t {
            out.push((t, a, false));
        }
        out.push((a, b, true));
        t = b;
    }
    if t < t1 {
        out.push((t, t1, false));
    }
    out
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
    /// plaza furniture: (library prop key, position, yaw)
    pub props: Vec<(&'static str, DVec3, f64)>,
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
    // the inner face (seen from the back rows, under the roof) is the concourse's concrete wall
    acc.add_prism(run, t0, t1, &Section::new(rect(sw, sw + 0.3, 0.0, h), vec![backing, backing, backing, TAG_CORE]), backing);
    prism(&mut acc, run, t0, t1, sw - 0.1, sw + 0.5, h - 0.6, h + 0.3, TAG_TRIM);
    let out_s = sw + 0.3;
    // entrances: the cladding stops above the door canopy there
    let doors: Vec<(f64, f64)> = bay_list
        .iter()
        .enumerate()
        .filter(|(b, &(ta, tb))| is_entrance(run, *b, ta, tb, entrance_every, sw))
        .map(|(_, &(ta, tb))| (0.5 * (ta + tb) - hw(DOOR_CLEAR, out_s), 0.5 * (ta + tb) + hw(DOOR_CLEAR, out_s)))
        .collect();
    let at_door = |t: f64| doors.iter().any(|&(a, b)| t > a && t < b);
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
            shopfront(&mut acc, run, sw, t0, t1, &doors);
            let (mut z, mut k) = (GROUND, 0);
            while z < h - 0.7 {
                let hh = 1.2f64.min(h - 0.7 - z);
                let tag = if k % 2 == 0 { TAG_MAIN } else { TAG_ACCENT };
                for (ta, tb, door) in split(t0, t1, &doors) {
                    let lo = if door { (z + 0.03).max(DOOR_TOP) } else { z + 0.03 };
                    if lo < z + hh - 0.03 {
                        prism(&mut acc, run, ta, tb, sw + 0.3, sw + 0.42, lo, z + hh - 0.03, tag);
                    }
                }
                z += hh;
                k += 1;
            }
        }
        "glass" => {
            for (ta, tb, door) in split(t0, t1, &doors) {
                prism(&mut acc, run, ta, tb, sw + 0.3, sw + 0.34, if door { DOOR_TOP } else { 0.0 }, h - 0.6, TAG_DARK);
            }
            let mut ts = repeat(run, out_s, 1.6, t0, t1, 0.0);
            ts.push(t1);
            for t in ts {
                let z0 = if at_door(t) { DOOR_TOP } else { 0.0 };
                prism(&mut acc, run, t - hw(0.08, out_s), t + hw(0.08, out_s), sw + 0.3, sw + 0.5, z0, h - 0.6, TAG_TRIM);
            }
            for z in arange(3.6, h - 0.6, 3.6) {
                prism(&mut acc, run, t0, t1, sw + 0.3, sw + 0.48, z - 0.06, z + 0.06, TAG_TRIM);
            }
        }
        "fins" => {
            shopfront(&mut acc, run, sw, t0, t1, &doors);
            let zs = skin_base(h, GROUND);
            if zs > GROUND {
                concourse(&mut acc, run, sw, t0, t1, GROUND, zs);
                prism(&mut acc, run, t0, t1, sw + 0.3, sw + 1.45, zs - 0.4, zs, TAG_TRIM);
            }
            for t in repeat(run, out_s, num(spec, "fin_spacing", 1.1), t0, t1, 0.5) {
                prism(&mut acc, run, t - hw(0.07, out_s), t + hw(0.07, out_s), sw + 0.35, sw + 1.35, zs, h - 0.3, TAG_MAIN);
            }
        }
        "etfe" => {
            // cushions above a glazed ground floor (mullions, and a fascia closing the cushion line)
            let zs = skin_base(h, ETFE_BASE);
            etfe(&mut acc, &mut tubes, run, sw + 1.2, h, spec, zs);
            if zs > ETFE_BASE {
                concourse(&mut acc, run, sw, t0, t1, ETFE_BASE, zs - 0.4);
                prism(&mut acc, run, t0, t1, sw + 0.3, sw + 1.3, zs - 0.4, zs, TAG_TRIM);
            }
            for t in repeat(run, out_s, 1.6, t0, t1, 0.5) {
                if !at_door(t) {
                    prism(&mut acc, run, t - hw(0.08, out_s), t + hw(0.08, out_s), sw + 0.3, sw + 0.45, 0.0, ETFE_BASE - 0.4, TAG_TRIM);
                }
            }
            prism(&mut acc, run, t0, t1, sw + 0.3, sw + 1.3, ETFE_BASE - 0.4, ETFE_BASE, TAG_TRIM);
        }
        _ => {}
    }
    if h < SMALL_FACADE {
        frontage(&mut acc, run, sw, h);
    }
    for (b, &(ta, tb)) in bay_list.iter().enumerate() {
        let tm = 0.5 * (ta + tb);
        let is_core = cores_every > 0 && b % cores_every == cores_every / 2 && h > 9.0;
        if is_entrance(run, b, ta, tb, entrance_every, sw) {
            prism(&mut acc, run, tm - hw(4.2, out_s), tm + hw(4.2, out_s), sw + 0.3, sw + 0.36, 0.0, 3.2, TAG_DARK);
            prism(&mut acc, run, tm - hw(4.8, out_s), tm + hw(4.8, out_s), sw + 0.3, sw + 1.4, 3.2, 3.55, TAG_TRIM);
            // door jambs and two mullions, so the opening reads as doors and not as more glazing
            for dx in [-4.3, -1.4, 1.4, 4.3] {
                let tc = tm + dx / run.metric(out_s);
                let wd = if dx.abs() > 4.0 { 0.25 } else { 0.1 };
                prism(&mut acc, run, tc - hw(wd, out_s), tc + hw(wd, out_s), sw + 0.3, sw + 0.55, 0.0, 3.2, TAG_TRIM);
            }
            if !is_core {
                turnstiles(&mut acc, run, tm, sw);
            }
        }
        if is_core {
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
    let props = plaza_props(run, bay_list, sw, entrance_every);
    FacadeOut { mesh: acc, tubes, sockets, line_s: sw, height: h, props }
}

/// lamp posts, bins and trees stand this far out from the facade line
pub const PLAZA_LAMPS: f64 = 10.0;
pub const LAMP_EVERY: f64 = 20.0;

/// Street furniture on the plaza in front of a facade (kit/stands/facades.py plaza_props): lamp
/// posts with bins, trees or benches between planters, bollards, a sign post and cycle stands (a
/// bicycle at one) at each entrance; pigeons, gulls, litter and leaves, a food van or a coffee cart;
/// the centrepiece by stand (the legend's statue in front of the main stand, the crest sculpture,
/// a clock pillar, a bronze ball). (key, position, yaw turning +Y towards the facade).
pub fn plaza_props(run: &Run, bay_list: &[(f64, f64)], sw: f64, entrance_every: usize) -> Vec<(&'static str, DVec3, f64)> {
    let mut out = Vec::new();
    let s_l = sw + PLAZA_LAMPS;
    let m = run.metric(s_l);
    let n = 1usize.max(((run.t1 - run.t0) * m / LAMP_EVERY) as i64 as usize);
    let step = (run.t1 - run.t0) / n as f64;
    let mut put = |key: &'static str, t: f64, s: f64, turn: f64| {
        let (p, nrm) = run.frame(t);
        let q = p + nrm * s;
        out.push((key, DVec3::new(q.x, q.y, 0.0), nrm.x.atan2(-nrm.y) + turn));
    };
    for k in 0..n {
        let t = run.t0 + (k as f64 + 0.5) * step;
        let kf = k as f64;
        put("prop.lamp_post", t, s_l, 0.0);
        put("prop.bin_litter", t + 1.2 / m, s_l, 0.0);
        if k % 3 == 1 {
            put("genprop.matchday_litter", t + 2.2 / m, s_l - 0.4, 0.7 * kf);
        }
        if k % 4 == 3 {
            put("genprop.seagull_standing", t - 2.5 / m, s_l + 2.5, 1.9 * kf);
        }
        if k % 2 == 0 && k + 1 < n {
            put("prop.tree_birch", t + 0.5 * step, s_l + 0.5, 0.0);
            if k % 4 == 0 {
                put("genprop.leaf_litter", t + 0.5 * step + 1.2 / m, s_l + 1.8, 0.9 * kf);
            }
        } else if k + 1 < n {
            // a bench between two planters in the bays without a tree
            let tm = t + 0.5 * step;
            put("prop.bench", tm, s_l + 0.5, 0.0);
            for sg in [-1.0, 1.0] {
                put("prop.planter", tm + sg * 2.4 / m, s_l + 0.5, 0.0);
            }
            if k % 4 == 1 {
                put("genprop.pigeon_pecking", tm + 1.4 / m, s_l - 1.5, 2.3 * kf);
                put("genprop.pigeon_standing", tm - 1.0 / m, s_l - 2.0, 1.1 * kf + 0.5);
            }
        }
        if n >= 4 && k % 4 == 2 {
            put(if (k / 4) % 2 == 0 { "genprop.food_van" } else { "genprop.coffee_cart" }, t, s_l + 15.0, 0.0);
        }
    }
    for (b, &(ta, tb)) in bay_list.iter().enumerate() {
        if !is_entrance(run, b, ta, tb, entrance_every, sw) {
            continue;
        }
        let tm = 0.5 * (ta + tb);
        let mb = run.metric(sw + 5.5);
        for j in 0..6 {
            put("prop.bollard", tm + (j as f64 - 2.5) * 1.6 / mb, sw + 5.5, 0.0);
        }
        put("prop.sign_post", tm - 6.5 / run.metric(sw + 4.0), sw + 4.0, 0.0);
        // cycle stands on the other side of the doors, a bicycle locked to the first
        for j in 0..3 {
            put("prop.bike_rack", tm + (6.0 + 0.9 * j as f64) / run.metric(sw + 4.0), sw + 4.0, 0.0);
        }
        put("genprop.bicycle", tm + (6.0 + 1.85) / run.metric(sw + 4.0), sw + 4.0, FRAC_PI_2);
    }
    // the centrepiece (facing out, to the people arriving)
    let span = run.t1 - run.t0;
    let (tm, ms) = (run.t0 + 0.5 * span, run.metric(s_l + 8.0));
    if n >= 2 {
        match run.name.as_str() {
            "north" => put("genprop.sculpture_crest", tm, s_l + 8.0, PI),
            "east" => put("genprop.clock_pillar", tm, s_l + 8.0, PI),
            "west" => put("genprop.bronze_football_plinth", tm, s_l + 8.0, PI),
            "south" => {
                put("genprop.statue_legend", tm, s_l + 8.0, PI);
                put("genprop.plaque_post_flowers", tm + 2.6 / ms, s_l + 7.0, PI);
                put("genprop.memorial_stone", tm - 3.2 / ms, s_l + 7.5, PI);
                put("genprop.merch_stall", run.t0 + 0.38 * span, s_l + 5.0, PI);
                put("genprop.flags_tifo", run.t0 + 0.38 * span + 2.6 / run.metric(s_l + 5.0), s_l + 5.0, PI);
                put("genprop.pillar_box", run.t0 + 0.62 * span, s_l - 2.0, PI);
                if n >= 6 {
                    put("genprop.statue_manager", run.t0 + 0.25 * span, s_l + 8.0, PI);
                    put("genprop.fountain_basin", run.t0 + 0.75 * span, s_l + 9.0, 0.0);
                }
            }
            // (none on a curved corner)
            _ => {}
        }
    }
    out
}

/// Glazed ground floor (concourse frontage) under a canopy, where the cladding starts above it.
fn shopfront(acc: &mut MeshAcc, run: &Run, sw: f64, t0: f64, t1: f64, doors: &[(f64, f64)]) {
    let out_s = sw + 0.3;
    let hw = |width: f64, s: f64| 0.5 * width / run.metric(s);
    for (ta, tb, door) in split(t0, t1, doors) {
        if !door {
            prism(acc, run, ta, tb, sw + 0.3, sw + 0.34, 0.0, GROUND - 0.35, TAG_DARK);
        }
    }
    let mut ts = repeat(run, out_s, 1.6, t0, t1, 0.0);
    ts.push(t1);
    for t in ts {
        if !doors.iter().any(|&(a, b)| a < t && t < b) {
            prism(acc, run, t - hw(0.08, out_s), t + hw(0.08, out_s), sw + 0.3, sw + 0.46, 0.0, GROUND - 0.35, TAG_TRIM);
        }
    }
    prism(acc, run, t0, t1, sw + 0.3, sw + 1.6, GROUND - 0.35, GROUND, TAG_TRIM);
}

/// Every `every`-th bay has an entrance, where its doors fit on the run (not past a stand end).
pub fn is_entrance(run: &Run, b: usize, ta: f64, tb: f64, every: usize, sw: f64) -> bool {
    if b % every != 0 {
        return false;
    }
    let (tm, half) = (0.5 * (ta + tb), 0.5 * DOOR_CLEAR / run.metric(sw + 0.3));
    tm - half >= run.t0 && tm + half <= run.t1
}

/// A turnstile block beside an entrance: a kiosk standing out from the facade with two turnstile
/// openings in its face, a flat roof and a sign band in the accent colour.
fn turnstiles(acc: &mut MeshAcc, run: &Run, tm: f64, sw: f64) {
    let hw = |width: f64, s: f64| 0.5 * width / run.metric(s);
    let s1 = sw + 2.5;
    let tc = tm + 6.6 / run.metric(s1);
    if tc + hw(3.9, s1) > run.t1 {
        return; // no room past the doors before the stand end
    }
    kiosk(acc, run, tc, sw);
}

fn kiosk(acc: &mut MeshAcc, run: &Run, tc: f64, sw: f64) {
    let hw = |width: f64, s: f64| 0.5 * width / run.metric(s);
    let s1 = sw + 2.5;
    prism(acc, run, tc - hw(3.6, s1), tc + hw(3.6, s1), sw + 0.3, s1, 0.0, 2.9, TAG_CORE);
    prism(acc, run, tc - hw(3.9, s1), tc + hw(3.9, s1), sw + 0.3, s1 + 0.2, 2.9, 3.1, TAG_TRIM);
    prism(acc, run, tc - hw(3.4, s1), tc + hw(3.4, s1), s1, s1 + 0.04, 2.25, 2.8, TAG_ACCENT);
    for dx in [-0.8, 0.8] {
        let tk = tc + dx / run.metric(s1);
        prism(acc, run, tk - hw(1.1, s1), tk + hw(1.1, s1), s1, s1 + 0.02, 0.0, 2.1, TAG_DARK);
    }
}

/// A club stand's back wall lower than this gets doors, a tea bar, vents and downpipes.
pub const SMALL_FACADE: f64 = 9.0;

/// The back of a small club stand: a turnstile block at one end, steel doors in frames, a tea-bar
/// hatch with a roller shutter, counter, canopy and sign in the middle, high louvre vents and
/// downpipes.
fn frontage(acc: &mut MeshAcc, run: &Run, sw: f64, h: f64) {
    let out_s = sw + 0.3;
    let hw = |width: f64, s: f64| 0.5 * width / run.metric(s);
    let m = run.metric(out_s);
    let l = (run.t1 - run.t0) * m;
    let tmid = 0.5 * (run.t0 + run.t1);
    let has_kiosk = l > 14.0;
    if has_kiosk {
        kiosk(acc, run, run.t0 + 2.4 / run.metric(sw + 2.5), sw);
    }
    prism(acc, run, tmid - hw(2.8, out_s), tmid + hw(2.8, out_s), sw + 0.3, sw + 0.38, 0.9, 2.45, TAG_TRIM);
    prism(acc, run, tmid - hw(2.6, out_s), tmid + hw(2.6, out_s), sw + 0.3, sw + 0.4, 1.0, 2.3, TAG_CORE);
    prism(acc, run, tmid - hw(2.8, out_s), tmid + hw(2.8, out_s), sw + 0.3, sw + 0.75, 0.95, 1.0, TAG_TRIM);
    prism(acc, run, tmid - hw(3.2, out_s), tmid + hw(3.2, out_s), sw + 0.3, sw + 1.3, 2.45, 2.55, TAG_TRIM);
    prism(acc, run, tmid - hw(2.6, out_s), tmid + hw(2.6, out_s), sw + 0.3, sw + 0.36, 2.6, 3.1, TAG_ACCENT);
    let n = 1usize.max((l / 8.0) as i64 as usize);
    for k in 0..n {
        let t = run.t0 + (k as f64 + 0.5) * (run.t1 - run.t0) / n as f64;
        if (t - tmid).abs() * m < 3.6 || (has_kiosk && (t - run.t0) * m < 5.0) {
            continue;
        }
        prism(acc, run, t - hw(1.24, out_s), t + hw(1.24, out_s), sw + 0.3, sw + 0.38, 0.0, 2.27, TAG_TRIM);
        prism(acc, run, t - hw(1.0, out_s), t + hw(1.0, out_s), sw + 0.3, sw + 0.42, 0.0, 2.1, TAG_DARK);
    }
    let nv = 1usize.max((l / 6.0) as i64 as usize);
    for k in 0..nv {
        let t = run.t0 + (k as f64 + 0.5) * (run.t1 - run.t0) / nv as f64;
        prism(acc, run, t - hw(0.6, out_s), t + hw(0.6, out_s), sw + 0.3, sw + 0.36, h - 1.9, h - 1.5, TAG_DARK);
    }
    let nd = 1usize.max((l / 7.0) as i64 as usize);
    for k in 0..=nd {
        let shift = if k == 0 { 0.3 / m } else if k == nd { -0.3 / m } else { 0.0 };
        let t = run.t0 + shift + k as f64 * (run.t1 - run.t0) / nd as f64;
        prism(acc, run, t - hw(0.1, out_s), t + hw(0.1, out_s), sw + 0.32, sw + 0.44, 0.0, h - 0.6, TAG_DARK);
    }
}

/// Where the cladding starts on a tall facade: a glazed concourse (floor slabs showing through)
/// fills about the lower 40%, the skin the rest.
pub fn skin_base(h: f64, ground: f64) -> f64 {
    if h <= 3.0 * LEVEL {
        return ground;
    }
    ground.max((0.4 * h / LEVEL).floor() * LEVEL)
}

/// Concourse glazing between z0 and z1: dark glass, floor-slab edges every level, mullions.
#[allow(clippy::too_many_arguments)]
fn concourse(acc: &mut MeshAcc, run: &Run, sw: f64, t0: f64, t1: f64, z0: f64, z1: f64) {
    let out_s = sw + 0.3;
    let hw = |width: f64, s: f64| 0.5 * width / run.metric(s);
    prism(acc, run, t0, t1, sw + 0.3, sw + 0.34, z0, z1, TAG_DARK);
    for z in arange(z0 + LEVEL, z1 - 0.5, LEVEL) {
        prism(acc, run, t0, t1, sw + 0.3, sw + 0.7, z - 0.2, z + 0.2, TAG_TRIM);
    }
    let mut ts = repeat(run, out_s, 1.6, t0, t1, 0.0);
    ts.push(t1);
    for t in ts {
        prism(acc, run, t - hw(0.08, out_s), t + hw(0.08, out_s), sw + 0.3, sw + 0.45, z0, z1, TAG_TRIM);
    }
}

/// A stair tower: a solid core with a full-height glazed stair window, its landings showing as
/// bands across the glass, and a cap.
fn core(acc: &mut MeshAcc, run: &Run, tm: f64, sw: f64, h: f64) {
    let s1 = sw + 6.0;
    let hw = |width: f64, s: f64| 0.5 * width / run.metric(s);
    prism(acc, run, tm - hw(7.0, s1), tm + hw(7.0, s1), sw, s1, 0.0, h + 1.5, TAG_CORE);
    prism(acc, run, tm - hw(3.0, s1), tm + hw(3.0, s1), s1, s1 + 0.03, 0.5, h + 1.0, TAG_DARK);
    for z in arange(3.5, h + 1.0, 3.5) {
        prism(acc, run, tm - hw(3.2, s1), tm + hw(3.2, s1), s1, s1 + 0.12, z - 0.12, z + 0.12, TAG_TRIM);
    }
    prism(acc, run, tm - hw(7.6, s1), tm + hw(7.6, s1), sw, s1 + 0.3, h + 1.5, h + 1.8, TAG_TRIM);
}

fn gable(acc: &mut MeshAcc, run: &Run, tiers: &[Tier], sw: f64, h: f64, at_start: bool, kind: &str) {
    let mut hull = section_hull(tiers);
    hull.extend([DVec2::new(sw + 0.3, 0.0), DVec2::new(sw + 0.3, h)]);
    let poly = convex_hull(hull);
    let tag = if matches!(kind, "brick" | "metal_bands" | "fins") { TAG_MAIN } else { TAG_CORE };
    let d = 0.3 / run.metric(0.0);
    let (ta, tb) = if at_start { (run.t0 - d, run.t0) } else { (run.t1, run.t1 + d) };
    acc.add_prism(run, ta, tb, &Section::uniform(poly.clone(), tag), tag);
    // a coping along the raking top and a plinth, both standing a little proud of the wall
    let e = 0.06 / run.metric(0.0);
    let top = (0..poly.len()).max_by(|&i, &j| (poly[i].x, poly[i].y).partial_cmp(&(poly[j].x, poly[j].y)).unwrap()).unwrap();
    let mut upper: Vec<DVec2> = poly[top..].to_vec();
    upper.push(poly[0]);
    for w in upper.windows(2) {
        let (a, b) = (w[0], w[1]);
        if (b.x - a.x).abs() < 0.05 {
            continue;
        }
        let band = vec![a, b, DVec2::new(b.x, b.y + 0.15), DVec2::new(a.x, a.y + 0.15)];
        acc.add_prism(run, ta - e, tb + e, &Section::uniform(band, TAG_TRIM), TAG_TRIM);
    }
    let s_lo = poly[0].x;
    acc.add_prism(run, ta - e, tb + e, &Section::uniform(rect(s_lo, sw + 0.3, 0.0, 0.6), TAG_TRIM), TAG_TRIM);
}

fn etfe(acc: &mut MeshAcc, tubes: &mut Vec<Polyline>, run: &Run, s: f64, h: f64, spec: &Value, z0: f64) {
    let (mut wd, hd) = match spec.get("cushion").and_then(|v| v.as_array()) {
        Some(a) if a.len() == 2 => (a[0].as_f64().unwrap_or(4.0), a[1].as_f64().unwrap_or(4.0)),
        _ => (4.0, 4.0),
    };
    let bulge = num(spec, "bulge", 0.35);
    let m = run.metric(s);
    let length = (run.t1 - run.t0) * m;
    let n_u = 1i64.max(round_half_even(length / wd) as i64);
    wd = length / n_u as f64;
    let n_z = 1i64.max(((h - z0 - 0.5) / hd) as i64);
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
