//! Parametric buildings (port of kit/site/buildings.py): a rectangular block of storeys with a
//! plinth, windows and doors laid out per facade, a roof (flat with parapet, gable, hip,
//! mono-pitch, barrel), and the details that make a style (quoins and a portico, exposed frames
//! and roller doors, timber and solar panels, render and an arcade, fins and an LED band). The same
//! generator makes the club's buildings and, with decay, the derelict ones it inherits.
//!
//! A building's local frame: centred on its footprint, front face towards -y, z up from the ground.

use super::kernel::{self, bx, cylinder, quad, tri, Frame};
use super::{mats, Out};
use crate::catalog::{num, text, Catalog};
use crate::mesh::MeshAcc;
use crate::prng::uniform;
use crate::stadium::Mat;
use crate::stands::Polyline;
use glam::{DVec2, DVec3};
use serde_json::Value;

pub const T_WALL: i32 = 0;
pub const T_BASE: i32 = 1;
pub const T_ROOF: i32 = 2;
pub const T_GLASS: i32 = 3;
pub const T_TRIM: i32 = 4;
pub const T_DOOR: i32 = 5;
pub const T_ACCENT: i32 = 6;
pub const T_METAL: i32 = 7;
pub const T_STONE: i32 = 8;
pub const T_LED: i32 = 9;
pub const T_SOLAR: i32 = 10;
pub const T_BOARD: i32 = 11;
pub const T_DARK: i32 = 12;
pub const T_LIT: i32 = 13;
pub const T_BLIND: i32 = 14;
pub const T_GRAVEL: i32 = 15;

pub const PLINTH: f64 = 0.45;

#[derive(Clone, Debug)]
pub struct Spec {
    pub w: f64,
    pub d: f64,
    pub storeys: usize,
    pub storey_h: f64,
    /// flat | gable | hip | mono | shallow | barrel | none
    pub roof: String,
    pub pitch_deg: f64,
    pub eaves: f64,
    pub parapet: f64,
    /// ribbon | punched | strip | large | shuttered | none
    pub window: String,
    /// glass | timber | none
    pub door: String,
    pub roller_doors: usize,
    /// cantilever | portico | steel | timber | arcade | none
    pub canopy: String,
    pub quoins: bool,
    pub clock: bool,
    pub fins: bool,
    pub led_band: bool,
    pub arcade: bool,
    pub solar: bool,
    pub frame: bool,
    /// 0 = kept, 1 = derelict (broken and boarded windows, holes in the roof)
    pub decay: f64,
    pub sign: Option<String>,
    /// fraction of windows lit (at night)
    pub lit: f64,
    pub seed: u64,
    pub mats: Vec<Mat>,
}

pub struct Built {
    /// the front door (building-local xy)
    pub entrance: DVec2,
    pub height: f64,
}

/// The materials of a building, by tag.
#[allow(clippy::too_many_arguments)]
pub fn materials(wall: &str, wall_color: &str, base: &str, roof: &str, roof_color: Option<&str>, trim_color: &str, club: &str, night: bool) -> Vec<Mat> {
    let tint_or_named = |k: &str, c: Option<&str>| match c {
        Some(c) => mats::tint(k, c),
        None => mats::named(k),
    };
    vec![
        mats::tint(wall, wall_color),
        mats::named(base),
        tint_or_named(roof, roof_color),
        mats::named("Facade_DarkGlass"),
        mats::tint("TINT_trim", trim_color),
        mats::tint("TINT_door", club),
        mats::tint("TINT_club", club),
        mats::tint("TINT_trim", "#8b8e91"),
        mats::named("Facade_Stone"),
        mats::glow("TINT_club", club, if night { 3.0 } else { 0.4 }),
        mats::named("Solar_Panel"),
        mats::named("Site_Plywood"),
        mats::named("Bldg_Opening"),
        if night { mats::glow("Bldg_Glass_Lit", "#ffd8a6", 1.6) } else { mats::named("Facade_DarkGlass") },
        mats::tint("Facade_DarkGlass", "#8f8b82"), // blinds behind the glazing (its reflection over them)
        mats::named("Site_Gravel"),                // pebble margins round a green roof
    ]
}

/// A building spec in a catalogued style (catalog/styles.json) with the player's colours.
#[allow(clippy::too_many_arguments)]
pub fn styled(cat: &Catalog, style: &str, w: f64, d: f64, storeys: usize, club: &str, night: bool, opts: &Value, seed: u64) -> Spec {
    let st = cat.styles.get(style).cloned().unwrap_or(Value::Null);
    let wall_color = opts.get("wall_color").and_then(|v| v.as_str()).unwrap_or(text(&st, "wall_color", "#c8c4bb")).to_string();
    let trim_color = opts.get("trim_color").and_then(|v| v.as_str()).unwrap_or(text(&st, "trim_color", "#2a2c30")).to_string();
    let roof_mat = text(&st, "roof_mat", "Roof_Flat");
    let roof_color = if roof_mat == "TINT_roof" { Some("#5d6166") } else { None };
    let b = |k: &str| st.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    Spec {
        w,
        d,
        storeys,
        storey_h: num(&st, "storey_h", 4.0),
        roof: text(&st, "roof", "flat").into(),
        pitch_deg: num(&st, "pitch_deg", 30.0),
        eaves: num(&st, "eaves", 0.4),
        parapet: num(&st, "parapet", 0.8),
        window: text(&st, "window", "punched").into(),
        door: if style == "classic" { "timber".into() } else { "glass".into() },
        roller_doors: 0,
        canopy: text(&st, "canopy", "none").into(),
        quoins: b("quoins"),
        clock: b("clock"),
        fins: b("fins"),
        led_band: b("led_band"),
        arcade: b("arcade"),
        solar: b("solar"),
        frame: b("frame"),
        decay: 0.0,
        sign: None,
        lit: if night { 0.7 } else { 0.0 },
        seed,
        mats: materials(text(&st, "wall", "Facade_Concrete"), &wall_color, text(&st, "base", "Facade_Concrete"), roof_mat, roof_color, &trim_color, club, night),
    }
}

// -- facade coordinates: side 0 front (-y), 1 right (+x), 2 back (+y), 3 left (-x); s runs along the
// side seen from outside (left to right), `out` is the distance in front of the wall

fn side_len(w: f64, d: f64, side: usize) -> f64 {
    if side % 2 == 0 {
        w
    } else {
        d
    }
}

fn fpt(w: f64, d: f64, side: usize, s: f64, z: f64, out: f64) -> DVec3 {
    match side {
        0 => DVec3::new(-w / 2.0 + s, -d / 2.0 - out, z),
        1 => DVec3::new(w / 2.0 + out, -d / 2.0 + s, z),
        2 => DVec3::new(w / 2.0 - s, d / 2.0 + out, z),
        _ => DVec3::new(-w / 2.0 - out, d / 2.0 - s, z),
    }
}

#[allow(clippy::too_many_arguments)]
fn fbox(m: &mut MeshAcc, w: f64, d: f64, side: usize, s0: f64, s1: f64, z0: f64, z1: f64, out0: f64, out1: f64, tag: i32) {
    let a = fpt(w, d, side, s0, z0, out0);
    let b = fpt(w, d, side, s1, z1, out1);
    bx(m, a.x.min(b.x), a.y.min(b.y), z0, a.x.max(b.x), a.y.max(b.y), z1, tag);
}

#[allow(clippy::too_many_arguments)]
fn fquad(m: &mut MeshAcc, w: f64, d: f64, side: usize, s0: f64, s1: f64, z0: f64, z1: f64, out: f64, tag: i32) {
    quad(m, fpt(w, d, side, s0, z0, out), fpt(w, d, side, s1, z0, out), fpt(w, d, side, s1, z1, out), fpt(w, d, side, s0, z1, out), tag);
}

/// A polygon in a facade's plane, counter-clockwise seen from outside, extruded between two offsets.
#[allow(clippy::too_many_arguments)]
fn fpoly(m: &mut MeshAcc, w: f64, d: f64, side: usize, pts: &[DVec2], out0: f64, out1: f64, tag: i32) {
    let n = pts.len();
    let front: Vec<DVec3> = pts.iter().map(|p| fpt(w, d, side, p.x, p.y, out1)).collect();
    let base = kernel::add_verts(m, &front);
    m.add_face((base..base + n as u32).collect(), tag, Some(pts.to_vec()));
    let back: Vec<DVec3> = pts.iter().rev().map(|p| fpt(w, d, side, p.x, p.y, out0)).collect();
    let base = kernel::add_verts(m, &back);
    m.add_face((base..base + n as u32).collect(), tag, Some(pts.iter().rev().cloned().collect()));
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        quad(m, fpt(w, d, side, a.x, a.y, out0), fpt(w, d, side, b.x, b.y, out0), fpt(w, d, side, b.x, b.y, out1), fpt(w, d, side, a.x, a.y, out1), tag);
    }
}

/// One pane of glazing: occupied rooms have their blinds part-way down (a pale band at the top
/// of the pane, a different drop in every window), classic windows get glazing bars.
#[allow(clippy::too_many_arguments)]
fn pane(m: &mut MeshAcc, sp: &Spec, side: usize, s0: f64, s1: f64, z0: f64, z1: f64, tag: i32, k: u64, bars: (usize, usize)) {
    let (w, d) = (sp.w, sp.d);
    let mut top = z1;
    if tag == T_GLASS && sp.decay == 0.0 {
        let r = uniform(sp.seed + 303, k);
        if r > 0.62 {
            top = z1 - (z1 - z0) * (0.1 + 0.4 * (r - 0.62) / 0.38);
            fquad(m, w, d, side, s0, s1, top, z1, 0.015, T_BLIND);
            fbox(m, w, d, side, s0, s1, top - 0.025, top, 0.0, 0.02, T_TRIM); // the blind's bottom rail
        }
    }
    fquad(m, w, d, side, s0, s1, z0, top, 0.01, tag);
    let (nv, nh) = bars;
    for j in 1..=nv {
        let s = s0 + (s1 - s0) * j as f64 / (nv + 1) as f64;
        fbox(m, w, d, side, s - 0.02, s + 0.02, z0, z1, 0.0, 0.05, T_TRIM);
    }
    for j in 1..=nh {
        let z = z0 + (z1 - z0) * j as f64 / (nh + 1) as f64;
        fbox(m, w, d, side, s0, s1, z - 0.025, z + 0.025, 0.0, 0.05, T_TRIM);
    }
}

fn window_tag(sp: &Spec, k: u64) -> i32 {
    let r = uniform(sp.seed + 101, k);
    if sp.decay > 0.0 {
        if r < sp.decay * 0.45 {
            return T_DARK;
        }
        if r < sp.decay * 0.8 {
            return T_BOARD;
        }
        return T_GLASS;
    }
    if r < sp.lit {
        T_LIT
    } else {
        T_GLASS
    }
}

/// Windows, doors and facade details on one side.
#[allow(clippy::too_many_arguments)]
fn facade(m: &mut MeshAcc, sp: &Spec, side: usize, zs: &[f64], door: Option<(f64, f64)>, k: &mut u64) {
    let (w, d) = (sp.w, sp.d);
    let len = side_len(w, d, side);
    let sh = sp.storey_h;
    let clear = |s0: f64, s1: f64, storey: usize| -> bool {
        match door {
            Some((a, b)) if side == 0 && storey == 0 => s1 < a - 0.4 || s0 > b + 0.4,
            _ => true,
        }
    };
    for (si, &z) in zs.iter().enumerate() {
        match sp.window.as_str() {
            "ribbon" => {
                let (z0, z1) = (z + 0.85, z + sh - 0.55);
                // the band, split around the door
                let mut runs = vec![(0.6, len - 0.6)];
                if let Some((a, b)) = door.filter(|_| side == 0 && si == 0) {
                    runs = vec![(0.6, a - 0.5), (b + 0.5, len - 0.6)];
                }
                for (s0, s1) in runs {
                    if s1 - s0 < 1.0 {
                        continue;
                    }
                    // the run sits back in a reveal: a deep head (a shading lip), sill and end jambs
                    fbox(m, w, d, side, s0 - 0.05, s1 + 0.05, z0 - 0.12, z0, 0.0, 0.18, T_TRIM);
                    fbox(m, w, d, side, s0 - 0.05, s1 + 0.05, z1, z1 + 0.12, 0.0, 0.3, T_TRIM);
                    fbox(m, w, d, side, s0 - 0.05, s0, z0, z1, 0.0, 0.3, T_TRIM);
                    fbox(m, w, d, side, s1, s1 + 0.05, z0, z1, 0.0, 0.3, T_TRIM);
                    let n = ((s1 - s0) / 1.5).round().max(1.0) as usize;
                    for j in 0..=n {
                        let s = s0 + (s1 - s0) * j as f64 / n as f64;
                        fbox(m, w, d, side, s - 0.035, s + 0.035, z0, z1, 0.0, 0.09, T_TRIM);
                        if j < n {
                            *k += 1;
                            let s2 = s0 + (s1 - s0) * (j + 1) as f64 / n as f64;
                            pane(m, sp, side, s, s2, z0, z1, window_tag(sp, *k), *k, (0, 0));
                        }
                    }
                }
            }
            "strip" => {
                let (z0, z1) = (z + sh - 1.6, z + sh - 0.6);
                let n = ((len - 2.0) / 2.0).floor().max(1.0) as usize;
                let s0 = (len - n as f64 * 2.0) / 2.0;
                *k += 1;
                fquad(m, w, d, side, s0, len - s0, z0, z1, 0.01, window_tag(sp, *k));
                for j in 0..=n {
                    let s = s0 + j as f64 * 2.0;
                    fbox(m, w, d, side, s - 0.04, s + 0.04, z0, z1, 0.0, 0.07, T_TRIM);
                }
                fbox(m, w, d, side, s0, len - s0, z0 - 0.08, z0, 0.0, 0.1, T_TRIM);
                fbox(m, w, d, side, s0, len - s0, z1, z1 + 0.08, 0.0, 0.1, T_TRIM);
            }
            "none" => {}
            kind => {
                let (bay, ww, z0, z1, frame) = match kind {
                    "large" => (3.8, 2.6, z + 0.35, z + sh - 0.6, 0.12),
                    "shuttered" => (3.0, 1.1, z + 0.95, z + 0.95 + 1.7, 0.07),
                    _ => (3.2, 1.3, z + 0.9, z + 0.9 + 2.0_f64.min(sh - 1.4), 0.08),
                };
                let n = ((len - 1.6) / bay).floor().max(1.0) as usize;
                let first = (len - (n as f64 - 1.0) * bay) / 2.0;
                for j in 0..n {
                    let c = first + j as f64 * bay;
                    let (s0, s1) = (c - ww / 2.0, c + ww / 2.0);
                    if s0 < 0.4 || s1 > len - 0.4 || !clear(s0, s1, si) {
                        continue;
                    }
                    *k += 1;
                    let bars = if kind == "punched" && sp.quoins { (1, 1) } else if kind == "shuttered" { (1, 2) } else { (0, 0) };
                    pane(m, sp, side, s0, s1, z0, z1, window_tag(sp, *k), *k, bars);
                    // the window sits back in its surround (a reveal deep enough to cast a shadow line)
                    fbox(m, w, d, side, s0 - frame, s0, z0, z1, 0.0, 0.2, T_TRIM);
                    fbox(m, w, d, side, s1, s1 + frame, z0, z1, 0.0, 0.2, T_TRIM);
                    fbox(m, w, d, side, s0 - frame, s1 + frame, z1, z1 + frame, 0.0, 0.2, T_TRIM);
                    // sill
                    fbox(m, w, d, side, s0 - frame - 0.05, s1 + frame + 0.05, z0 - 0.08, z0, 0.0, 0.24, if sp.quoins { T_STONE } else { T_TRIM });
                    if kind == "large" {
                        fbox(m, w, d, side, c - 0.04, c + 0.04, z0, z1, 0.0, 0.08, T_TRIM);
                    }
                    if kind == "punched" && sp.quoins {
                        // a stone lintel over each classic window
                        fbox(m, w, d, side, s0 - 0.15, s1 + 0.15, z1 + frame, z1 + frame + 0.22, 0.0, 0.05, T_STONE);
                    }
                    if kind == "shuttered" {
                        for (a, b) in [(s0 - 0.6, s0 - 0.05), (s1 + 0.05, s1 + 0.6)] {
                            fbox(m, w, d, side, a, b, z0, z1, 0.02, 0.06, T_TRIM);
                        }
                    }
                }
            }
        }
        // a string course between storeys
        if si > 0 && (sp.quoins || sp.window == "shuttered") {
            fbox(m, w, d, side, -0.05, len + 0.05, z - 0.1, z + 0.08, 0.0, 0.07, T_STONE);
        }
    }
    if sp.fins && side == 0 {
        let top = zs[0] + sp.storey_h * zs.len() as f64;
        let n = (len / 1.25).floor() as usize;
        for j in 0..=n {
            let s = j as f64 * len / n.max(1) as f64;
            fbox(m, w, d, side, s - 0.06, s + 0.06, zs[0] + 0.2, top, 0.0, 0.55, T_TRIM);
        }
    }
    if sp.frame {
        // a clad portal frame: the steel is inside; outside are the folded flashings in the trim
        // colour (corners, the drip over the plinth, the gutter along the eaves)
        let top = zs[0] + sp.storey_h * zs.len() as f64;
        for (a, b) in [(0.0, 0.16), (len - 0.16, len)] {
            fbox(m, w, d, side, a, b, zs[0] - 0.05, top, 0.0, 0.05, T_TRIM);
        }
        fbox(m, w, d, side, 0.0, len, zs[0] - 0.04, zs[0] + 0.04, 0.0, 0.05, T_TRIM);
        fbox(m, w, d, side, -0.05, len + 0.05, top - 0.22, top, 0.0, 0.16, T_TRIM);
        if len >= w.max(d) - 1e-6 {
            // rainwater downpipes off the eaves gutter, one every 15-20 m, into a shoe at the foot
            let n = (len / 18.0).round().max(1.0) as usize;
            for j in 0..n {
                let s = (j as f64 + 0.5) * len / n as f64;
                fbox(m, w, d, side, s - 0.05, s + 0.05, 0.15, top - 0.22, 0.03, 0.13, T_TRIM);
                fbox(m, w, d, side, s - 0.06, s + 0.06, 0.0, 0.15, 0.03, 0.22, T_TRIM);
            }
            // fire exits between every other pair of downpipes (not on the front with its doors):
            // steel double doors in a frame under a hood, out onto a concrete landing and step
            let z0 = zs[0];
            if side != 0 {
                for j in (1..n).step_by(2) {
                    let c = j as f64 * len / n as f64;
                    fquad(m, w, d, side, c - 0.9, c - 0.01, z0, z0 + 2.1, 0.03, T_DOOR);
                    fquad(m, w, d, side, c + 0.01, c + 0.9, z0, z0 + 2.1, 0.03, T_DOOR);
                    fbox(m, w, d, side, c - 1.0, c - 0.9, z0, z0 + 2.2, 0.0, 0.08, T_TRIM);
                    fbox(m, w, d, side, c + 0.9, c + 1.0, z0, z0 + 2.2, 0.0, 0.08, T_TRIM);
                    fbox(m, w, d, side, c - 1.0, c + 1.0, z0 + 2.1, z0 + 2.2, 0.0, 0.08, T_TRIM);
                    fbox(m, w, d, side, c - 1.3, c + 1.3, z0 + 2.45, z0 + 2.55, 0.0, 0.7, T_TRIM);
                    fbox(m, w, d, side, c - 1.4, c + 1.4, 0.0, z0, 0.04, 1.6, T_BASE); // (from the plinth's face)
                    fbox(m, w, d, side, c - 1.4, c + 1.4, 0.0, z0 * 0.5, 1.6, 1.95, T_BASE);
                }
            }
        }
    }
}

/// The roof over walls of height h. Ridges run along the longer side.
fn roof(m: &mut MeshAcc, sp: &Spec, h: f64, tubes: &mut Vec<Polyline>) {
    let (w, d) = (sp.w, sp.d);
    let e = sp.eaves;
    match sp.roof.as_str() {
        "flat" | "none" => {
            if sp.roof == "flat" {
                quad(m, DVec3::new(-w / 2.0, -d / 2.0, h), DVec3::new(w / 2.0, -d / 2.0, h), DVec3::new(w / 2.0, d / 2.0, h), DVec3::new(-w / 2.0, d / 2.0, h), T_ROOF);
                let p = sp.parapet;
                let t = 0.3;
                for (x0, y0, x1, y1) in [(-w / 2.0, -d / 2.0, w / 2.0, -d / 2.0 + t), (-w / 2.0, d / 2.0 - t, w / 2.0, d / 2.0), (-w / 2.0, -d / 2.0 + t, -w / 2.0 + t, d / 2.0 - t), (w / 2.0 - t, -d / 2.0 + t, w / 2.0, d / 2.0 - t)] {
                    bx(m, x0, y0, h, x1, y1, h + p, T_WALL);
                    bx(m, x0 - 0.03, y0 - 0.03, h + p, x1 + 0.03, y1 + 0.03, h + p + 0.06, T_METAL);
                }
                // plant on the roof
                if sp.decay <= 0.0 && w > 12.0 {
                    bx(m, w * 0.12, -d * 0.15, h, w * 0.12 + 3.2, -d * 0.15 + 2.0, h + 1.6, T_METAL);
                    bx(m, -w * 0.3, d * 0.05, h, -w * 0.3 + 2.4, d * 0.05 + 2.4, h + 1.1, T_METAL);
                }
                if sp.solar {
                    solar_rows(m, -w / 2.0 + 1.5, w / 2.0 - 1.5, -d / 2.0 + 1.5, d / 2.0 - 1.5, h + 0.3, 0.0);
                }
            }
        }
        "hip" => {
            let (l, s, swap) = if w >= d { (w, d, false) } else { (d, w, true) };
            let mut r = MeshAcc::new();
            let tp = sp.pitch_deg.to_radians().tan();
            let (lx, dy) = (l / 2.0 + e, s / 2.0 + e);
            let ze = h - e * tp;
            let zr = h + s / 2.0 * tp;
            let rx = (l / 2.0 - s / 2.0).max(0.0);
            let v = DVec3::new;
            quad(&mut r, v(-lx, -dy, ze), v(lx, -dy, ze), v(rx, 0.0, zr), v(-rx, 0.0, zr), T_ROOF);
            quad(&mut r, v(lx, dy, ze), v(-lx, dy, ze), v(-rx, 0.0, zr), v(rx, 0.0, zr), T_ROOF);
            tri(&mut r, v(lx, -dy, ze), v(lx, dy, ze), v(rx, 0.0, zr), T_ROOF);
            tri(&mut r, v(-lx, dy, ze), v(-lx, -dy, ze), v(-rx, 0.0, zr), T_ROOF);
            for (x0, y0, x1, y1) in [(-lx, -dy - 0.03, lx, -dy + 0.12), (-lx, dy - 0.12, lx, dy + 0.03), (-lx - 0.03, -dy, -lx + 0.12, dy), (lx - 0.12, -dy, lx + 0.03, dy)] {
                bx(&mut r, x0, y0, ze - 0.22, x1, y1, ze + 0.02, T_TRIM);
            }
            if rx > 0.0 {
                bx(&mut r, -rx, -0.1, zr - 0.05, rx, 0.1, zr + 0.1, T_ROOF);
            }
            if swap {
                rot90(&mut r);
            }
            m.append(&r, 0);
        }
        "mono" => {
            let tp = sp.pitch_deg.to_radians().tan();
            let rise = d * tp;
            let (lx, dy) = (w / 2.0 + e, d / 2.0 + e);
            let zf = h + (d + e) * tp;
            let zb = h - e * tp;
            let v = DVec3::new;
            quad(m, v(-lx, -dy, zf), v(lx, -dy, zf), v(lx, dy, zb), v(-lx, dy, zb), T_ROOF);
            // the high front wall and the side triangles
            fquad(m, w, d, 0, 0.0, w, h, h + rise, 0.0, T_WALL);
            tri(m, v(w / 2.0, -d / 2.0, h), v(w / 2.0, d / 2.0, h), v(w / 2.0, -d / 2.0, h + rise), T_WALL);
            tri(m, v(-w / 2.0, d / 2.0, h), v(-w / 2.0, -d / 2.0, h), v(-w / 2.0, -d / 2.0, h + rise), T_WALL);
            bx(m, -lx, -dy - 0.05, zf - 0.3, lx, -dy + 0.1, zf + 0.05, T_TRIM);
            bx(m, -lx, dy - 0.1, zb - 0.25, lx, dy + 0.05, zb + 0.05, T_TRIM);
            if sp.mats[T_ROOF as usize].key == "Roof_Sedum" {
                // a pebble margin round the green roof (a fire break, and drainage at the eaves)
                let zr = |y: f64| zf + (zb - zf) * (y + dy) / (2.0 * dy) + 0.02;
                let b = 0.45;
                for (x0, y0, x1, y1) in [(-lx, -dy, lx, -dy + b), (-lx, dy - b, lx, dy), (-lx, -dy + b, -lx + b, dy - b), (lx - b, -dy + b, lx, dy - b)] {
                    quad(m, v(x0, y0, zr(y0)), v(x1, y0, zr(y0)), v(x1, y1, zr(y1)), v(x0, y1, zr(y1)), T_GRAVEL);
                }
            }
            if sp.solar {
                solar_rows(m, -w / 2.0 + 1.0, w / 2.0 - 1.0, -d / 2.0 + 1.2, d / 2.0 - 1.2, h + 0.15, sp.pitch_deg);
            }
        }
        "barrel" => {
            let (l, s, swap) = if w >= d { (w, d, false) } else { (d, w, true) };
            let mut r = MeshAcc::new();
            let rise = 0.14 * s;
            let n = 12usize;
            let z_at = |y: f64| h + rise * (1.0 - (y / (s / 2.0)).powi(2));
            let ys: Vec<f64> = (0..=n).map(|i| -(s / 2.0 + e) + (s + 2.0 * e) * i as f64 / n as f64).collect();
            let segs = (l / 3.0).ceil().max(1.0) as usize;
            for g in 0..segs {
                let (x0, x1) = (-l / 2.0 - e + (l + 2.0 * e) * g as f64 / segs as f64, -l / 2.0 - e + (l + 2.0 * e) * (g + 1) as f64 / segs as f64);
                let hole = sp.decay > 0.3 && uniform(sp.seed + 202, g as u64) < sp.decay * 0.35;
                for i in 0..n {
                    let (ya, yb) = (ys[i], ys[i + 1]);
                    if hole && i > 2 && i < n - 2 {
                        continue;
                    }
                    quad(&mut r, DVec3::new(x0, ya, z_at(ya)), DVec3::new(x1, ya, z_at(ya)), DVec3::new(x1, yb, z_at(yb)), DVec3::new(x0, yb, z_at(yb)), T_ROOF);
                }
                if hole {
                    let xm = (x0 + x1) / 2.0;
                    tubes.push((ys.iter().map(|&y| DVec3::new(xm, y, z_at(y) - 0.15)).collect(), 0.06));
                }
            }
            // end walls under the arch
            for sx in [1.0, -1.0] {
                let x = sx * l / 2.0;
                let mut pts = vec![DVec3::new(x, -s / 2.0, h), DVec3::new(x, s / 2.0, h)];
                for i in 1..n {
                    let y = s / 2.0 - s * i as f64 / n as f64;
                    pts.push(DVec3::new(x, y, z_at(y)));
                }
                if sx < 0.0 {
                    pts.reverse();
                }
                let base = kernel::add_verts(&mut r, &pts);
                let uv = kernel::planar_uv(&pts);
                r.add_face((base..base + pts.len() as u32).collect(), T_WALL, Some(uv));
            }
            if swap {
                rot90(&mut r);
            }
            m.append(&r, 0);
        }
        _ => {
            // gable / shallow
            let (l, s, swap) = if w >= d { (w, d, false) } else { (d, w, true) };
            let mut r = MeshAcc::new();
            let tp = sp.pitch_deg.to_radians().tan();
            let (lx, dy) = (l / 2.0 + e, s / 2.0 + e);
            let ze = h - e * tp;
            let rise = s / 2.0 * tp;
            let zr = h + rise;
            let v = DVec3::new;
            let segs = (l / 3.0).ceil().max(1.0) as usize;
            for g in 0..segs {
                let (x0, x1) = (-lx + 2.0 * lx * g as f64 / segs as f64, -lx + 2.0 * lx * (g + 1) as f64 / segs as f64);
                let hole = sp.decay > 0.3 && uniform(sp.seed + 203, g as u64) < sp.decay * 0.35;
                if !hole || uniform(sp.seed + 204, g as u64) < 0.5 {
                    quad(&mut r, v(x0, -dy, ze), v(x1, -dy, ze), v(x1, 0.0, zr), v(x0, 0.0, zr), T_ROOF);
                }
                if !hole || uniform(sp.seed + 205, g as u64) < 0.5 {
                    quad(&mut r, v(x1, dy, ze), v(x0, dy, ze), v(x0, 0.0, zr), v(x1, 0.0, zr), T_ROOF);
                }
                if hole {
                    let xm = (x0 + x1) / 2.0;
                    for k in [-1.0, 1.0] {
                        tubes.push((vec![v(xm + k * 0.6, -dy + 0.2, ze + 0.05), v(xm + k * 0.6, 0.0, zr - 0.1), v(xm + k * 0.6, dy - 0.2, ze + 0.05)], 0.07));
                    }
                }
            }
            for sx in [1.0, -1.0] {
                let x = sx * l / 2.0;
                if sx > 0.0 {
                    tri(&mut r, v(x, -s / 2.0, h), v(x, s / 2.0, h), v(x, 0.0, zr), if swap { T_WALL } else { T_WALL });
                } else {
                    tri(&mut r, v(x, s / 2.0, h), v(x, -s / 2.0, h), v(x, 0.0, zr), T_WALL);
                }
                // barge boards
                let xb = sx * lx;
                for sy in [-1.0, 1.0] {
                    let (a, b) = (v(xb, sy * dy, ze), v(xb, 0.0, zr));
                    let down = v(0.0, 0.0, -0.24);
                    if sx * sy > 0.0 {
                        quad(&mut r, a + down, b + down, b, a, T_TRIM);
                    } else {
                        quad(&mut r, b + down, a + down, a, b, T_TRIM);
                    }
                }
            }
            for sy in [-1.0, 1.0] {
                bx(&mut r, -lx, sy * dy - 0.06, ze - 0.24, lx, sy * dy + 0.06, ze + 0.02, T_TRIM);
            }
            bx(&mut r, -lx, -0.12, zr - 0.04, lx, 0.12, zr + 0.12, T_METAL);
            if swap {
                rot90(&mut r);
            }
            m.append(&r, 0);
        }
    }
}

/// Rotate a building-local mesh by 90 degrees about z (x -> y).
fn rot90(m: &mut MeshAcc) {
    for p in m.verts.iter_mut() {
        *p = DVec3::new(-p.y, p.x, p.z);
    }
}

/// Rows of solar panels on a roof plane rising towards -y at pitch_deg.
fn solar_rows(m: &mut MeshAcc, x0: f64, x1: f64, y0: f64, y1: f64, z: f64, pitch_deg: f64) {
    let tp = pitch_deg.to_radians().tan();
    let rows = ((y1 - y0) / 2.2).floor().max(1.0) as usize;
    for r in 0..rows {
        let ya = y0 + r as f64 * 2.2;
        let yb = ya + 1.7;
        let z_of = |y: f64| z + (y1 - y) * tp;
        let (za, zb) = if pitch_deg > 0.0 { (z_of(ya), z_of(yb)) } else { (z + 0.55, z) };
        let n = ((x1 - x0) / 1.05).floor().max(1.0) as usize;
        for c in 0..n {
            let xa = x0 + c as f64 * 1.05;
            quad(m, DVec3::new(xa, ya, za), DVec3::new(xa + 1.0, ya, za), DVec3::new(xa + 1.0, yb, zb), DVec3::new(xa, yb, zb), T_SOLAR);
        }
    }
}

/// The stroke font (catalog/font_strokes.json), parsed once.
fn font() -> &'static Value {
    static FONT: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    FONT.get_or_init(|| serde_json::from_str(include_str!("../../../../catalog/font_strokes.json")).expect("catalog/font_strokes.json"))
}

/// A stroke's polyline (glyph units); arcs become segments of at most 15 degrees.
fn stroke_points(st: &Value) -> Vec<DVec2> {
    let a = st.as_array().map(|v| v.as_slice()).unwrap_or(&[]);
    if a.len() == 1 && a[0].get(0).is_some_and(|v| v.is_string()) {
        let g = |i: usize| a[0][i].as_f64().unwrap_or(0.0);
        let (cx, cy, rx, ry, a0, a1) = (g(1), g(2), g(3), g(4), g(5), g(6));
        let n = ((a1 - a0).abs() / 15.0).ceil().max(4.0) as usize;
        return (0..=n)
            .map(|t| {
                let ang = (a0 + (a1 - a0) * t as f64 / n as f64).to_radians();
                DVec2::new(cx + rx * ang.cos(), cy + ry * ang.sin())
            })
            .collect();
    }
    a.iter().map(|p| DVec2::new(p[0].as_f64().unwrap_or(0.0), p[1].as_f64().unwrap_or(0.0))).collect()
}

/// Raised letters for the club's name (a geometric sans stroke font, catalog/font_strokes.json):
/// every stroke segment is a small prism standing proud of the wall, centred at facade position
/// `s_mid`, bottom at `z`, at most `max_w` wide and 7 * `max_px` tall.
#[allow(clippy::too_many_arguments)]
pub fn letters(m: &mut MeshAcc, w: f64, d: f64, side: usize, text: &str, s_mid: f64, z: f64, max_w: f64, tag: i32, max_px: f64, out: f64) {
    let font = font();
    let (glyphs, widths) = (&font["glyphs"], &font["widths"]);
    let chars: Vec<String> = text.chars().map(|c| c.to_uppercase().collect()).collect();
    let adv: Vec<f64> = chars.iter().map(|c| widths.get(c.as_str()).and_then(|v| v.as_f64()).unwrap_or(if glyphs.get(c.as_str()).is_some() { 4.0 } else { 2.4 })).collect();
    let total = adv.iter().sum::<f64>() + 1.2 * chars.len().saturating_sub(1) as f64;
    if total <= 0.0 {
        return;
    }
    let u = (max_w / total).max(0.04).min(max_px * 7.0 / 6.0);
    let hw = 0.36 * u;
    let mut x = s_mid - total * u / 2.0;
    for (c, a) in chars.iter().zip(&adv) {
        for st in glyphs.get(c.as_str()).and_then(|v| v.as_array()).map(|v| v.as_slice()).unwrap_or(&[]) {
            let pts = stroke_points(st);
            for seg in pts.windows(2) {
                let (p0, p1) = (seg[0], seg[1]);
                let (mut ds, mut dz) = ((p1.x - p0.x) * u, (p1.y - p0.y) * u);
                let mut ln = (ds * ds + dz * dz).sqrt();
                if ln < 1e-9 {
                    (ds, dz, ln) = (0.0, 1.0, 1.0);
                }
                let (ts, tz) = (ds / ln, dz / ln);
                let (ns, nz) = (-tz * hw, ts * hw);
                let (ps, pz) = (x + p0.x * u - ts * hw, z + p0.y * u - tz * hw);
                let (qs, qz) = (x + p1.x * u + ts * hw, z + p1.y * u + tz * hw);
                let ring = [(ps - ns, pz - nz), (qs - ns, qz - nz), (qs + ns, qz + nz), (ps + ns, pz + nz)];
                let back: Vec<DVec3> = ring.iter().map(|(a_, b_)| fpt(w, d, side, *a_, *b_, out + 0.02)).collect();
                let front: Vec<DVec3> = ring.iter().map(|(a_, b_)| fpt(w, d, side, *a_, *b_, out + 0.09)).collect();
                quad(m, front[0], front[1], front[2], front[3], tag);
                for k in 0..4 {
                    let k2 = (k + 1) % 4;
                    quad(m, back[k], back[k2], front[k2], front[k], tag);
                }
            }
        }
        x += (a + 1.2) * u;
    }
}

/// Generate a building at frame `f` (its centre, front towards local -y). Returns the entrance.
pub fn build(o: &mut Out, f: &Frame, name: &str, sp: &Spec) -> Built {
    let (w, d) = (sp.w, sp.d);
    let mut m = MeshAcc::new();
    let mut tubes: Vec<Polyline> = Vec::new();
    let h = PLINTH + sp.storeys as f64 * sp.storey_h;
    let foot = kernel::rect(w, d, 0.0, 0.0);
    kernel::prism(&mut m, &kernel::rect(w + 0.08, d + 0.08, 0.0, 0.0), 0.0, PLINTH, T_BASE, None, None);
    kernel::prism(&mut m, &foot, PLINTH, h, T_WALL, None, None);
    let zs: Vec<f64> = (0..sp.storeys).map(|i| PLINTH + i as f64 * sp.storey_h).collect();
    // front door
    let door_w = if sp.door == "timber" { 1.8 } else { 2.8 };
    let door = (sp.door != "none").then(|| (w / 2.0 - door_w / 2.0, w / 2.0 + door_w / 2.0));
    let mut k = 0u64;
    for side in 0..4 {
        facade(&mut m, sp, side, &zs, door, &mut k);
    }
    if let Some((a, b)) = door {
        let dh = 2.7f64.min(sp.storey_h - 0.6);
        let tag = if sp.door == "timber" { T_DOOR } else if sp.decay > 0.5 { T_BOARD } else { T_GLASS };
        fquad(&mut m, w, d, 0, a, b, PLINTH, PLINTH + dh, 0.02, tag);
        fbox(&mut m, w, d, 0, a - 0.12, a, PLINTH, PLINTH + dh + 0.12, 0.0, 0.14, T_TRIM);
        fbox(&mut m, w, d, 0, b, b + 0.12, PLINTH, PLINTH + dh + 0.12, 0.0, 0.14, T_TRIM);
        fbox(&mut m, w, d, 0, a - 0.12, b + 0.12, PLINTH + dh, PLINTH + dh + 0.12, 0.0, 0.14, T_TRIM);
        fbox(&mut m, w, d, 0, (a + b) / 2.0 - 0.04, (a + b) / 2.0 + 0.04, PLINTH, PLINTH + dh, 0.0, 0.08, T_TRIM);
        if sp.door == "timber" && sp.storey_h - dh > 1.0 {
            // a fanlight
            fquad(&mut m, w, d, 0, a, b, PLINTH + dh + 0.12, PLINTH + dh + 0.7, 0.02, T_GLASS);
            fbox(&mut m, w, d, 0, a - 0.12, b + 0.12, PLINTH + dh + 0.7, PLINTH + dh + 0.82, 0.0, 0.14, T_TRIM);
        }
        // steps up to the plinth
        fbox(&mut m, w, d, 0, a - 0.6, b + 0.6, 0.0, PLINTH * 0.5, 0.0, 1.4, T_BASE);
        fbox(&mut m, w, d, 0, a - 0.6, b + 0.6, 0.0, PLINTH, 0.0, 0.7, T_BASE);
    }
    // roller doors (sheds, depots)
    if sp.roller_doors > 0 {
        let n = sp.roller_doors;
        let span = w / n as f64;
        for j in 0..n {
            let c = span * (j as f64 + 0.5);
            let (a, b) = (c - 2.0, c + 2.0);
            if door.is_some_and(|(da, db)| b > da - 0.5 && a < db + 0.5) {
                continue;
            }
            let top = PLINTH + 4.4f64.min(sp.storey_h * sp.storeys as f64 - 0.6);
            fquad(&mut m, w, d, 0, a, b, 0.05, top, 0.03, T_DOOR);
            let ribs = ((top - 0.05) / 0.32) as usize;
            for r in 0..ribs {
                let z = 0.05 + r as f64 * 0.32;
                fbox(&mut m, w, d, 0, a, b, z, z + 0.04, 0.03, 0.06, T_DOOR);
            }
            fbox(&mut m, w, d, 0, a - 0.15, a, 0.0, top + 0.15, 0.0, 0.2, T_TRIM);
            fbox(&mut m, w, d, 0, b, b + 0.15, 0.0, top + 0.15, 0.0, 0.2, T_TRIM);
            fbox(&mut m, w, d, 0, a - 0.15, b + 0.15, top, top + 0.5, 0.0, 0.3, T_TRIM);
        }
    }
    // corners
    if sp.quoins {
        let mut z = PLINTH;
        let mut i = 0;
        while z + 0.32 < h {
            let long = if i % 2 == 0 { 0.6 } else { 0.35 };
            for side in 0..4 {
                let len = side_len(w, d, side);
                fbox(&mut m, w, d, side, -0.02, long, z, z + 0.3, 0.0, 0.05, T_STONE);
                fbox(&mut m, w, d, side, len - long, len + 0.02, z, z + 0.3, 0.0, 0.05, T_STONE);
            }
            z += 0.34;
            i += 1;
        }
    }
    if sp.led_band {
        for side in 0..4 {
            let len = side_len(w, d, side);
            fbox(&mut m, w, d, side, 0.0, len, h - 0.32, h - 0.1, 0.0, 0.05, T_LED);
        }
    }
    roof(&mut m, sp, h, &mut tubes);
    // entrance canopy
    let dz = PLINTH + 2.7f64.min(sp.storey_h - 0.6) + 0.4;
    match sp.canopy.as_str() {
        "cantilever" => {
            fbox(&mut m, w, d, 0, w / 2.0 - 5.0, w / 2.0 + 5.0, dz + 0.4, dz + 0.75, 0.0, 3.6, T_TRIM);
            if sp.led_band {
                fbox(&mut m, w, d, 0, w / 2.0 - 5.0, w / 2.0 + 5.0, dz + 0.32, dz + 0.4, 3.45, 3.6, T_LED);
            }
        }
        "portico" => {
            let depth = 3.0;
            for sx in [-1.0, 1.0] {
                let p = fpt(w, d, 0, w / 2.0 + sx * 2.2, 0.0, depth - 0.35);
                cylinder(&mut m, p.x, p.y, 0.0, dz + 0.9, 0.24, 12, T_STONE, true);
            }
            fbox(&mut m, w, d, 0, w / 2.0 - 2.7, w / 2.0 + 2.7, dz + 0.9, dz + 1.35, 0.0, depth, T_STONE);
            // pediment
            let pts = [DVec2::new(w / 2.0 - 2.9, dz + 1.35), DVec2::new(w / 2.0 + 2.9, dz + 1.35), DVec2::new(w / 2.0, dz + 2.45)];
            fpoly(&mut m, w, d, 0, &pts, depth - 0.25, depth + 0.05, T_STONE);
            fpoly(&mut m, w, d, 0, &pts, 0.0, 0.3, T_STONE);
            let v = |s: f64, z: f64, out: f64| fpt(w, d, 0, s, z, out);
            quad(&mut m, v(w / 2.0 - 3.0, dz + 1.3, depth + 0.1), v(w / 2.0, dz + 2.5, depth + 0.1), v(w / 2.0, dz + 2.5, 0.0), v(w / 2.0 - 3.0, dz + 1.3, 0.0), T_ROOF);
            quad(&mut m, v(w / 2.0, dz + 2.5, depth + 0.1), v(w / 2.0 + 3.0, dz + 1.3, depth + 0.1), v(w / 2.0 + 3.0, dz + 1.3, 0.0), v(w / 2.0, dz + 2.5, 0.0), T_ROOF);
        }
        "steel" => {
            fbox(&mut m, w, d, 0, w / 2.0 - 3.5, w / 2.0 + 3.5, dz + 0.3, dz + 0.45, 0.0, 3.0, T_METAL);
            for sx in [-1.0, 1.0] {
                fbox(&mut m, w, d, 0, w / 2.0 + sx * 3.2 - 0.08, w / 2.0 + sx * 3.2 + 0.08, 0.0, dz + 0.3, 2.7, 2.9, T_METAL);
            }
        }
        "timber" => {
            fbox(&mut m, w, d, 0, w / 2.0 - 4.0, w / 2.0 + 4.0, dz + 0.5, dz + 0.75, 0.0, 3.4, T_ROOF);
            fbox(&mut m, w, d, 0, w / 2.0 - 4.0, w / 2.0 + 4.0, dz + 0.35, dz + 0.5, 0.0, 3.4, T_WALL);
            for sx in [-1.0, 1.0] {
                fbox(&mut m, w, d, 0, w / 2.0 + sx * 3.6 - 0.1, w / 2.0 + sx * 3.6 + 0.1, 0.0, dz + 0.35, 3.0, 3.2, T_WALL);
            }
        }
        _ => {}
    }
    if sp.arcade {
        let depth = 3.0;
        let zt = PLINTH + sp.storey_h - 0.2;
        let n = ((w - 1.0) / 3.6).floor().max(2.0) as usize;
        let bay = (w - 0.6) / n as f64;
        let cols: Vec<f64> = (0..=n).map(|j| 0.3 + j as f64 * bay).collect();
        for &c in &cols {
            fbox(&mut m, w, d, 0, c - 0.24, c + 0.24, 0.0, zt, depth - 0.48, depth, T_WALL);
        }
        for j in 0..n {
            let (s0, s1) = (cols[j] + 0.24, cols[j + 1] - 0.24);
            let rr = (s1 - s0) / 2.0;
            let zs_ = zt - 0.45 - rr;
            let sm = (s0 + s1) / 2.0;
            let mut pts = vec![DVec2::new(s0, zs_)];
            for q in 1..10 {
                let a = std::f64::consts::PI - std::f64::consts::PI * q as f64 / 10.0;
                pts.push(DVec2::new(sm + rr * a.cos(), zs_ + rr * a.sin()));
            }
            pts.extend([DVec2::new(s1, zs_), DVec2::new(s1, zt), DVec2::new(s0, zt)]);
            fpoly(&mut m, w, d, 0, &pts, depth - 0.48, depth, T_WALL);
        }
        fbox(&mut m, w, d, 0, 0.0, w, zt, zt + 0.35, 0.0, depth + 0.1, T_WALL);
        let v = |s: f64, z: f64, out: f64| fpt(w, d, 0, s, z, out);
        quad(&mut m, v(-0.2, zt + 0.35, depth + 0.4), v(w + 0.2, zt + 0.35, depth + 0.4), v(w + 0.2, zt + 1.1, 0.0), v(-0.2, zt + 1.1, 0.0), T_ROOF);
    }
    if sp.clock {
        // a pediment with a clock face over the entrance, at the eaves
        let pts = [DVec2::new(w / 2.0 - 3.4, h - 0.2), DVec2::new(w / 2.0 + 3.4, h - 0.2), DVec2::new(w / 2.0, h + 2.0)];
        fpoly(&mut m, w, d, 0, &pts, 0.0, 0.25, T_STONE);
        let c = DVec2::new(w / 2.0, h + 0.55);
        let ring: Vec<DVec2> = (0..16).map(|q| {
            let a = std::f64::consts::TAU * q as f64 / 16.0;
            DVec2::new(c.x + 0.55 * a.cos(), c.y + 0.55 * a.sin())
        }).collect();
        fpoly(&mut m, w, d, 0, &ring, 0.25, 0.3, T_TRIM);
    }
    if let Some(t) = &sp.sign {
        // the name goes where the wall is blank: on a flat roof's parapet, else in the band between
        // the top windows' heads and the eaves (never across windows or under the cornice)
        // (else on the entrance portico's entablature or the canopy fascia; with neither, the name is
        // left to the sign at the gate)
        if sp.roof == "flat" {
            let room = sp.parapet - 0.2;
            letters(&mut m, w, d, 0, t, w / 2.0, h + 0.1, (w * 0.45).min(14.0), T_ACCENT, room / 7.0, 0.0);
        } else if sp.canopy == "portico" {
            letters(&mut m, w, d, 0, t, w / 2.0, dz + 0.95, 4.8, T_ACCENT, 0.36 / 7.0, 3.0);
        } else if sp.canopy == "cantilever" {
            letters(&mut m, w, d, 0, t, w / 2.0, dz + 0.43, 9.0, T_ACCENT, 0.29 / 7.0, 3.6);
        }
    }
    if sp.decay > 0.0 {
        // drainpipes hanging off, weeds at the foot
        for (j, side) in [0usize, 2].iter().enumerate() {
            let len = side_len(w, d, *side);
            let p = fpt(w, d, *side, len * 0.15, 0.0, 0.08);
            tubes.push((vec![DVec3::new(p.x, p.y, 0.0), DVec3::new(p.x, p.y, h - 0.1 - uniform(sp.seed + 301, j as u64) * 2.0)], 0.05));
        }
    }
    o.part_at(f, name, m, sp.mats.clone(), 0.0);
    if sp.decay <= 0.0 && matches!(sp.roof.as_str(), "gable" | "shallow" | "hip") {
        o.tubes_at(f, &format!("{name}_rainwater"), rainwater(sp, h), mats::tint("TINT_trim", "#26282b"), 8);
    }
    if !tubes.is_empty() {
        o.tubes_at(f, &format!("{name}_steel"), tubes, mats::named("Site_Rust"), 6);
    }
    Built { entrance: DVec2::new(0.0, -d / 2.0 - 1.5), height: h }
}

/// Rainwater goods for a pitched roof: a gutter under each long eave and a downpipe from each
/// end of it, set back to the wall and down to a shoe at the ground.
fn rainwater(sp: &Spec, h: f64) -> Vec<Polyline> {
    let (w, d) = (sp.w, sp.d);
    let (l, s, swap) = if w >= d { (w, d, false) } else { (d, w, true) };
    let e = sp.eaves;
    let zf = h - e * sp.pitch_deg.to_radians().tan();
    let lx = l / 2.0 + e;
    let place = |x: f64, y: f64, z: f64| if swap { DVec3::new(-y, x, z) } else { DVec3::new(x, y, z) };
    let mut out = Vec::new();
    for sy in [-1.0, 1.0] {
        // the gutter hangs just outside the fascia, the downpipes come back to the wall
        let yg = sy * (s / 2.0 + e + 0.07);
        out.push((vec![place(-lx, yg, zf - 0.12), place(lx, yg, zf - 0.12)], 0.065));
        for sx in [-1.0, 1.0] {
            let x = sx * (l / 2.0 - 0.8);
            let yw = sy * (s / 2.0 + 0.07);
            out.push((
                vec![place(x, yg, zf - 0.16), place(x, yg, zf - 0.32), place(x, yw, zf - 0.7), place(x, yw, 0.25), place(x, yw + sy * 0.12, 0.08)],
                0.035,
            ));
        }
    }
    out
}

/// A ruin: walls broken down to uneven heights, no roof (a building half demolished).
#[allow(clippy::too_many_arguments)]
pub fn ruin(o: &mut Out, f: &Frame, name: &str, w: f64, d: f64, h_max: f64, wall: Mat, seed: u64) {
    let mut m = MeshAcc::new();
    bx(&mut m, -w / 2.0, -d / 2.0, 0.0, w / 2.0, d / 2.0, 0.25, 1);
    let mut k = 0u64;
    for side in 0..4 {
        let len = side_len(w, d, side);
        let n = (len / 2.4).ceil().max(1.0) as usize;
        for j in 0..n {
            k += 1;
            let r = uniform(seed + 401, k);
            if r < 0.25 {
                continue;
            }
            let (s0, s1) = (len * j as f64 / n as f64, len * (j + 1) as f64 / n as f64);
            let top = 0.4 + (h_max - 0.4) * uniform(seed + 402, k).powf(1.6);
            fbox(&mut m, w, d, side, s0, s1, 0.0, top, -0.35, 0.0, 0);
        }
    }
    o.part_at(f, name, m, vec![wall, mats::named("Facade_Concrete")], 0.0);
}

/// A small gatehouse beside the drive in a style, with the club's name.
pub fn gatehouse(o: &mut Out, cat: &Catalog, f: &Frame, style: &str) {
    let club = o.club.clone();
    let mut sp = styled(cat, style, 7.5, 5.0, 1, &club.primary, o.night, &Value::Null, 9);
    sp.canopy = "none".into();
    sp.arcade = false;
    sp.clock = false;
    sp.solar = false;
    sp.window = if sp.window == "ribbon" { "ribbon".into() } else { "large".into() };
    // beside the drive, facing it
    let g = f.sub(-(super::landscape::ROAD_W / 2.0 + 6.0), 12.0, -std::f64::consts::FRAC_PI_2);
    build(o, &g, "Gate:gatehouse", &sp);
    // a sign wall with the club's name at the entrance
    let mut m = MeshAcc::new();
    let (ww, dd) = (6.0, 0.6);
    bx(&mut m, -ww / 2.0, -dd / 2.0, 0.0, ww / 2.0, dd / 2.0, 1.4, T_WALL);
    bx(&mut m, -ww / 2.0 - 0.05, -dd / 2.0 - 0.05, 1.4, ww / 2.0 + 0.05, dd / 2.0 + 0.05, 1.5, T_STONE);
    letters(&mut m, ww, dd, 0, &club.name, ww / 2.0, 0.45, 5.0, T_ACCENT, 0.16, 0.0);
    let s = f.sub(super::landscape::GATE_W / 2.0 + 5.5, -1.0, 0.0);
    o.part_at(&s, "Gate:sign", m, sp.mats.clone(), 0.0);
}
