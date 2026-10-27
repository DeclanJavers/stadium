//! Buildings with interiors (port of kit/site/interior.py): the club's medical centre and scouting
//! building, planned from their rooms outwards and built so they can be seen into with the roof off. A corridor along the
//! building with a band of rooms each side (the quality tier's programme, each room built if it
//! still fits), stairs and a lift once there is an upper floor; an outer leaf, room linings and
//! partitions in each room's finish, doors, windows to the room's glazing, floors, suspended
//! ceilings with LED panels, a hydrotherapy pool, a stair; furniture from catalog/rooms.json.
//! Views: closed, dollhouse (roof off), cutaway (walls cut at 1.3 m).

use super::buildings;
use super::facilities::{forecourt, place};
use super::kernel::{self, bx, quad, strip, Frame};
use super::{mats, option, tier_spec, Out, Plot, COVER_Z};
use crate::catalog::{num, text, Catalog};
use crate::mesh::MeshAcc;
use crate::stadium::Mat;
use crate::stands::Polyline;
use glam::{DVec2, DVec3};
use serde_json::{json, Value};
use std::f64::consts::{FRAC_PI_2, PI};

const STOREY: f64 = 3.6;
const T_EXT: f64 = 0.3;
const T_INT: f64 = 0.12;
const CEIL: f64 = 2.7;
const CUT: f64 = 1.3;
const PAINT: [&str; 4] = ["#e7e3d7", "#eceeee", "#f1f2f1", "#f4f4f2"];

fn floor_key(kind: &str) -> &'static str {
    match kind {
        "rubber" => "Floor_Rubber",
        "tile" => "Floor_Tile",
        "carpet" => "Floor_Carpet",
        "timber" => "Site_Wood",
        _ => "Floor_Vinyl",
    }
}

fn glazing(kind: &str) -> Option<(f64, f64)> {
    match kind {
        "window" => Some((0.9, 2.3)),
        "full" => Some((0.1, 2.6)),
        "high" => Some((1.6, 2.3)),
        _ => None,
    }
}

/// Material slots of one building by key, in first-use order.
#[derive(Default)]
struct Mats {
    list: Vec<Mat>,
    keys: Vec<String>,
}

impl Mats {
    fn tag(&mut self, key: &str, make: impl FnOnce() -> Mat) -> i32 {
        if let Some(i) = self.keys.iter().position(|k| k == key) {
            return i as i32;
        }
        self.keys.push(key.to_string());
        self.list.push(make());
        (self.list.len() - 1) as i32
    }
}

#[derive(Clone)]
struct Room {
    kind: String,
    spec: Value,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    front: bool,
    storey: usize,
}

impl Room {
    /// (width, kind, centre along the corridor wall in local u).
    fn door(&self) -> (f64, String, f64) {
        let dw = num(&self.spec, "door", 0.9);
        let dk = text(&self.spec, "door_kind", "leaf").to_string();
        let w = self.x1 - self.x0;
        let u = if dk == "opening" && dw >= 2.0 { 0.0 } else { -w / 2.0 + 0.35 + dw / 2.0 };
        (dw, dk, u)
    }
    /// The room frame (origin at the middle of its corridor wall, +y into the room): (x, y, yaw).
    fn frame_xy(&self) -> (f64, f64, f64) {
        let cx = (self.x0 + self.x1) / 2.0;
        if self.front {
            (cx, self.y1, PI)
        } else {
            (cx, self.y0, 0.0)
        }
    }
    /// (the room's depth in use, the depth of the plant space behind its cross wall or 0).
    fn depth(&self) -> (f64, f64) {
        let d = self.y1 - self.y0;
        let md = num(&self.spec, "max_d", 0.0);
        if md > 0.0 && d > md + 1.2 {
            (md, d - md - T_INT)
        } else {
            (d, 0.0)
        }
    }
    fn u_to_x(&self, u: f64) -> f64 {
        let cx = (self.x0 + self.x1) / 2.0;
        if self.front {
            cx - u
        } else {
            cx + u
        }
    }
    fn pool(&self) -> bool {
        self.spec.get("pool").and_then(|v| v.as_bool()) == Some(true)
    }
}

/// The rooms of a band: kinds in priority order, each built if it still fits, then stretched.
fn fill(length: f64, depth: f64, kinds: &[String], rc: &serde_json::Map<String, Value>) -> Vec<(String, f64)> {
    let mut chosen: Vec<String> = Vec::new();
    let mut used = 0.0;
    for k in kinds {
        let Some(r) = rc.get(k).filter(|r| r.is_object()) else { continue };
        let (mw, md) = (num(r, "min_w", 4.0), num(r, "min_d", 3.0));
        if used + mw <= length + 1e-9 && md <= depth + 1e-9 {
            chosen.push(k.clone());
            used += mw;
        }
    }
    if chosen.is_empty() {
        return vec![];
    }
    let s = length / used;
    chosen.into_iter().map(|k| {
        let w = num(&rc[&k], "min_w", 4.0) * s;
        (k, w)
    }).collect()
}

fn plan(cat: &Catalog, t: &Value, w: f64, d: f64, storeys: usize) -> (Vec<Room>, (f64, f64)) {
    let rc = &cat.rooms;
    let (xi0, xi1) = (-w / 2.0 + T_EXT, w / 2.0 - T_EXT);
    let (yi0, yi1) = (-d / 2.0 + T_EXT, d / 2.0 - T_EXT);
    let inner_d = yi1 - yi0;
    let corr = if inner_d < 10.0 { 1.4 } else { 1.8 };
    let df = 6.0f64.min((inner_d - corr) * 0.6);
    let (yc0, yc1) = (yi0 + df, yi0 + df + corr);
    let list = |key: &str| -> Vec<String> { t.get(key).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default() };
    let mut out = Vec::new();
    for s in 0..storeys {
        let pre = if s == 0 { "" } else { "upper_" };
        let front = list(&format!("{pre}front"));
        let mut back = list(&format!("{pre}back"));
        if storeys > 1 {
            let mut b = vec!["stairs".to_string(), "lift".to_string()];
            b.extend(back);
            back = b;
        }
        for (is_front, kinds, y0, y1) in [(true, front, yi0, yc0), (false, back, yc1, yi1)] {
            let mut rs = fill(xi1 - xi0, y1 - y0, &kinds, rc);
            if is_front && s == 0 {
                // (the entrance, a room glazed as one, mid-front)
                if let Some(e) = rs.iter().position(|(k, _)| rc.get(k.as_str()).map(|r| text(r, "glazing", "")) == Some("entrance")) {
                    let ent = rs.remove(e);
                    let at = rs.len() / 2;
                    rs.insert(at, ent);
                }
            }
            let (ry0, ry1) = if is_front { (y0, y1 - T_INT / 2.0) } else { (y0 + T_INT / 2.0, y1) };
            let mut x = xi0;
            let n = rs.len();
            for (i, (k, rw)) in rs.into_iter().enumerate() {
                let a = x + if i > 0 { T_INT / 2.0 } else { 0.0 };
                let b = x + rw - if i + 1 < n { T_INT / 2.0 } else { 0.0 };
                out.push(Room { spec: rc[&k].clone(), kind: k, x0: a, x1: b, y0: ry0, y1: ry1, front: is_front, storey: s });
                x += rw;
            }
        }
    }
    (out, (yc0, yc1))
}

/// A wall piece: an axis-aligned box in `side`, its top face (a cut section) in `top`.
#[allow(clippy::too_many_arguments)]
fn wbox(m: &mut MeshAcc, x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64, side: i32, top: i32) {
    bx(m, x0, y0, z0, x1, y1, z1, side);
    let n = m.tags.len();
    m.tags[n - 5] = top;
}

type Opening = (f64, f64, f64, f64);

/// A wall along x (axis 0) or y (axis 1) with openings cut through it (kit/site/interior.py wall).
#[allow(clippy::too_many_arguments)]
fn wall(m: &mut MeshAcc, axis: usize, c0: f64, c1: f64, a0: f64, a1: f64, zb: f64, zt: f64, openings: &[Opening], side: i32, top: i32) {
    let piece = |m: &mut MeshAcc, p0: f64, p1: f64, z0: f64, z1: f64| {
        if p1 - p0 < 1e-6 || z1 - z0 < 1e-6 {
            return;
        }
        let tp = if z1 >= zt - 1e-9 { top } else { side };
        if axis == 0 {
            wbox(m, p0, c0, z0, p1, c1, z1, side, tp);
        } else {
            wbox(m, c0, p0, z0, c1, p1, z1, side, tp);
        }
    };
    let mut ops: Vec<Opening> = openings.iter().copied().filter(|o| o.1 > a0 + 1e-6 && o.0 < a1 - 1e-6).collect();
    ops.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut a = a0;
    for (o0, o1, oz0, oz1) in ops {
        let (o0, o1) = (o0.max(a0), o1.min(a1));
        piece(m, a, o0, zb, zt);
        piece(m, o0, o1, zb, oz0.min(zt));
        piece(m, o0, o1, oz1.max(zb), zt);
        a = o1;
    }
    piece(m, a, a1, zb, zt);
}

fn moved(sub: &mut MeshAcc, rf: &Frame) {
    for v in sub.verts.iter_mut() {
        let q = rf.pt(v.x, v.y);
        *v = DVec3::new(q.x, q.y, v.z);
    }
}

pub fn build(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize, style: &str, opts: &Value, fac: &str) {
    let t = tier_spec(cat, fac, tier).cloned().unwrap_or(json!({}));
    let sizes: Vec<Value> = cat.facilities.get(fac).and_then(|f| f.get("sizes")).and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let sv = option(cat, fac, tier, opts, "size");
    let si = (sv.as_f64().map(|v| v as i64).unwrap_or(2)).max(1).min(sizes.len().max(1) as i64) as usize;
    let sz = sizes.get(si - 1).cloned().unwrap_or(json!({"w": 28.8, "d": 15.6, "storeys": 1}));
    let storeys = num(&sz, "storeys", 1.0) as usize;
    let pl = place(p, num(&sz, "w", 28.8), num(&sz, "d", 15.6), 16.0);
    let (f, w, d) = (pl.f, pl.w, pl.d);
    forecourt(o, p, &pl);
    let club = o.club.clone();
    let vv = option(cat, fac, tier, opts, "view");
    let view = match vv.as_str() {
        Some(v @ ("closed" | "dollhouse" | "cutaway")) => v.to_string(),
        _ => "closed".to_string(),
    };
    let fv = option(cat, fac, tier, opts, "floor");
    let top = if view == "closed" {
        storeys - 1
    } else {
        (fv.as_f64().map(|v| v as i64).unwrap_or(storeys as i64 - 1)).max(0).min(storeys as i64 - 1) as usize
    };
    let sp = buildings::styled(cat, style, w, d, storeys, &club.primary, o.night, opts, p.seed);
    let pp = buildings::PLINTH;
    let (rooms, (yc0, yc1)) = plan(cat, &t, w, d, storeys);
    let mut mm = Mats::default();
    let q = tier.clamp(1, 4);
    let paint = PAINT[q - 1];
    let floor_tag = |mm: &mut Mats, kind: &str| -> i32 {
        let key = floor_key(kind);
        mm.tag(&format!("floor:{key}"), || mats::named(key))
    };
    // (an accent room has one feature wall in the club colour, its right-hand side wall seen from
    // the door; the rest painted)
    let finish_tag = |mm: &mut Mats, spec: &Value, feature: bool| -> i32 {
        let k = text(spec, "walls", "paint");
        if k == "tile" {
            return mm.tag("wall:tile", || mats::named("Floor_Tile"));
        }
        if k == "accent" && q >= 3 && feature {
            return mm.tag("wall:accent", || mats::tint("TINT_interior", &club.primary));
        }
        mm.tag("wall:paint", || mats::tint("TINT_interior", paint))
    };
    let room_floor = |mm: &mut Mats, spec: &Value| -> i32 {
        let mut k = text(spec, "floor", "vinyl");
        if let Some(tf) = spec.get("timber_from").and_then(|v| v.as_f64()) {
            if q as i64 >= tf as i64 {
                k = "timber";
            }
        }
        floor_tag(mm, k)
    };
    let t_fac = mm.tag("facade", || sp.mats[buildings::T_WALL as usize].clone());
    let t_base = mm.tag("base", || sp.mats[buildings::T_BASE as usize].clone());
    let t_sec = mm.tag("section", || mats::tint("TINT_trim", "#2f3236"));
    let t_trim = mm.tag("trim", || sp.mats[buildings::T_TRIM as usize].clone());
    let t_glass = mm.tag("glass", || mats::named("Balustrade_Glass"));
    let t_ceil = mm.tag("ceiling", || mats::named("Ceiling_Grid"));
    let night = o.night;
    let t_led = mm.tag("led", || mats::glow("TINT_trim", "#f6f8fa", if night { 3.0 } else { 0.9 }));
    let t_corr_f = floor_tag(&mut mm, if q >= 4 { "timber" } else { "vinyl" });
    let t_corr_w = mm.tag("wall:paint", || mats::tint("TINT_interior", paint));
    let t_door = mm.tag("door", || mats::tint("TINT_door", if q >= 3 { &club.primary } else { "#c9b48a" }));
    let t_lead = mm.tag("door:lead", || mats::tint("TINT_door", "#4a4d52"));
    let t_steel = mm.tag("steel", || mats::named("Galvanised"));
    let t_warn = mm.tag("warn", || mats::glow("TINT_trim", "#e0302a", 2.0));
    let t_water = mm.tag("water", || mats::named("Site_Water"));
    let t_tile = mm.tag("wall:tile", || mats::named("Floor_Tile"));
    let t_accent = mm.tag("accent", || sp.mats[buildings::T_ACCENT as usize].clone());
    let t_roof = mm.tag("roof", || mats::named("Roof_Flat"));
    let mut m = MeshAcc::new();
    let v3 = DVec3::new;
    let (xo0, xo1, yo0, yo1) = (-w / 2.0, w / 2.0, -d / 2.0, d / 2.0);
    let (xi0, xi1, yi0, yi1) = (xo0 + T_EXT, xo1 - T_EXT, yo0 + T_EXT, yo1 - T_EXT);
    let zf = |s: usize| pp + s as f64 * STOREY;
    let height = |s: usize| {
        if view == "closed" || s < top {
            STOREY
        } else if view == "dollhouse" {
            STOREY - 0.3
        } else {
            CUT
        }
    };
    let ent = rooms.iter().find(|r| r.storey == 0 && text(&r.spec, "glazing", "") == "entrance").cloned();
    let ycm = (yc0 + yc1) / 2.0;
    let outer = [
        (0usize, yo0, yo0 + T_EXT - 0.02, xo0, xo1),
        (0, yo1 - T_EXT + 0.02, yo1, xo0, xo1),
        (1, xo0, xo0 + T_EXT - 0.02, yo0 + T_EXT - 0.02, yo1 - T_EXT + 0.02),
        (1, xo1 - T_EXT + 0.02, xo1, yo0 + T_EXT - 0.02, yo1 - T_EXT + 0.02),
    ];
    for s in 0..=top {
        let (z0, h) = (zf(s), height(s));
        let zt = z0 + h;
        let mut ops: [Vec<Opening>; 4] = Default::default();
        for r in rooms.iter().filter(|r| r.storey == s) {
            let g = text(&r.spec, "glazing", "window");
            let side = if r.front { 0 } else { 1 };
            if g == "entrance" {
                ops[side].push(((r.x0 + r.x1) / 2.0 - 1.1, (r.x0 + r.x1) / 2.0 + 1.1, z0, z0 + 2.5));
                for k in [-1.0, 1.0] {
                    let a = (r.x0 + r.x1) / 2.0 + k * ((r.x1 - r.x0) / 2.0 - 1.4) - 0.8;
                    if (r.x1 - r.x0) >= 6.4 {
                        ops[side].push((a, a + 1.6, z0 + 0.1, z0 + 2.6));
                    }
                }
            } else if let Some((sill, head)) = glazing(g).filter(|_| r.depth().1 == 0.0) {
                let ww = (r.x1 - r.x0 - 0.9).min(7.2);
                let c = (r.x0 + r.x1) / 2.0;
                if ww > 0.6 {
                    ops[side].push((c - ww / 2.0, c + ww / 2.0, z0 + sill, z0 + head));
                }
            }
        }
        for side in [2usize, 3] {
            if s == 0 {
                ops[side].push((ycm - 0.6, ycm + 0.6, z0, z0 + 2.2));
            } else {
                ops[side].push((ycm - 0.6, ycm + 0.6, z0 + 0.9, z0 + 2.3));
            }
        }
        if s == 0 {
            for &(axis, c0, c1, a0, a1) in &outer {
                wall(&mut m, axis, c0, c1, a0, a1, 0.0, pp, &[], t_base, t_base);
            }
        }
        for (side, &(axis, c0, c1, a0, a1)) in outer.iter().enumerate() {
            wall(&mut m, axis, c0, c1, a0, a1, z0, zt, &ops[side], t_fac, t_sec);
            for &(a, b, oz0, oz1) in &ops[side] {
                let mid = (c0 + c1) / 2.0;
                if axis == 0 {
                    quad(&mut m, v3(a, mid, oz0), v3(b, mid, oz0), v3(b, mid, oz1.min(zt)), v3(a, mid, oz1.min(zt)), t_glass);
                    for (p0, p1, q0, q1) in [(a, a + 0.06, oz0, oz1.min(zt)), (b - 0.06, b, oz0, oz1.min(zt))] {
                        wbox(&mut m, p0, mid - 0.04, q0, p1, mid + 0.04, q1, t_trim, t_trim);
                    }
                    if oz1 <= zt {
                        wbox(&mut m, a, mid - 0.04, oz1 - 0.06, b, mid + 0.04, oz1, t_trim, t_trim);
                    }
                    if oz0 > z0 + 0.05 {
                        wbox(&mut m, a, mid - 0.04, oz0, b, mid + 0.04, oz0 + 0.06, t_trim, t_trim);
                    }
                    let n = ((b - a) / 1.5).floor() as usize;
                    for k in 1..=n {
                        let x = a + (b - a) * k as f64 / (n + 1) as f64;
                        wbox(&mut m, x - 0.03, mid - 0.04, oz0, x + 0.03, mid + 0.04, oz1.min(zt), t_trim, t_trim);
                    }
                } else {
                    quad(&mut m, v3(mid, a, oz0), v3(mid, b, oz0), v3(mid, b, oz1.min(zt)), v3(mid, a, oz1.min(zt)), t_glass);
                    for (p0, p1, q0, q1) in [(a, a + 0.06, oz0, oz1.min(zt)), (b - 0.06, b, oz0, oz1.min(zt))] {
                        wbox(&mut m, mid - 0.04, p0, q0, mid + 0.04, p1, q1, t_trim, t_trim);
                    }
                    if oz1 <= zt {
                        wbox(&mut m, mid - 0.04, a, oz1 - 0.06, mid + 0.04, b, oz1, t_trim, t_trim);
                    }
                }
            }
        }
        // linings: each room's finish on its stretch of the outside wall
        for r in rooms.iter().filter(|r| r.storey == s) {
            let ft = finish_tag(&mut mm, &r.spec, false);
            let side = if r.front { 0 } else { 1 };
            let (c0, c1) = if side == 0 { (yi0 - 0.02, yi0) } else { (yi1, yi1 + 0.02) };
            wall(&mut m, 0, c0, c1, r.x0, r.x1, z0, zt, &ops[side], ft, t_sec);
            if r.x0 <= xi0 + 1e-6 {
                let o2: Vec<Opening> = ops[2].iter().copied().filter(|o| o.1 > r.y0 && o.0 < r.y1).collect();
                let fx = finish_tag(&mut mm, &r.spec, r.front);
                wall(&mut m, 1, xi0 - 0.02, xi0, r.y0, r.y1, z0, zt, &o2, fx, t_sec);
            }
            if r.x1 >= xi1 - 1e-6 {
                let o3: Vec<Opening> = ops[3].iter().copied().filter(|o| o.1 > r.y0 && o.0 < r.y1).collect();
                let fx = finish_tag(&mut mm, &r.spec, !r.front);
                wall(&mut m, 1, xi1, xi1 + 0.02, r.y0, r.y1, z0, zt, &o3, fx, t_sec);
            }
        }
        for (x0, x1) in [(xi0 - 0.02, xi0), (xi1, xi1 + 0.02)] {
            let o = if x0 < 0.0 { ops[2].clone() } else { ops[3].clone() };
            wall(&mut m, 1, x0, x1, yc0, yc1, z0, zt, &o, t_corr_w, t_sec);
        }
        // partitions between rooms and the corridor walls
        for r in rooms.iter().filter(|r| r.storey == s) {
            if r.x1 >= xi1 - 1e-6 {
                continue;
            }
            let Some(nxt) = rooms.iter().find(|n| n.storey == s && n.front == r.front && (n.x0 - r.x1 - T_INT).abs() < 1e-6) else { continue };
            let (ya, yb) = if r.front { (r.y0 - 0.02, r.y1 + T_INT * 0.75) } else { (r.y0 - T_INT * 0.75, r.y1 + 0.02) };
            let fa = finish_tag(&mut mm, &r.spec, !r.front);
            wall(&mut m, 1, r.x1, r.x1 + T_INT / 2.0, ya, yb, z0, zt, &[], fa, t_sec);
            let fb = finish_tag(&mut mm, &nxt.spec, nxt.front);
            wall(&mut m, 1, r.x1 + T_INT / 2.0, nxt.x0, ya, yb, z0, zt, &[], fb, t_sec);
        }
        // cross walls closing deep rooms, mechanical plant behind (an air-handling unit, its duct)
        for r in rooms.iter().filter(|r| r.storey == s) {
            let (de, dp) = r.depth();
            if dp == 0.0 {
                continue;
            }
            let (yw0, yw1) = if r.front { (r.y1 - de - T_INT, r.y1 - de) } else { (r.y0 + de, r.y0 + de + T_INT) };
            let ft = finish_tag(&mut mm, &r.spec, false);
            wall(&mut m, 0, yw0, yw1, r.x0 - T_INT * 0.25, r.x1 + T_INT * 0.25, z0, zt, &[], ft, t_sec);
            let yp = if r.front { (r.y0 + yw0) / 2.0 } else { (yw1 + r.y1) / 2.0 };
            let xc = (r.x0 + r.x1) / 2.0;
            let aw = (r.x1 - r.x0 - 0.8).min(2.4);
            if aw > 0.8 && dp > 1.6 {
                bx(&mut m, xc - aw / 2.0, yp - 0.5, z0, xc + aw / 2.0, yp + 0.5, z0 + 1.8f64.min(zt - z0 - 0.05), t_steel);
                if zt - z0 > 2.2 {
                    bx(&mut m, xc - 0.3, yp - 0.3, z0 + 1.8, xc + 0.3, yp + 0.3, zt, t_steel);
                }
            }
        }
        for (front, ycw) in [(true, yc0), (false, yc1)] {
            let mut doors: Vec<Opening> = Vec::new();
            for r in rooms.iter().filter(|r| r.storey == s && r.front == front) {
                let (dw, _, u) = r.door();
                let x = r.u_to_x(u);
                if dw > 0.0 {
                    doors.push((x - dw / 2.0, x + dw / 2.0, z0, z0 + if dw < 2.0 { 2.1 } else { 2.6 }));
                }
            }
            for r in rooms.iter().filter(|r| r.storey == s && r.front == front) {
                let (c0, c1) = if front { (ycw - T_INT / 2.0, ycw) } else { (ycw, ycw + T_INT / 2.0) };
                let ft = finish_tag(&mut mm, &r.spec, false);
                let a0 = r.x0 - if r.x0 <= xi0 + 1e-6 { 0.01 } else { 0.0 };
                let a1 = r.x1 + if r.x1 >= xi1 - 1e-6 { 0.01 } else { 0.0 };
                wall(&mut m, 0, c0, c1, a0, a1, z0, zt, &doors, ft, t_sec);
            }
            let (c0, c1) = if front { (ycw, ycw + T_INT / 2.0) } else { (ycw - T_INT / 2.0, ycw) };
            wall(&mut m, 0, c0, c1, xi0 - 0.01, xi1 + 0.01, z0, zt, &doors, t_corr_w, t_sec);
        }
        // floors, the corridor, ceilings with LED panels
        let zfl = z0 + 0.002;
        for r in rooms.iter().filter(|r| r.storey == s) {
            let ft = room_floor(&mut mm, &r.spec);
            if r.pool() {
                continue;
            }
            if r.kind == "stairs" && s > 0 {
                let fx = if !r.front { r.x1 - 1.45 } else { r.x0 + 1.45 };
                let (xa, xb) = if !r.front { (r.x0, fx) } else { (fx, r.x1) };
                kernel::ground(&mut m, &kernel::rect(xb - xa, r.y1 - r.y0, (xa + xb) / 2.0, (r.y0 + r.y1) / 2.0), zfl, ft);
                wbox(&mut m, fx - 0.012, r.y0 + 1.4, z0, fx + 0.012, r.y1, z0 + 1.1, t_glass, t_trim);
                continue;
            }
            kernel::ground(&mut m, &kernel::rect(r.x1 - r.x0, r.y1 - r.y0, (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0), zfl, ft);
        }
        kernel::ground(&mut m, &kernel::rect(xi1 - xi0, yc1 - yc0, (xi0 + xi1) / 2.0, ycm), zfl, t_corr_f);
        if view == "closed" || s < top {
            let zc = z0 + CEIL;
            let mut areas: Vec<(f64, f64, f64, f64, bool)> = rooms.iter().filter(|r| r.storey == s && r.depth().1 == 0.0).map(|r| (r.x0, r.x1, r.y0, r.y1, r.kind == "stairs")).collect();
            for r in rooms.iter().filter(|r| r.storey == s && r.depth().1 > 0.0) {
                let de = r.depth().0;
                areas.push(if r.front { (r.x0, r.x1, r.y1 - de, r.y1, r.kind == "stairs") } else { (r.x0, r.x1, r.y0, r.y0 + de, r.kind == "stairs") });
            }
            areas.push((xi0, xi1, yc0, yc1, false));
            for (x0, x1, y0, y1, stairs) in areas {
                if stairs {
                    continue;
                }
                quad(&mut m, v3(x0, y0, zc), v3(x0, y1, zc), v3(x1, y1, zc), v3(x1, y0, zc), t_ceil);
                let (nx, ny) = ((((x1 - x0) / 2.4) as usize).max(1), (((y1 - y0) / 2.4) as usize).max(1));
                for i in 0..nx {
                    for j in 0..ny {
                        let cx = x0 + (x1 - x0) * (i as f64 + 0.5) / nx as f64;
                        let cy = y0 + (y1 - y0) * (j as f64 + 0.5) / ny as f64;
                        quad(&mut m, v3(cx - 0.3, cy - 0.3, zc - 0.01), v3(cx - 0.3, cy + 0.3, zc - 0.01), v3(cx + 0.3, cy + 0.3, zc - 0.01), v3(cx + 0.3, cy - 0.3, zc - 0.01), t_led);
                    }
                }
            }
        }
        // doors
        for r in rooms.iter().filter(|r| r.storey == s) {
            let (dw, dk, u) = r.door();
            if dw <= 0.0 || dk == "opening" {
                continue;
            }
            let (cx, cy, yaw) = r.frame_xy();
            let rf = Frame::rad(cx, cy, yaw);
            let mut dm = MeshAcc::new();
            let (ua, ub) = (u - dw / 2.0, u + dw / 2.0);
            // (clipped to the cut: nothing stands above the walls)
            let cbox = |dm: &mut MeshAcc, x0: f64, y0: f64, za: f64, x1: f64, y1: f64, zb: f64, tag: i32| {
                if za < zt - 1e-6 {
                    bx(dm, x0, y0, za, x1, y1, zb.min(zt), tag);
                }
            };
            for (p0, p1, q0, q1) in [(ua - 0.05, ua, 0.0, 2.15), (ub, ub + 0.05, 0.0, 2.15), (ua - 0.05, ub + 0.05, 2.1, 2.15)] {
                cbox(&mut dm, p0, -T_INT - 0.01, z0 + q0, p1, 0.01, z0 + q1, t_trim);
            }
            if dk == "leaf" {
                cbox(&mut dm, ua, 0.02, z0 + 0.01, ua + 0.045, 0.02 + dw - 0.02, z0 + 2.09, t_door);
                cbox(&mut dm, ua - 0.005, 0.25, z0 + 1.1, ua + 0.05, 0.35, z0 + 1.6, t_glass);
            } else {
                cbox(&mut dm, ua + 0.01, -T_INT / 2.0 - 0.025, z0 + 0.01, ub - 0.01, -T_INT / 2.0 + 0.025, z0 + 2.09, if dk == "lead" { t_lead } else { t_steel });
                if dk == "lead" {
                    cbox(&mut dm, u - 0.15, -T_INT - 0.06, z0 + 2.25, u + 0.15, -T_INT - 0.01, z0 + 2.4, t_warn);
                }
            }
            moved(&mut dm, &rf);
            m.append(&dm, 0);
        }
        // the pool
        for r in rooms.iter().filter(|r| r.storey == s && r.pool()) {
            let (cx, cy, yaw) = r.frame_xy();
            let rf = Frame::rad(cx, cy, yaw);
            let (ww, dd) = (r.x1 - r.x0, r.y1 - r.y0);
            let (pu0, pu1, pv0, pv1) = (-ww / 2.0 + 1.2, ww / 2.0 - 1.2, 1.8, dd - 1.0);
            let mut pm = MeshAcc::new();
            let ft = room_floor(&mut mm, &r.spec);
            for (a0, a1, b0, b1) in [(-ww / 2.0, ww / 2.0, 0.0, pv0), (-ww / 2.0, ww / 2.0, pv1, dd), (-ww / 2.0, pu0, pv0, pv1), (pu1, ww / 2.0, pv0, pv1)] {
                kernel::ground(&mut pm, &kernel::rect(a1 - a0, b1 - b0, (a0 + a1) / 2.0, (b0 + b1) / 2.0), zfl, ft);
            }
            let zb = z0 - 1.3;
            quad(&mut pm, v3(pu0, pv0, zb), v3(pu1, pv0, zb), v3(pu1, pv0, z0), v3(pu0, pv0, z0), t_tile);
            quad(&mut pm, v3(pu1, pv1, zb), v3(pu0, pv1, zb), v3(pu0, pv1, z0), v3(pu1, pv1, z0), t_tile);
            quad(&mut pm, v3(pu0, pv1, zb), v3(pu0, pv0, zb), v3(pu0, pv0, z0), v3(pu0, pv1, z0), t_tile);
            quad(&mut pm, v3(pu1, pv0, zb), v3(pu1, pv1, zb), v3(pu1, pv1, z0), v3(pu1, pv0, z0), t_tile);
            kernel::ground(&mut pm, &kernel::rect(pu1 - pu0, pv1 - pv0, (pu0 + pu1) / 2.0, (pv0 + pv1) / 2.0), zb, t_tile);
            for k in 0..4 {
                let k = k as f64;
                bx(&mut pm, pu1 - 1.2, pv0 + k * 0.3, zb, pu1 - 0.01, pv0 + (k + 1.0) * 0.3, z0 - 0.3 - k * 0.25, t_tile);
            }
            kernel::ground(&mut pm, &kernel::rect(pu1 - pu0, pv1 - pv0, (pu0 + pu1) / 2.0, (pv0 + pv1) / 2.0), z0 - 0.12, t_water);
            moved(&mut pm, &rf);
            m.append(&pm, 0);
            o.prop_at(&f.sub(cx, cy, yaw), "prop.pool_hoist", pu0 + 0.5, pv0 - 0.35, z0, 0.0, &[]);
        }
        // stairs
        for r in rooms.iter().filter(|r| r.storey == s && r.kind == "stairs" && s + 1 < storeys) {
            let (cx, cy, yaw) = r.frame_xy();
            let rf = Frame::rad(cx, cy, yaw);
            let ww = r.x1 - r.x0;
            let mut sm = MeshAcc::new();
            let n = 20;
            let (rise, going) = (STOREY / n as f64, 0.25);
            let (ua, ub) = (ww / 2.0 - 1.35, ww / 2.0 - 0.05);
            let tread = room_floor(&mut mm, &r.spec);
            for k in 0..n {
                let k = k as f64;
                bx(&mut sm, ua, 0.4 + k * going, z0, ub, 0.4 + (k + 1.0) * going, z0 + (k + 1.0) * rise, tread);
            }
            moved(&mut sm, &rf);
            m.append(&sm, 0);
            let rail: Polyline = (vec![v3(ua - 0.04, 0.4, z0 + 0.9), v3(ua - 0.04, 0.4 + n as f64 * going, z0 + STOREY + 0.9)], 0.025);
            o.tubes_at(&f.sub(cx, cy, yaw), &format!("{}:stair_rail{s}", p.id), vec![rail], mats::named("Galvanised"), 8);
        }
        for r in rooms.iter().filter(|r| r.storey == s) {
            furnish(o, &f, r, q, z0, &club.primary, view == "cutaway" && s == top);
        }
        for (i, r) in rooms.iter().enumerate() {
            if r.storey != s {
                continue;
            }
            let de = r.depth().0;
            let yc = if r.front { r.y1 - de / 2.0 } else { r.y0 + de / 2.0 };
            o.socket_at(&f, "room", &format!("{}:{}:{i}", p.id, r.kind), (r.x0 + r.x1) / 2.0, yc, z0, DVec2::new(0.0, 1.0), vec![("width", r.x1 - r.x0), ("depth", de), ("storey", s as f64)]);
        }
    }
    if view == "closed" {
        let zr = zf(storeys);
        quad(&mut m, v3(xo0, yo0, zr), v3(xo1, yo0, zr), v3(xo1, yo1, zr), v3(xo0, yo1, zr), t_roof);
        for (axis, c0, c1, a0, a1) in [(0usize, yo0, yo0 + 0.25, xo0, xo1), (0, yo1 - 0.25, yo1, xo0, xo1), (1, xo0, xo0 + 0.25, yo0 + 0.25, yo1 - 0.25), (1, xo1 - 0.25, xo1, yo0 + 0.25, yo1 - 0.25)] {
            wall(&mut m, axis, c0, c1, a0, a1, zr, zr + 0.9, &[], t_fac, t_trim);
        }
        for k in 0..((w / 12.0).max(1.0) as usize) {
            let x = xo0 + 4.0 + k as f64 * 12.0;
            bx(&mut m, x, yo1 - 4.0, zr, x + 2.4, yo1 - 2.6, zr + 1.3, t_steel);
        }
    }
    let ent_x = ent.as_ref().map(|e| (e.x0 + e.x1) / 2.0);
    if let Some(ent) = ent {
        let ex = (ent.x0 + ent.x1) / 2.0;
        // (the canopy and the name only where the ground floor's front is standing to their height)
        let front_up = view == "closed" || top > 0;
        if front_up || view == "dollhouse" {
            bx(&mut m, ex - 2.4, yo0 - 2.6, pp + 2.55, ex + 2.4, yo0, pp + 2.8, t_trim);
        }
        for k in 0..3 {
            let k = k as f64;
            bx(&mut m, ex - 1.6, yo0 - 1.0 + k * 0.33, 0.0, ex + 1.6, yo0 - 0.67 + k * 0.33, (k + 1.0) * 0.15, t_base);
        }
        let (ra, rb) = (ex + 1.6, ex + 1.6 + pp * 15.0);
        quad(&mut m, v3(rb, yo0 - 1.6, 0.0), v3(rb, yo0 - 0.1, 0.0), v3(ra, yo0 - 0.1, pp), v3(ra, yo0 - 1.6, pp), t_base);
        quad(&mut m, v3(rb, yo0 - 1.6, 0.0), v3(ra, yo0 - 1.6, pp), v3(ra, yo0 - 1.6, 0.0), v3(rb, yo0 - 1.6, 0.0), t_base);
        let mut rails: Vec<Polyline> = vec![(vec![v3(rb, yo0 - 1.62, 0.9), v3(ra, yo0 - 1.62, pp + 0.9), v3(ra - 0.3, yo0 - 1.62, pp + 0.9)], 0.022)];
        for x in [ra, (ra + rb) / 2.0, rb] {
            let zz = pp * (rb - x) / (rb - ra);
            rails.push((vec![v3(x, yo0 - 1.62, zz), v3(x, yo0 - 1.62, zz + 0.9)], 0.022));
        }
        o.tubes_at(&f, &format!("{}:ramp_rail", p.id), rails, mats::named("Galvanised"), 8);
        let sign = text(&t, "sign", "");
        let name = if sign.is_empty() { club.name.clone() } else { format!("{} {}", club.name, sign) };
        if view == "closed" {
            // (over the canopy on one storey; on two, high on the front above the first-floor windows;
            // not in the open views, where the wall it is fixed to is cut or gone)
            let (zs, hs) = if storeys == 1 { (pp + 3.0, 0.42) } else { (zf(storeys) - 0.72, 0.5) };
            buildings::letters(&mut m, w, d, 0, &name, w / 2.0 + ex, zs, (w * 0.5).min(16.0), t_accent, hs / 7.0, 0.0);
        }
        if t.get("ambulance_bay").and_then(|v| v.as_bool()) == Some(true) {
            let mut pm = MeshAcc::new();
            let (bx0, by0) = (ex + 5.0, yo0 - 12.0);
            let r = kernel::rect(3.6, 8.0, bx0 + 1.8, by0 + 4.0);
            strip(&mut pm, &[r[0], r[1], r[2], r[3], r[0]], 0.15, COVER_Z + 0.004, 0, None);
            for k in 0..6 {
                let y = by0 + 0.6 + k as f64 * 1.3;
                strip(&mut pm, &[DVec2::new(bx0 + 0.2, y), DVec2::new(bx0 + 3.4, y + 1.0)], 0.12, COVER_Z + 0.004, 0, None);
            }
            o.part_at(&f, &format!("{}:ambulance_bay", p.id), pm, vec![mats::tint("TINT_road_paint", "#e3b92a")], 0.0);
        }
    }
    let bay = t.get("ambulance_bay").and_then(|v| v.as_bool()) == Some(true);
    let statue = Some(text(&t, "statue", "")).filter(|s| !s.is_empty());
    frontage(o, &f, w, d, pl.gap, ent_x, bay, q, &club.primary, &club.secondary, statue);
    o.part_at(&f, &format!("{}:building", p.id), m, mm.list, 0.0);
}

/// Street furniture on the forecourt, kept off the entrance steps, ramp and ambulance bay
/// (kit/site/interior.py frontage).
#[allow(clippy::too_many_arguments)]
fn frontage(o: &mut Out, f: &Frame, w: f64, d: f64, gap: f64, ex: Option<f64>, bay: bool, q: usize, primary: &str, secondary: &str, statue: Option<&str>) {
    let (yf, ye) = (-d / 2.0, -d / 2.0 - gap + 0.5);
    let mut clear: Vec<(f64, f64)> = Vec::new();
    if let Some(ex) = ex {
        clear.push((ex - 2.9, ex + 1.6 + buildings::PLINTH * 15.0 + 0.4));
        if bay {
            clear.push((ex + 4.6, ex + 9.0));
        }
    }
    let free = |x: f64, half: f64| clear.iter().all(|&(a, b)| x + half < a || x - half > b);
    let mut n = 0;
    let mut x = -w / 2.0 + 2.0;
    while x <= w / 2.0 - 2.0 + 1e-9 {
        if free(x, 1.0) {
            o.prop_at(f, "prop.planter", x, yf - 0.9, COVER_Z, 0.0, &[]);
            if q >= 2 && n % 2 == 1 && yf - 2.0 > ye + 1.0 {
                o.prop_at(f, "prop.bench", x, yf - 2.1, COVER_Z, PI, &[]);
            }
            n += 1;
        }
        x += 6.0;
    }
    o.prop_at(f, "prop.bin_litter", ex.map(|e| e - 3.4).unwrap_or(0.0), yf - 1.6, COVER_Z, 0.0, &[("TINT_bin", "#2a2c2e")]);
    let rx = if free(w / 2.0 - 3.5, 2.6) { w / 2.0 - 3.5 } else { -w / 2.0 + 3.5 };
    o.prop_at(f, "prop.bike_rack", rx, yf - 3.6, COVER_Z, 0.0, &[]);
    o.prop_at(f, "genprop.bicycle", rx - 1.85, yf - 3.6, COVER_Z, FRAC_PI_2, &[]);
    let bx = ex.map(|e| e - 3.4).unwrap_or(0.0);
    if q == 1 {
        o.prop_at(f, "genprop.leaf_litter", bx + 1.2, yf - 2.3, COVER_Z, 0.6, &[]);
    } else {
        o.prop_at(f, "genprop.pigeon_pecking", bx - 1.6, yf - 2.6, COVER_Z, 1.3, &[]);
        if q >= 3 {
            o.prop_at(f, "genprop.pigeon_standing", bx - 2.4, yf - 3.1, COVER_Z, 4.0, &[]);
        }
    }
    if let (Some(key), Some(ex)) = (statue, ex) {
        if free(ex - 6.5, 1.2) && (ex - 6.5 - rx).abs() > 3.8 {
            o.prop_at(f, key, ex - 6.5, yf - 5.0, COVER_Z, PI, &[]);
        }
    }
    if let Some(ex) = ex {
        if q >= 2 {
            for k in 0..3 {
                let y = yf - 3.0 - 4.0 * k as f64;
                if y > ye + 0.5 {
                    for s in [-1.0, 1.0] {
                        o.prop_at(f, "prop.path_light", ex + s * 2.2, y, COVER_Z, 0.0, &[]);
                    }
                }
            }
        }
        if q >= 3 {
            for k in -3..4 {
                o.prop_at(f, "prop.bollard", ex + k as f64 * 1.6, ye + 0.4, COVER_Z, 0.0, &[]);
            }
        }
    }
    if q >= 3 {
        let side = if rx > 0.0 { -1.0 } else { 1.0 };
        for (i, s) in [-1.0, 0.0, 1.0].iter().enumerate() {
            let col = if i == 1 { secondary } else { primary };
            o.prop_at(f, "prop.flagpole", side * (w / 2.0 - 4.5) + s * 2.5, ye + 2.5, COVER_Z, 0.0, &[("TINT_flag", col)]);
        }
    }
}

/// Lay out a room's fixtures (catalog/rooms.json) in its frame.
fn furnish(o: &mut Out, f: &Frame, r: &Room, q: usize, z0: f64, club: &str, cut: bool) {
    let (cx, cy, yaw) = r.frame_xy();
    let rf = f.sub(cx, cy, yaw);
    let (ww, dd) = (r.x1 - r.x0, r.depth().0);
    let Some(fixtures) = r.spec.get("fixtures").and_then(|v| v.as_array()) else { return };
    for fx in fixtures {
        let Some(key) = fx.get("key").and_then(|v| v.as_str()) else { continue };
        if num(fx, "min_w", 0.0) > ww + 1e-9 || num(fx, "min_d", 0.0) > dd + 1e-9 || num(fx, "min_tier", 1.0) > q as f64 || num(fx, "max_tier", 9.0) < q as f64 || num(fx, "max_w", 999.0) < ww - 1e-9 {
            continue;
        }
        if cut && fx.get("cut").and_then(|v| v.as_bool()) == Some(false) {
            continue;
        }
        let at = text(fx, "at", "centre");
        let off = num(fx, "off", 0.0);
        let rot = num(fx, "rot", 0.0).to_radians();
        let mut tints: Vec<(String, String)> = Vec::new();
        if let Some(tv) = fx.get("tints").and_then(|v| v.as_object()) {
            let mut slots: Vec<&String> = tv.keys().collect();
            slots.sort();
            for slot in slots {
                if let Some(c) = tv[slot].as_str() {
                    tints.push((slot.clone(), if c == "club" { club.to_string() } else { c.to_string() }));
                }
            }
        }
        let tr: Vec<(&str, &str)> = tints.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        let fracs: Vec<Option<f64>> = if at == "centre" {
            vec![None]
        } else if let Some(rep) = fx.get("repeat").and_then(|v| v.as_f64()) {
            let span = fx.get("span").and_then(|v| v.as_array()).filter(|a| a.len() == 2);
            let (s0, s1) = span.map(|a| (a[0].as_f64().unwrap_or(0.0), a[1].as_f64().unwrap_or(1.0))).unwrap_or((0.0, 1.0));
            let ln = (s1 - s0) * if at == "back" || at == "door" { ww } else { dd };
            let n = ((ln / rep).floor() as i64).min(num(fx, "max", 99.0) as i64).max(0) as usize;
            (0..n).map(|k| Some(s0 + (s1 - s0) * (k as f64 + 0.5) / n as f64)).collect()
        } else {
            vec![Some(num(fx, "along", 0.5))]
        };
        for a in fracs {
            let (u, v, y) = match (at, a) {
                ("back", Some(a)) => ((a - 0.5) * ww, dd - off, PI),
                ("door", Some(a)) => ((a - 0.5) * ww, off, 0.0),
                ("left", Some(a)) => (-ww / 2.0 + off, a * dd, -FRAC_PI_2),
                ("right", Some(a)) => (ww / 2.0 - off, a * dd, FRAC_PI_2),
                _ => ((num(fx, "fu", 0.5) - 0.5) * ww, num(fx, "fv", 0.5) * dd, 0.0),
            };
            o.prop_at(&rf, key, u, v, z0 + num(fx, "z", 0.0), y + rot, &tr);
        }
    }
}
