//! Facilities by tier (port of kit/site/facilities.py): training pitches (rough grass to match
//! replica, with ball-stop nets, floodlights, shelters, a filming tower and a viewing deck),
//! artificial pitches, goalkeeper areas, the training building (portacabins to a flagship), indoor
//! halls, recovery, car parks and grounds maintenance.

use super::buildings::{self, Spec};
use super::kernel::{self, bx, cylinder, obox, quad, strip, Frame};
use super::occupants::{bays, cover, mound, park, park_dir};
use super::{landscape, mats, option, option_bool, tier_spec, Out, Plot, SitePitch, COVER_Z, LAWN_Z};
use crate::catalog::{num, text, Catalog};
use crate::floodlights;
use crate::mesh::MeshAcc;
use crate::prng::uniform;
use crate::stadium::Placement;
use crate::stands::Polyline;
use glam::{DVec2, DVec3};
use serde_json::{json, Value};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

pub fn build(o: &mut Out, cat: &Catalog, p: &Plot, fac: &str, tier: usize, style: &str) {
    let opts = p.cfg.get("options").cloned().unwrap_or(json!({}));
    match fac {
        "pitch" => pitch(o, cat, p, tier, &opts),
        "youth_pitches" => youth_pitches(o, cat, p, tier, &opts),
        "artificial_pitch" => artificial(o, cat, p, tier, &opts),
        "cages" => cages(o, cat, p, tier, &opts),
        "showpiece_pitch" => showpiece(o, cat, p, tier, &opts),
        "goalkeeper_area" => goalkeeper(o, cat, p, tier),
        "training_building" | "academy_building" => training_building(o, cat, p, tier, style, &opts, fac),
        "indoor_hall" => hall(o, cat, p, tier, style, &opts),
        "recovery" => recovery(o, cat, p, tier, style),
        "car_park" => car_park(o, cat, p, tier, &opts),
        "drop_off" => drop_off(o, cat, p, tier, &opts),
        "medical_centre" | "scouting_centre" | "commercial_centre" | "media_centre" => super::interior::build(o, cat, p, tier, style, &opts, fac),
        "maintenance" => maintenance(o, cat, p, tier, style),
        _ => {}
    }
}

// -- pitches ---------------------------------------------------------------------------------------

/// Pitch size within the plot (length along its x, width along its y).
pub fn pitch_dims(p: &Plot, length: f64, width: f64, runoff_end: f64, runoff_side: f64) -> (f64, f64) {
    (length.min(p.w - 2.0 * runoff_end - 10.0), width.min(p.d - 2.0 * runoff_side - 16.0))
}

#[allow(clippy::too_many_arguments)]
fn pitch_params(p: &Plot, l: f64, w: f64, re: f64, rs: f64, grass: &str, height: f64, wear: f64, wet: f64, mow: &str, hand: f64, crown: f64, markings: &str) -> Value {
    json!({
        "length": l, "width": w, "runoff_end": re, "runoff_side": rs, "grass": grass, "height_mm": height,
        "wear": wear, "wetness": wet, "mow": {"pattern": mow, "width": 5.25, "angle": 0.0, "offset": 0.0},
        "hand_painted": hand, "crown": crown, "line_color": "#f5f5f0", "optional_marks": false,
        "markings": markings, "seed": (p.seed % 997) as i64 + 1
    })
}

/// Floodlight poles along both touchlines, designed to a lux target (the stadium's pole design).
#[allow(clippy::too_many_arguments)]
fn pole_lights(o: &mut Out, f: &Frame, name: &str, l: f64, w: f64, runoff_side: f64, per_side: usize, height: f64, lamps: usize, lux: f64) {
    pole_lights_behind(o, f, name, l, w, runoff_side, per_side, height, lamps, lux, 0.0);
}

/// As pole_lights, the +y side's poles moved behind_n further back (behind a stand), still aimed
/// at the same spots on the pitch.
#[allow(clippy::too_many_arguments)]
fn pole_lights_behind(o: &mut Out, f: &Frame, name: &str, l: f64, w: f64, runoff_side: f64, per_side: usize, height: f64, lamps: usize, lux: f64, behind_n: f64) {
    let spec = json!({"type": "poles", "per_side": per_side, "height": height, "lamps": lamps, "lux": lux, "fixture": "led", "kelvin": 5000});
    let Ok(d) = floodlights::design(&spec, l, w, runoff_side, None) else { return };
    let back = |p: DVec3, y: f64| DVec3::new(p.x, p.y + if y > 0.0 { behind_n } else { 0.0 }, p.z);
    for (pos, t) in &d.fixtures {
        let pos = &back(*pos, pos.y);
        let (a, b) = (f.pt(pos.x, pos.y), f.pt(t.x, t.y));
        let (pa, pb) = (DVec3::new(a.x, a.y, pos.z), DVec3::new(b.x, b.y, t.z));
        let mount = floodlights::mount_point(pa, pb, floodlights::fixture_pivot(&d.fixture));
        o.prop(&format!("floodlight_fixture.{}", d.fixture), Placement::aim(mount, pb - pa), &[]);
        if o.night {
            let dir = (pb - pa).normalize();
            o.spot(pa + dir * 0.3, dir, d.power, d.beam, floodlights::kelvin_rgb(d.kelvin));
        }
    }
    let masts: Vec<Polyline> = d.tube_masts.iter().map(|(a, b, r)| (vec![back(*a, a.y), back(*b, a.y)], *r)).collect();
    o.tubes_at(f, &format!("{name}:poles"), masts, mats::named("Galvanised"), 12);
}

/// Ball-stop netting behind a goal line: posts every 4 m and a net, `x` metres from the centre.
#[allow(clippy::too_many_arguments)]
fn ball_stop(o: &mut Out, f: &Frame, name: &str, x: f64, half: f64, h: f64) {
    let mut net = MeshAcc::new();
    let mut posts: Vec<Polyline> = Vec::new();
    for s in [-1.0, 1.0] {
        let xx = s * x;
        quad(&mut net, DVec3::new(xx, -half, LAWN_Z + 0.05), DVec3::new(xx, half, LAWN_Z + 0.05), DVec3::new(xx, half, h), DVec3::new(xx, -half, h), 0);
        let n = (2.0 * half / 4.0).round().max(1.0) as usize;
        for j in 0..=n {
            let y = -half + 2.0 * half * j as f64 / n as f64;
            posts.push((vec![DVec3::new(xx, y, LAWN_Z), DVec3::new(xx, y, h + 0.1)], 0.045));
        }
        posts.push((vec![DVec3::new(xx, -half, h), DVec3::new(xx, half, h)], 0.02));
    }
    o.part_at(f, &format!("{name}:ballstop"), net, vec![mats::ball_stop()], 0.0);
    o.tubes_at(f, &format!("{name}:ballstop_posts"), posts, mats::named("Galvanised"), 8);
}

/// A scaffold filming tower with a covered platform at `top` metres, front towards +y.
fn filming_tower(o: &mut Out, f: &Frame, name: &str, top: f64) {
    let mut t: Vec<Polyline> = Vec::new();
    let a = 1.3;
    let corners = [DVec2::new(-a, -a), DVec2::new(a, -a), DVec2::new(a, a), DVec2::new(-a, a)];
    for c in corners {
        t.push((vec![DVec3::new(c.x, c.y, LAWN_Z), DVec3::new(c.x, c.y, top + 2.3)], 0.03));
    }
    let mut z = LAWN_Z + 0.3;
    let mut k = 0;
    while z <= top + 0.01 {
        let mut ring: Vec<DVec3> = corners.iter().map(|c| DVec3::new(c.x, c.y, z)).collect();
        ring.push(ring[0]);
        t.push((ring, 0.024));
        if z + 2.0 <= top + 0.01 {
            let (c0, c1) = (corners[k % 4], corners[(k + 1) % 4]);
            t.push((vec![DVec3::new(c0.x, c0.y, z), DVec3::new(c1.x, c1.y, z + 2.0)], 0.022));
        }
        z += 2.0;
        k += 1;
    }
    // ladder at the back
    for x in [-0.25, 0.25] {
        t.push((vec![DVec3::new(x, a + 0.1, LAWN_Z), DVec3::new(x, a + 0.1, top + 1.0)], 0.02));
    }
    let mut zz = LAWN_Z + 0.3;
    while zz < top + 1.0 {
        t.push((vec![DVec3::new(-0.25, a + 0.1, zz), DVec3::new(0.25, a + 0.1, zz)], 0.015));
        zz += 0.3;
    }
    let mut m = MeshAcc::new();
    bx(&mut m, -a - 0.2, -a - 0.2, top - 0.08, a + 0.2, a + 0.2, top + 0.04, 0);
    bx(&mut m, -a - 0.3, -a - 0.3, top + 2.3, a + 0.3, a + 0.3, top + 2.42, 1);
    // a canvas skirt round the rail
    for (x0, y0, x1, y1) in [(-a, -a - 0.2, a, -a - 0.15), (-a - 0.2, -a, -a - 0.15, a), (a + 0.15, -a, a + 0.2, a)] {
        bx(&mut m, x0, y0, top + 0.04, x1, y1, top + 1.05, 2);
    }
    o.tubes_at(f, &format!("{name}:tower"), t, mats::named("Galvanised"), 6);
    o.part_at(f, &format!("{name}:tower_deck"), m, vec![mats::named("Site_Plywood"), mats::tint("TINT_trim", "#2b2d30"), mats::tint("TINT_club", &o.club.primary.clone())], 0.0);
}

/// A raised viewing deck along +y of frame f: steel posts, a slab, glass balustrade and a canopy.
fn viewing_deck(o: &mut Out, f: &Frame, name: &str, len: f64, roof_mat: crate::stadium::Mat) {
    let (dep, z) = (4.5, 2.8);
    let mut m = MeshAcc::new();
    bx(&mut m, -len / 2.0, -dep / 2.0, z, len / 2.0, dep / 2.0, z + 0.3, 0);
    let n = (len / 4.0).round() as usize;
    for j in 0..=n {
        let x = -len / 2.0 + 0.2 + (len - 0.4) * j as f64 / n as f64;
        for y in [-dep / 2.0 + 0.2, dep / 2.0 - 0.2] {
            bx(&mut m, x - 0.12, y - 0.12, LAWN_Z, x + 0.12, y + 0.12, z, 1);
            bx(&mut m, x - 0.08, y - 0.08, z + 0.3, x + 0.08, y + 0.08, z + 3.2, 1);
        }
    }
    // stairs at the west end
    for s in 0..14 {
        let x0 = -len / 2.0 - 0.3 * (s + 1) as f64;
        bx(&mut m, x0, -dep / 2.0, LAWN_Z, x0 + 0.3, -dep / 2.0 + 1.4, z + 0.3 - 0.2 * s as f64, 0);
    }
    // glass balustrade on the pitch side, canopy
    quad(&mut m, DVec3::new(-len / 2.0, -dep / 2.0, z + 0.3), DVec3::new(len / 2.0, -dep / 2.0, z + 0.3), DVec3::new(len / 2.0, -dep / 2.0, z + 1.4), DVec3::new(-len / 2.0, -dep / 2.0, z + 1.4), 2);
    bx(&mut m, -len / 2.0 - 0.3, -dep / 2.0 - 0.6, z + 3.2, len / 2.0 + 0.3, dep / 2.0 + 0.3, z + 3.45, 3);
    o.part_at(f, &format!("{name}:deck"), m, vec![mats::named("Facade_Concrete"), mats::tint("TINT_trim", "#2b2d30"), mats::named("Balustrade_Glass"), roof_mat], 0.0);
}

const DISC_PAIRS: [(&str, &str); 5] = [("#ffd21f", "#2c7be5"), ("#ff6a1a", "#f4f4f2"), ("#e53935", "#ffd21f"), ("#2c7be5", "#f4f4f2"), ("#7cd43a", "#ff6a1a")];
const KIT_COLOURS: [&str; 4] = ["#ff6a1a", "#ffd21f", "#2c7be5", "#e53935"];

/// Training gear set out for a session (pitch-local, x along the length). Every pitch has its own
/// session: which half and side it uses, the cone colours, the size of the marker grid, and which
/// drills are out (a slalom, a mannequin wall, a pole and ladder station, mini goals or a rondo
/// circle), with the balls scattered.
fn drills(o: &mut Out, f: &Frame, l: f64, w: f64, seed: u64) {
    let u = |k: u64| uniform(seed + 10 + k, 0);
    let sx = if u(0) < 0.5 { 1.0 } else { -1.0 };
    let sy = if u(1) < 0.5 { 1.0 } else { -1.0 };
    let discs = DISC_PAIRS[(u(2) * DISC_PAIRS.len() as f64) as usize % DISC_PAIRS.len()];
    let tall = KIT_COLOURS[(u(3) * KIT_COLOURS.len() as f64) as usize % KIT_COLOURS.len()];
    let pole = KIT_COLOURS[(u(4) * KIT_COLOURS.len() as f64) as usize % KIT_COLOURS.len()];
    let mut at = |key: &str, x: f64, y: f64, yaw: f64, tints: &[(&str, &str)]| o.prop_at(f, key, sx * x, sy * y, LAWN_Z, yaw, tints);
    let j = |i: u64| (uniform(seed, i) - 0.5) * 0.3;
    if u(5) < 0.8 {
        for i in 0..8u64 {
            at("prop.cone_tall", -l / 4.0 - 12.0 + i as f64 * 3.0 + j(i), -w / 4.0 + if i % 2 == 0 { 0.8 } else { -0.8 }, 0.0, &[("TINT_cone", tall)]);
        }
    }
    let n = if u(6) < 0.4 { 3u64 } else { 4 };
    let step = 5.0 + 3.0 * u(7);
    for a in 0..n {
        for b in 0..n {
            let c = if (a + b) % 2 == 0 { discs.0 } else { discs.1 };
            at("prop.cone_disc", -l / 4.0 + a as f64 * step, w / 8.0 + b as f64 * step, 0.0, &[("TINT_cone", c)]);
        }
    }
    if u(8) < 0.7 {
        for i in 0..5u64 {
            at("prop.mannequin", l / 2.0 - 20.0, -1.2 + i as f64 * 0.6, -FRAC_PI_2 * sx, &[("TINT_mannequin", "#f4d21c")]);
        }
    }
    if u(9) < 0.75 {
        for i in 0..6u64 {
            at("prop.agility_pole", -8.0 + i as f64 * 1.8, w / 2.0 - 8.0 + if i % 2 == 0 { 0.6 } else { -0.6 }, 0.0, &[("TINT_pole_train", pole)]);
        }
        at("prop.agility_ladder", 6.0, w / 2.0 - 6.0, FRAC_PI_2, &[]);
    }
    if u(10) < 0.6 {
        for s2 in [-1.0, 1.0] {
            at("prop.mini_goal", -l / 2.0 + 16.0, s2 * 14.0, -s2 * sx * FRAC_PI_2, &[]);
        }
    } else {
        for i in 0..10u64 {
            let a = i as f64 / 10.0 * 2.0 * PI;
            at("prop.cone_disc", -l / 2.0 + 18.0 + 7.0 * a.cos(), 7.0 * a.sin(), 0.0, &[("TINT_cone", discs.0)]);
        }
    }
    at("prop.ball_cart", 2.0, -w / 2.0 - 1.8, 0.3, &[]);
    at("prop.rebounder", -4.0, -w / 2.0 + 2.5, if sy > 0.0 { 0.0 } else { PI }, &[]);
    // the touchline station: a bag of balls, bibs, water carriers, the coaches' board, a kit bag
    if u(12) < 0.6 {
        at("genprop.ball_bag", 3.4, -w / 2.0 - 1.6, 0.6, &[]);
    }
    at("genprop.bibs_stack", 0.6, -w / 2.0 - 1.5, 0.2, &[]);
    for i in 0..2 {
        at("genprop.water_carrier", -1.0 - i as f64 * 0.5, -w / 2.0 - 1.4, 0.4 * i as f64, &[]);
    }
    if u(13) < 0.5 {
        at("genprop.tactics_easel", -3.0, -w / 2.0 - 2.2, if sy > 0.0 { 0.0 } else { PI }, &[]);
    }
    at("genprop.kit_bag", 5.0, -w / 2.0 - 1.5, 1.2, &[]);
    if u(14) < 0.5 {
        at("genprop.gk_gloves", l / 2.0 - 2.0, 1.5, 0.8, &[]);
    }
    // (one ball in three a muddy one)
    for i in 0..(8 + (u(11) * 9.0) as u64) {
        let (x, y) = ((uniform(seed + 3, i) - 0.5) * l * 0.7, (uniform(seed + 4, i) - 0.5) * w * 0.7);
        at(if i % 3 == 0 { "genprop.ball_muddy" } else { "prop.ball" }, x, y, uniform(seed + 5, i) * TAU, &[("TINT_ball", "#f4f4f2")]);
    }
    // gulls on the grass
    if u(15) < 0.35 {
        for i in 0..(1 + (u(16) * 3.0) as u64) {
            at("genprop.seagull_standing", (uniform(seed + 27, i) - 0.5) * l * 0.6, (uniform(seed + 28, i) - 0.5) * w * 0.6, uniform(seed + 29, i) * TAU, &[]);
        }
    }
}

fn pitch(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize, opts: &Value) {
    let t = tier_spec(cat, "pitch", tier).cloned().unwrap_or(Value::Null);
    let (re, rs) = (4.0, 3.0);
    let (l, w) = pitch_dims(p, option(cat, "pitch", tier, opts, "length").as_f64().unwrap_or(105.0), option(cat, "pitch", tier, opts, "width").as_f64().unwrap_or(68.0), re, rs);
    let mow = option(cat, "pitch", tier, opts, "mow").as_str().map(String::from).unwrap_or_else(|| text(&t, "mow", "stripes").to_string());
    let grass = text(&t, "grass", "perennial_ryegrass").to_string();
    let params = pitch_params(p, l, w, re, rs, &grass, num(&t, "height_mm", 30.0), num(&t, "wear", 0.4), num(&t, "wetness", 0.0), &mow, num(&t, "hand_painted", 0.0), num(&t, "crown", 0.12), "full");
    let lit = option_bool(cat, "pitch", tier, opts, "floodlights");
    o.pitches.push(SitePitch {
        plot: p.id.clone(),
        params,
        origin: p.frame.pt(0.0, 0.0),
        yaw: p.frame.yaw,
        goals: Some(text(&t, "goals", "classic").to_string()),
        both_goals: true,
        flags: Some(text(&t, "flags", "pro_spring").to_string()),
        lit,
        surface: if grass.starts_with("hybrid") { "hybrid".into() } else { "grass".into() },
    });
    let f = p.frame;
    if option_bool(cat, "pitch", tier, opts, "ball_stop") {
        ball_stop(o, &f, &p.id, l / 2.0 + re + 2.0, (w / 2.0 + 8.0).min(p.d / 2.0 - 2.0), if tier >= 3 { 8.0 } else { 6.0 });
    }
    if lit {
        let (per, h, lamps, lux) = match tier {
            1 | 2 => (3, 15.0, 2, 200.0),
            3 => (4, 18.0, 3, 400.0),
            _ => (4, 20.0, 4, 600.0),
        };
        pole_lights(o, &f, &p.id, l, w, rs, per, h, lamps, lux);
    }
    if option_bool(cat, "pitch", tier, opts, "shelters") {
        for s in [-1.0, 1.0] {
            o.prop_at(&f, "prop.team_shelter", s * 9.0, -(w / 2.0 + rs + 1.6), LAWN_Z, 0.0, &[("TINT_club", &o.club.primary.clone())]);
        }
    }
    if option_bool(cat, "pitch", tier, opts, "filming_tower") {
        let tf = f.sub(0.0, -(w / 2.0 + rs + 6.0).min(p.d / 2.0 - 2.0), 0.0);
        filming_tower(o, &tf, &p.id, 8.0);
    }
    if option_bool(cat, "pitch", tier, opts, "viewing_deck") {
        let df = f.sub(0.0, (w / 2.0 + rs + 4.8).min(p.d / 2.0 - 2.4), PI);
        viewing_deck(o, &df, &p.id, 24.0, mats::named("Roof_Flat"));
    }
    if option_bool(cat, "pitch", tier, opts, "equipment") {
        drills(o, &f, l, w, p.seed + 500);
    }
}

/// FA youth formats: pitch (length, width), the pitch generator's markings, the goal style (the
/// FA's goalpost size for the format) and the FA line colour used where formats share ground.
pub fn youth_format(name: &str) -> (f64, f64, &'static str, &'static str, &'static str) {
    match name {
        "5v5" => (36.6, 27.4, "5v5", "youth_mini", "#d42a2a"),
        "7v7" => (54.9, 36.6, "7v7", "youth_mini", "#f0c419"),
        "11v11_u13" => (82.3, 50.3, "full", "youth_u13", "#f5f5f0"),
        _ => (73.2, 45.7, "9v9", "youth_9v9", "#2a62c9"),
    }
}
/// FA: 3 m clear round a youth pitch.
pub const YOUTH_RUNOFF: f64 = 3.0;

/// How youth pitches of l x w (plus run-off) pack into a pw x pd plot: (rotated, nx, ny) with the
/// most pitches, length along the plot when that fits as many; at most n_max.
pub fn youth_grid(pw: f64, pd: f64, l: f64, w: f64, n_max: usize) -> (bool, usize, usize) {
    let mut best: Option<(bool, usize, usize)> = None;
    for rot in [false, true] {
        let (cl, cw) = (l + 2.0 * YOUTH_RUNOFF, w + 2.0 * YOUTH_RUNOFF);
        let (cx, cy) = if rot { (cw, cl) } else { (cl, cw) };
        let (nx, ny) = (((pw - 4.0).max(0.0) / cx) as usize, ((pd - 4.0).max(0.0) / cy) as usize);
        if best.is_none_or(|b| nx * ny > b.1 * b.2) {
            best = Some((rot, nx, ny));
        }
    }
    let (rot, mut nx, mut ny) = best.unwrap();
    while nx * ny > n_max {
        if nx >= ny {
            nx -= 1;
        } else {
            ny -= 1;
        }
    }
    (rot, nx, ny)
}

/// Youth pitches in one of the FA formats, as many as fit the plot (or `count`), each with its own
/// goals, corner flags and session; the grass, line finish and kit improve by tier. Parents watch
/// from behind a post-and-rail on one touchline, the teams have the other.
fn youth_pitches(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize, opts: &Value) {
    let fac = "youth_pitches";
    let t = tier_spec(cat, fac, tier).cloned().unwrap_or(Value::Null);
    let fv = option(cat, fac, tier, opts, "format");
    let (l, w, marks, goals, fa_line) = youth_format(fv.as_str().unwrap_or("9v9"));
    let n_max = option(cat, fac, tier, opts, "count").as_f64().unwrap_or(99.0).max(1.0) as usize;
    let (rot, nx, ny) = youth_grid(p.w, p.d, l, w, n_max);
    if nx * ny == 0 {
        return;
    }
    let mow = option(cat, fac, tier, opts, "mow").as_str().map(String::from).unwrap_or_else(|| text(&t, "mow", "stripes").to_string());
    let line = if option(cat, fac, tier, opts, "lines").as_str() == Some("fa_colours") { fa_line } else { "#f5f5f0" };
    let lit = option_bool(cat, fac, tier, opts, "floodlights");
    let r = YOUTH_RUNOFF;
    let (cl, cw) = (l + 2.0 * r, w + 2.0 * r);
    let (cx, cy) = if rot { (cw, cl) } else { (cl, cw) };
    let mut k = 0u64;
    for j in 0..ny {
        for i in 0..nx {
            let (u, v) = ((i as f64 - (nx as f64 - 1.0) / 2.0) * cx, (j as f64 - (ny as f64 - 1.0) / 2.0) * cy);
            let f = p.frame.sub(u, v, if rot { FRAC_PI_2 } else { 0.0 });
            let pid = format!("{}-{}", p.id, k + 1);
            let seed = p.seed + 37 * k;
            let mut params = pitch_params(p, l, w, r, r, text(&t, "grass", "perennial_ryegrass"), num(&t, "height_mm", 30.0), num(&t, "wear", 0.4), num(&t, "wetness", 0.0), &mow, num(&t, "hand_painted", 0.0), num(&t, "crown", 0.1), marks);
            params["line_width"] = json!(0.1);
            params["line_color"] = json!(line);
            params["seed"] = json!((seed % 997) as i64 + 1);
            o.pitches.push(SitePitch {
                plot: pid.clone(),
                params,
                origin: f.pt(0.0, 0.0),
                yaw: f.yaw,
                goals: Some(goals.to_string()),
                both_goals: true,
                flags: Some(text(&t, "flags", "grassroots").to_string()),
                lit,
                surface: "grass".into(),
            });
            // side by side, alternate pitches are mirrored: parents share one gap, teams the next
            let side = if (if rot { i } else { j }) % 2 == 1 { -1.0 } else { 1.0 };
            if option_bool(cat, fac, tier, opts, "rails") {
                spectator_rail(o, &f, &pid, l / 2.0 + 1.0, -side * (w / 2.0 + 2.5));
            }
            if option_bool(cat, fac, tier, opts, "shelters") {
                for s in [-1.0, 1.0] {
                    o.prop_at(&f, "prop.team_shelter", s * 6.0f64.min(l / 6.0), side * (w / 2.0 + 2.2), LAWN_Z, if side > 0.0 { PI } else { 0.0 }, &[("TINT_club", &o.club.primary.clone())]);
                }
            }
            if option_bool(cat, fac, tier, opts, "ball_stop") {
                ball_stop(o, &f, &pid, l / 2.0 + 2.8, w / 2.0 + 2.0, 4.5);
            }
            if lit {
                pole_lights(o, &f, &pid, l, w, r, 2, 12.0, 2, 120.0);
            }
            if option_bool(cat, fac, tier, opts, "equipment") && k % 2 == 0 {
                youth_session(o, &f, l, w, seed + 500);
            }
            k += 1;
        }
    }
}

/// A post-and-rail spectator barrier along a touchline (pitch-local y): 48 mm galvanised posts
/// every 2.5 m, a top rail at 1.1 m and a mid rail.
fn spectator_rail(o: &mut Out, f: &Frame, name: &str, half: f64, y: f64) {
    let mut posts: Vec<Polyline> = Vec::new();
    let n = (2.0 * half / 2.5).round().max(1.0) as usize;
    for j in 0..=n {
        let x = -half + 2.0 * half * j as f64 / n as f64;
        posts.push((vec![DVec3::new(x, y, LAWN_Z), DVec3::new(x, y, LAWN_Z + 1.1)], 0.024));
    }
    for z in [1.1, 0.55] {
        posts.push((vec![DVec3::new(-half, y, LAWN_Z + z), DVec3::new(half, y, LAWN_Z + z)], 0.021));
    }
    o.tubes_at(f, &format!("{name}:rail"), posts, mats::named("Galvanised"), 8);
}

/// A youth session set out on a small pitch: a grid of disc cones, a cone slalom, pop-up mini goals
/// at one end and balls about (pitch-local, sized to the pitch).
fn youth_session(o: &mut Out, f: &Frame, l: f64, w: f64, seed: u64) {
    let u = |k: u64| uniform(seed + 10 + k, 0);
    let sx = if u(0) < 0.5 { 1.0 } else { -1.0 };
    let discs = DISC_PAIRS[(u(1) * DISC_PAIRS.len() as f64) as usize % DISC_PAIRS.len()];
    let tall = KIT_COLOURS[(u(2) * KIT_COLOURS.len() as f64) as usize % KIT_COLOURS.len()];
    let mut at = |key: &str, x: f64, y: f64, yaw: f64, tints: &[(&str, &str)]| o.prop_at(f, key, sx * x, y, LAWN_Z, yaw, tints);
    let step = 4.0f64.min(w / 8.0);
    for a in 0..3 {
        for b in 0..3 {
            let c = if (a + b) % 2 == 0 { discs.0 } else { discs.1 };
            at("prop.cone_disc", -l / 4.0 + (a as f64 - 1.0) * step, (b as f64 - 1.0) * step, 0.0, &[("TINT_cone", c)]);
        }
    }
    for i in 0..6 {
        at("prop.cone_tall", l / 8.0 + i as f64 * 1.5, -w / 4.0 + if i % 2 == 0 { 0.5 } else { -0.5 }, 0.0, &[("TINT_cone", tall)]);
    }
    if u(3) < 0.6 {
        for s2 in [-1.0, 1.0] {
            at("prop.mini_goal", l / 2.0 - 6.0, s2 * w / 5.0, -sx * FRAC_PI_2, &[]);
        }
    }
    for i in 0..(5 + (u(4) * 6.0) as u64) {
        let (x, y) = ((uniform(seed + 3, i) - 0.5) * l * 0.6, (uniform(seed + 4, i) - 0.5) * w * 0.6);
        at("prop.ball", x, y, uniform(seed + 5, i) * TAU, &[("TINT_ball", "#f4f4f2")]);
    }
    at("prop.ball_cart", 0.0, -w / 2.0 - 1.2, 0.3, &[]);
}

fn artificial(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize, opts: &Value) {
    let t = tier_spec(cat, "artificial_pitch", tier).cloned().unwrap_or(Value::Null);
    let (re, rs) = if tier == 1 { (2.0, 2.0) } else { (3.0, 3.0) };
    let (l, w) = pitch_dims(p, num(&t, "length", 60.0), num(&t, "width", 40.0), re, rs);
    let params = pitch_params(p, l, w, re, rs, "artificial_3g", 40.0, 0.1, 0.0, "stripes", 0.0, 0.03, if tier == 1 { "outline" } else { "full" });
    let lit = option_bool(cat, "artificial_pitch", tier, opts, "floodlights");
    o.pitches.push(SitePitch {
        plot: p.id.clone(),
        params,
        origin: p.frame.pt(0.0, 0.0),
        yaw: p.frame.yaw,
        goals: Some(text(&t, "goals", "portable").to_string()),
        both_goals: true,
        flags: None,
        lit,
        surface: "artificial".into(),
    });
    let f = p.frame;
    // the cage: 3G apron to the fence, the fence all round with a gate on the front
    let (hx, hy) = (l / 2.0 + re + 1.0, w / 2.0 + rs + 1.0);
    let mut apron = MeshAcc::new();
    kernel::ground(&mut apron, &kernel::rect(2.0 * hx, 2.0 * hy, 0.0, 0.0), LAWN_Z + 0.01, 0);
    o.part_at(&f, &format!("{}:apron", p.id), apron, vec![mats::turf_3g(f.pt(0.0, 0.0), f.yaw)], 0.0);
    let gate = DVec2::new(0.0, -hy);
    let ring = [gate + DVec2::new(1.2, 0.0), DVec2::new(hx, -hy), DVec2::new(hx, hy), DVec2::new(-hx, hy), DVec2::new(-hx, -hy), gate - DVec2::new(1.2, 0.0)];
    let world: Vec<DVec2> = ring.iter().map(|q| f.pt(q.x, q.y)).collect();
    landscape::chainlink(o, &world, num(&t, "fence", 4.5), &format!("{}:cage", p.id));
    if lit {
        if tier == 1 {
            pole_lights(o, &f, &p.id, l, w, rs, 2, 8.0, 2, 250.0);
        } else {
            pole_lights(o, &f, &p.id, l, w, rs, 3, 15.0, 3, 350.0);
        }
    }
    if option_bool(cat, "artificial_pitch", tier, opts, "equipment") {
        for i in 0..6u64 {
            o.prop_at(&f, "prop.cone_tall", -6.0 + i as f64 * 2.5, -w / 4.0, LAWN_Z, 0.0, &[("TINT_cone", "#ff6a1a")]);
        }
        o.prop_at(&f, "prop.ball_cart", 3.0, -w / 2.0 - 1.2, LAWN_Z, 0.0, &[]);
    }
}

/// The academy's full-size match pitch (U18 and U21 games): a perimeter rail round a railed
/// hardstanding, dugouts in front of a covered seated stand along one touchline, floodlights and
/// ball-stop nets; the stand grows with the tier, the top tier adds a filming tower.
fn showpiece(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize, opts: &Value) {
    let fac = "showpiece_pitch";
    let t = tier_spec(cat, fac, tier).cloned().unwrap_or(Value::Null);
    let rows = num(&t, "stand_rows", 0.0) as usize;
    let (re, rs) = (4.0, 3.0);
    let (l, w) = pitch_dims(p, num(&t, "length", 100.0), num(&t, "width", 64.0), re, rs);
    // the pitch sits off-centre, leaving room for the stand on the +y side
    let pf = p.frame.sub(0.0, -(rows as f64 * 0.8 + 1.0) / 2.0, 0.0);
    let grass = text(&t, "grass", "perennial_ryegrass").to_string();
    let params = pitch_params(p, l, w, re, rs, &grass, num(&t, "height_mm", 25.0), num(&t, "wear", 0.2), 0.0, "stripes", 0.0, num(&t, "crown", 0.15), "full");
    let lit = option_bool(cat, fac, tier, opts, "floodlights");
    o.pitches.push(SitePitch {
        plot: p.id.clone(),
        params,
        origin: pf.pt(0.0, 0.0),
        yaw: pf.yaw,
        goals: Some(text(&t, "goals", "classic").to_string()),
        both_goals: true,
        flags: Some(text(&t, "flags", "striped").to_string()),
        lit,
        surface: if grass.starts_with("hybrid") { "hybrid".into() } else { "grass".into() },
    });
    let (hx, hy) = (l / 2.0 + re - 0.5, w / 2.0 + rs - 0.5);
    spectator_rail(o, &pf, &format!("{}-n", p.id), hx, hy);
    spectator_rail(o, &pf, &format!("{}-s", p.id), hx, -hy);
    for sx in [-1.0, 1.0] {
        let ef = pf.sub(sx * hx, 0.0, FRAC_PI_2);
        spectator_rail(o, &ef, &format!("{}-{}", p.id, if sx > 0.0 { "e" } else { "w" }), hy, 0.0);
    }
    let mut walk = MeshAcc::new();
    let (r_out, r_in) = (kernel::rect(2.0 * hx + 4.0, 2.0 * hy + 4.0, 0.0, 0.0), kernel::rect(2.0 * hx, 2.0 * hy, 0.0, 0.0));
    for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
        kernel::ground(&mut walk, &[r_in[a], r_out[a], r_out[b], r_in[b]], COVER_Z + 0.01, 0);
    }
    o.part_at(&pf, &format!("{}:walkway", p.id), walk, vec![mats::named("Apron_Concrete")], 0.0);
    let club = o.club.clone();
    for sx in [-1.0, 1.0] {
        o.prop_at(&pf, "prop.team_shelter", sx * 6.0, w / 2.0 + 1.4, LAWN_Z, PI, &[("TINT_club", &club.primary)]);
    }
    ball_stop(o, &pf, &p.id, l / 2.0 + re + 2.5, w / 2.0 + 2.0, 6.0);
    if lit {
        // on the stand side the poles stand behind the stand, clear of its roof
        let behind = if rows > 0 { rows as f64 * 0.8 + 3.5 } else { 0.0 };
        pole_lights_behind(o, &pf, &p.id, l, w, rs, num(&t, "poles", 3.0) as usize, num(&t, "pole_h", 15.0), num(&t, "lamps", 2.0) as usize, num(&t, "lux", 200.0), behind);
    }
    if rows > 0 {
        let seat = option(cat, fac, tier, opts, "seat").as_str().unwrap_or("tipup_classic").to_string();
        let colour = option(cat, fac, tier, opts, "seat_color").as_str().map(String::from).unwrap_or_else(|| club.primary.clone());
        let sign = if option_bool(cat, fac, tier, opts, "sign") { Some(club.name.clone()) } else { None };
        stand(o, &pf, &p.id, num(&t, "stand_length", 40.0), rows, w / 2.0 + rs + 1.5, &seat, &colour, sign.as_deref());
    }
    if option_bool(cat, fac, tier, opts, "filming_tower") {
        let tf = pf.sub(0.0, -(w / 2.0 + rs + 5.0).min(p.d / 2.0 - 2.0), 0.0);
        filming_tower(o, &tf, &p.id, 7.0);
    }
}

/// A covered seated stand along a touchline (frame-local, the pitch at -y): concrete terraces
/// (0.8 m going, 0.3 m risers, the front row 0.6 m up behind a club-coloured wall), seats with an
/// aisle every 13 m and intermediate steps in it, a walkway behind, a profiled steel back wall and
/// end screens, and a roof sloping to the rear on white steel columns with raking rafters, its
/// fascia carrying the club's name.
#[allow(clippy::too_many_arguments)]
fn stand(o: &mut Out, f: &Frame, name: &str, ln: f64, rows: usize, y0: f64, seat: &str, seat_color: &str, sign: Option<&str>) {
    let (g, rr, z0) = (0.8, 0.3, 0.6);
    let v = DVec3::new;
    let mut m = MeshAcc::new();
    let hl = ln / 2.0;
    // (the concrete stops 3 cm short of the end screens and the back wall: never in their plane)
    for r in 0..rows {
        bx(&mut m, -hl + 0.03, y0 + r as f64 * g, LAWN_Z, hl - 0.03, y0 + (r + 1) as f64 * g, LAWN_Z + z0 + r as f64 * rr, 0);
    }
    let z_top = LAWN_Z + z0 + rows as f64 * rr;
    let yb = y0 + rows as f64 * g + 1.5;
    bx(&mut m, -hl + 0.03, y0 + rows as f64 * g, LAWN_Z, hl - 0.03, yb - 0.03, z_top - rr, 0);
    bx(&mut m, -hl, y0 - 0.15, LAWN_Z, hl, y0, LAWN_Z + z0 + 0.8, 4);
    let (zr_f, zr_b) = (z_top + 3.4, z_top + 2.8);
    let yf = y0 - 1.2;
    quad(&mut m, v(-hl, yb, LAWN_Z), v(hl, yb, LAWN_Z), v(hl, yb, zr_b), v(-hl, yb, zr_b), 1);
    // the way in from behind: three entrances (stairs up inside to the walkway) in white frames,
    // and a refreshment kiosk with a club-coloured roller shutter over its counter
    for x in [-hl / 2.0, 0.0, hl / 2.0] {
        quad(&mut m, v(x + 0.9, yb + 0.02, LAWN_Z), v(x - 0.9, yb + 0.02, LAWN_Z), v(x - 0.9, yb + 0.02, LAWN_Z + 2.3), v(x + 0.9, yb + 0.02, LAWN_Z + 2.3), 6);
        for (x0, x1, z1, z2) in [(x - 1.05, x - 0.9, LAWN_Z, LAWN_Z + 2.45), (x + 0.9, x + 1.05, LAWN_Z, LAWN_Z + 2.45), (x - 1.05, x + 1.05, LAWN_Z + 2.3, LAWN_Z + 2.45)] {
            bx(&mut m, x0, yb, z1, x1, yb + 0.1, z2, 5);
        }
    }
    kernel::ground(&mut m, &kernel::rect(ln + 4.0, 3.0, 0.0, yb + 1.5), COVER_Z + 0.01, 0);
    let xk = -hl / 4.0;
    bx(&mut m, xk - 1.6, yb, LAWN_Z + 0.95, xk + 1.6, yb + 0.45, LAWN_Z + 1.05, 0);
    bx(&mut m, xk - 1.5, yb, LAWN_Z + 1.05, xk + 1.5, yb + 0.06, LAWN_Z + 2.25, 4);
    bx(&mut m, xk - 1.6, yb, LAWN_Z + 2.25, xk + 1.6, yb + 0.25, LAWN_Z + 2.45, 5);
    for sx in [-1.0, 1.0] {
        let x = sx * hl;
        let pts = [v(x, y0, LAWN_Z), v(x, yb, LAWN_Z), v(x, yb, zr_b), v(x, yf + 1.4, zr_f - 0.1)];
        if sx > 0.0 {
            quad(&mut m, pts[0], pts[1], pts[2], pts[3], 1);
        } else {
            quad(&mut m, pts[3], pts[2], pts[1], pts[0], 1);
        }
    }
    let e = 0.6;
    quad(&mut m, v(-hl - e, yf, zr_f), v(hl + e, yf, zr_f), v(hl + e, yb + 0.3, zr_b), v(-hl - e, yb + 0.3, zr_b), 2);
    quad(&mut m, v(-hl - e, yb + 0.3, zr_b - 0.12), v(hl + e, yb + 0.3, zr_b - 0.12), v(hl + e, yf, zr_f - 0.12), v(-hl - e, yf, zr_f - 0.12), 3);
    bx(&mut m, -hl - e, yf - 0.25, zr_f - 0.75, hl + e, yf, zr_f + 0.08, 4);
    if let Some(sign) = sign {
        buildings::letters(&mut m, ln + 2.0 * e, -2.0 * (yf - 0.25), 0, sign, (ln + 2.0 * e) / 2.0, zr_f - 0.62, ln * 0.45, 5, 0.5 / 7.0, 0.0);
    }
    let mut steel: Vec<Polyline> = Vec::new();
    let n = (ln / 7.5).round().max(1.0) as usize;
    for k in 0..=n {
        let x = -hl + ln * k as f64 / n as f64;
        steel.push((vec![v(x, yb - 0.3, LAWN_Z), v(x, yb - 0.3, zr_b - 0.15)], 0.14));
        steel.push((vec![v(x, yb - 0.3, zr_b - 0.15), v(x, yf + 0.3, zr_f - 0.15)], 0.09));
        steel.push((vec![v(x, yb - 0.3, z_top + 0.6), v(x, (yb + yf) / 2.0, (zr_b + zr_f) / 2.0 - 0.15)], 0.06));
    }
    o.tubes_at(f, &format!("{name}:stand_steel"), steel, mats::tint("Roof_Steel", "#eceeee"), 8);
    // seats, an aisle every 13 m with a half step at each row
    let na = (ln / 13.0).round().max(1.0) as usize;
    let aisles: Vec<f64> = (0..na).map(|k| -hl + ln * (k as f64 + 0.5) / na as f64).collect();
    let key = format!("seat.{seat}.std");
    for r in 0..rows {
        let (y, z) = (y0 + r as f64 * g, LAWN_Z + z0 + r as f64 * rr);
        for &xa in &aisles {
            bx(&mut m, xa - 0.6, y, z, xa + 0.6, y + 0.4, z + rr / 2.0, 0);
        }
        let mut x = -hl + 0.6;
        while x <= hl - 0.6 {
            if aisles.iter().all(|xa| (x - xa).abs() > 0.95) {
                o.prop_at(f, &key, x, y + 0.45, z, PI, &[("TINT_seat", seat_color)]);
            }
            x += 0.5;
        }
    }
    let primary = o.club.primary.clone();
    o.part_at(f, &format!("{name}:stand"), m, vec![mats::named("Stand_Concrete"), mats::tint("TINT_corrugated", "#59616a"), mats::tint("TINT_roof", "#d9dbdc"), mats::named("Roof_Underside"), mats::tint("TINT_club", &primary), mats::tint("TINT_trim", "#f2f2f0"), mats::named("Bldg_Opening")], 0.0);
}

/// Small-sided 3G cages for the youngest age groups: 3G with 5v5 markings and youth goals, steel
/// rebound boards round the base (1.2 m, the ball plays off them), chain-link above, a gate on one
/// side and LED floodlights on short columns; one cage, or as many as fit the plot.
fn cages(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize, opts: &Value) {
    let t = tier_spec(cat, "cages", tier).cloned().unwrap_or(Value::Null);
    let (l, w) = (num(&t, "length", 30.0), num(&t, "width", 20.0));
    let (rot, nx, ny) = youth_grid(p.w, p.d, l, w, num(&t, "max", 1.0) as usize);
    let lit = option_bool(cat, "cages", tier, opts, "floodlights");
    let (cl, cw) = (l + 2.0 * YOUTH_RUNOFF, w + 2.0 * YOUTH_RUNOFF);
    let (cx, cy) = if rot { (cw, cl) } else { (cl, cw) };
    let mut k = 0u64;
    for j in 0..ny {
        for i in 0..nx {
            let f = p.frame.sub((i as f64 - (nx as f64 - 1.0) / 2.0) * cx, (j as f64 - (ny as f64 - 1.0) / 2.0) * cy, if rot { FRAC_PI_2 } else { 0.0 });
            let cid = format!("{}-{}", p.id, k + 1);
            let mut params = pitch_params(p, l, w, 1.0, 1.0, "artificial_3g", 40.0, 0.15, 0.0, "stripes", 0.0, 0.02, "5v5");
            params["line_width"] = json!(0.08);
            params["seed"] = json!(((p.seed + 37 * k) % 997) as i64 + 1);
            o.pitches.push(SitePitch { plot: cid.clone(), params, origin: f.pt(0.0, 0.0), yaw: f.yaw, goals: Some("youth_mini".into()), both_goals: true, flags: None, lit, surface: "artificial".into() });
            let (hx, hy) = (l / 2.0 + 1.0, w / 2.0 + 1.0);
            let mut boards = MeshAcc::new();
            let gate = 1.2;
            // (just inside the chain-link, mitred at the corners: no face shares another's plane)
            let (a, b) = (0.03, 0.08);
            for (x0, y0, x1, y1) in [(-hx + b, -hy + a, -gate, -hy + b), (gate, -hy + a, hx - b, -hy + b), (-hx + b, hy - b, hx - b, hy - a), (-hx + a, -hy + a, -hx + b, hy - a), (hx - b, -hy + a, hx - a, hy - a)] {
                bx(&mut boards, x0, y0, LAWN_Z, x1, y1, LAWN_Z + 1.2, 0);
            }
            o.part_at(&f, &format!("{cid}:boards"), boards, vec![mats::tint("TINT_fence", "#2b4433")], 0.0);
            let ring: Vec<DVec2> = [(-gate, -hy), (-hx, -hy), (-hx, hy), (hx, hy), (hx, -hy), (gate, -hy)].iter().map(|q| f.pt(q.0, q.1)).collect();
            landscape::chainlink(o, &ring, num(&t, "fence", 4.0), &format!("{cid}:cage"));
            if lit {
                pole_lights(o, &f, &cid, l, w, 1.0, 2, 8.0, 2, 250.0);
            }
            k += 1;
        }
    }
}

fn goalkeeper(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize) {
    let t = tier_spec(cat, "goalkeeper_area", tier).cloned().unwrap_or(Value::Null);
    let (l, w) = (num(&t, "length", 34.0).min(p.w - 8.0), num(&t, "width", 26.0).min(p.d - 6.0));
    let params = pitch_params(p, l, w, 2.0, 2.0, "perennial_ryegrass", 26.0, 0.5, 0.0, "stripes_long", 0.0, 0.03, "none");
    o.pitches.push(SitePitch { plot: format!("{}-gk", p.id), params, origin: p.frame.pt(0.0, 0.0), yaw: p.frame.yaw, goals: Some("box".into()), both_goals: tier >= 2, flags: None, lit: false, surface: "grass".into() });
    let f = p.frame;
    for i in 0..3u64 {
        o.prop_at(&f, "prop.mannequin", l / 2.0 - 9.0, -2.0 + i as f64 * 2.0, LAWN_Z, -FRAC_PI_2, &[("TINT_mannequin", "#f4d21c")]);
    }
    if tier >= 2 {
        for i in 0..3u64 {
            o.prop_at(&f, "prop.rebounder", -6.0 + i as f64 * 6.0, -w / 2.0 + 1.0, LAWN_Z, 0.0, &[]);
        }
        ball_stop(o, &f, &p.id, l / 2.0 + 3.0, w / 2.0 + 1.0, 6.0);
        o.prop_at(&f, "prop.storage_container", -l / 2.0 - 1.0, w / 2.0 + 0.5, COVER_Z, FRAC_PI_2, &[("TINT_container", "#3c5a3a")]);
    }
    o.prop_at(&f, "prop.ball_cart", 0.0, -w / 2.0 - 1.0, LAWN_Z, 0.0, &[]);
}

// -- buildings --------------------------------------------------------------------------------------

/// A building's place on a plot: its frame (front, local -y, to the plot's frontage), its size
/// (clamped to the plot) and the gap left in front of it for a forecourt or yard.
#[derive(Clone, Copy, Debug)]
pub struct Placed {
    pub f: Frame,
    pub w: f64,
    pub d: f64,
    pub gap: f64,
}

pub fn place(p: &Plot, bw: f64, bd: f64, forecourt: f64) -> Placed {
    let yaw = p.front.x.atan2(-p.front.y);
    let along_x = p.front.y.abs() > 0.5;
    let (pw, pd) = if along_x { (p.w, p.d) } else { (p.d, p.w) };
    let w = bw.min(pw - 6.0);
    let d = bd.min(pd - forecourt - 2.0).max(4.0);
    let lim = (pd / 2.0 - d / 2.0 - 1.0).max(0.0);
    let c = (pd / 2.0 - forecourt - d / 2.0).clamp(-lim, lim);
    let off = p.front * c;
    Placed { f: p.frame.sub(off.x, off.y, yaw), w, d, gap: pd / 2.0 - c - d / 2.0 }
}

/// Paving round a building and out to the plot frontage.
pub fn forecourt(o: &mut Out, p: &Plot, b: &Placed) {
    let mut m = MeshAcc::new();
    let (y0, y1) = (-b.d / 2.0 - b.gap + 0.5, b.d / 2.0 + 2.0);
    kernel::ground(&mut m, &kernel::rect(b.w + 4.0, y1 - y0, 0.0, (y0 + y1) / 2.0), COVER_Z, 0);
    o.part_at(&b.f, &format!("{}:forecourt", p.id), m, vec![mats::named("Plaza_Paving")], 0.0);
}

/// The footprint a facility's main building will have (for its construction site): frame, size,
/// storeys and storey height.
pub fn footprint(cat: &Catalog, p: &Plot, fac: &str, tier: usize) -> Option<(Placed, usize, f64)> {
    if cat.facilities.get(fac).and_then(|f| f.get("sizes")).is_some_and(|v| v.is_array()) {
        let sizes: Vec<Value> = cat.facilities.get(fac).and_then(|f| f.get("sizes")).and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let opts = p.cfg.get("options").filter(|v| v.is_object()).cloned().unwrap_or(json!({}));
        let sv = option(cat, fac, tier, &opts, "size");
        let si = (sv.as_f64().map(|v| v as i64).unwrap_or(2)).max(1).min(sizes.len().max(1) as i64) as usize;
        let sz = sizes.get(si - 1).cloned().unwrap_or(json!({"w": 28.8, "d": 15.6, "storeys": 1}));
        return Some((place(p, num(&sz, "w", 28.8), num(&sz, "d", 15.6), 16.0), num(&sz, "storeys", 1.0) as usize, 3.6));
    }
    let t = tier_spec(cat, fac, tier)?;
    let size = t.get("size")?;
    let (bw, bd) = (size[0].as_f64()?, size[1].as_f64()?);
    match fac {
        "training_building" | "academy_building" => Some((place(p, bw, bd, 14.0), num(t, "storeys", 1.0) as usize, 4.0)),
        "recovery" => Some((place(p, bw, bd, 8.0), 1, 4.0)),
        "maintenance" => Some((place(p, bw, bd, 9.0), 1, 5.6)),
        "indoor_hall" => {
            let yaw = p.front.x.atan2(-p.front.y);
            let along_x = p.front.y.abs() > 0.5;
            let (w, d) = (bw.min(p.w - 8.0), bd.min(p.d - 10.0));
            let (ww, dd) = if along_x { (w, d) } else { (d, w) };
            Some((Placed { f: p.frame.sub(0.0, 0.0, yaw), w: ww, d: dd, gap: 4.0 }, 1, num(t, "height", 13.0) - buildings::PLINTH))
        }
        _ => None,
    }
}

/// The training building, or (fac "academy_building") an academy's: the same building tiers
/// (portacabins to a flagship) with the academy's own modules: classrooms, sports science, a
/// parents' lounge and a scholars' lodge, and the academy's name on the sign.
fn training_building(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize, style: &str, opts: &Value, fac: &str) {
    let t = tier_spec(cat, fac, tier).cloned().unwrap_or(Value::Null);
    let club = o.club.clone();
    let academy = fac == "academy_building";
    let modules: Vec<String> = option(cat, fac, tier, opts, "modules").as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
    let has = |m: &str| modules.iter().any(|x| x == m);
    if tier == 1 {
        // portacabins: two on the ground, two stacked, an external stair and a walkway
        cover(o, p, "ground", 1.0, mats::named("Site_Gravel"));
        let f = place(p, 20.0, 7.0, 5.0).f;
        for (i, (x, z)) in [(-5.0, 0.0), (5.0, 0.0), (-5.0, 2.62), (5.0, 2.62)].iter().enumerate() {
            if i >= num(&t, "cabins", 4.0) as usize {
                break;
            }
            o.prop_at(&f, "prop.site_cabin", *x, 0.0, COVER_Z + z, PI, &[("TINT_cabin", "#e3e4df")]);
        }
        // the upper cabins' access: a galvanised stair climbing to a landing at the end of a
        // walkway along the doors, stringers, treads, handrails and a guard rail with posts
        let (top, n) = (2.62, 12usize);
        let rise = top / n as f64;
        let mut walk = MeshAcc::new();
        bx(&mut walk, -10.0, -2.4, COVER_Z + top - 0.07, 10.0, -1.5, COVER_Z + top, 0);
        bx(&mut walk, 10.0, -2.4, COVER_Z + top - 0.07, 11.0, -1.4, COVER_Z + top, 0);
        for k in 0..n - 1 {
            let x1 = 11.0 + (k + 1) as f64 * 0.26;
            let z = COVER_Z + top - (k + 1) as f64 * rise;
            bx(&mut walk, x1 - 0.26, -2.4, z - 0.04, x1, -1.4, z, 0);
        }
        o.part_at(&f, &format!("{}:walkway", p.id), walk, vec![mats::named("Galvanised")], 0.0);
        let foot = 11.0 + n as f64 * 0.26;
        let mut st: Vec<Polyline> = Vec::new();
        for y in [-2.42, -1.38] {
            st.push((vec![DVec3::new(foot, y, COVER_Z), DVec3::new(11.0, y, COVER_Z + top - 0.07)], 0.06));
            st.push((vec![DVec3::new(foot + 0.1, y, COVER_Z + 1.0), DVec3::new(11.0, y, COVER_Z + top + 1.0), DVec3::new(10.0, y, COVER_Z + top + 1.0)], 0.024));
            st.push((vec![DVec3::new(foot, y, COVER_Z), DVec3::new(foot + 0.1, y, COVER_Z + 1.0)], 0.024));
        }
        st.push((vec![DVec3::new(10.0, -2.42, COVER_Z + top + 1.1), DVec3::new(-10.0, -2.42, COVER_Z + top + 1.1)], 0.024));
        st.push((vec![DVec3::new(10.0, -2.42, COVER_Z + top + 0.55), DVec3::new(-10.0, -2.42, COVER_Z + top + 0.55)], 0.018));
        for k in 0..11 {
            let x = -10.0 + k as f64 * 2.0;
            st.push((vec![DVec3::new(x, -2.42, COVER_Z + top), DVec3::new(x, -2.42, COVER_Z + top + 1.1)], 0.024));
            st.push((vec![DVec3::new(x, -2.4, COVER_Z), DVec3::new(x, -2.4, COVER_Z + top - 0.07)], 0.05));
        }
        o.tubes_at(&f, &format!("{}:stairs", p.id), st, mats::named("Galvanised"), 6);
        for (i, key) in ["prop.bin_wheelie", "prop.bin_wheelie", "prop.picnic_table"].iter().enumerate() {
            // bins turned to face the yard (their front is +y), backs to the cabins
            let yaw = if i < 2 { PI } else { 0.0 };
            o.prop_at(&f, key, -12.0 + i as f64 * 1.0 + if i == 2 { -3.0 } else { 0.0 }, -4.5, COVER_Z, yaw, &[("TINT_bin", "#2f3a2f")]);
        }
        o.entrance = Some(f.pt(0.0, -3.0));
        return;
    }
    let size = &t["size"];
    let (bw, bd) = (size[0].as_f64().unwrap_or(40.0), size[1].as_f64().unwrap_or(18.0));
    let storeys = num(&t, "storeys", 1.0) as usize;
    let pl = place(p, bw, bd, 14.0);
    let (f, w, d) = (pl.f, pl.w, pl.d);
    forecourt(o, p, &pl);
    let mut sp: Spec = buildings::styled(cat, style, w, d, storeys, &club.primary, o.night, opts, p.seed);
    let signed = option(cat, fac, tier, opts, "sign").as_bool().unwrap_or(true);
    let name = if academy { format!("{} ACADEMY", club.name) } else { club.name.clone() };
    // a flagship carries its name on the glass atrium in front (the main facade is behind it)
    if signed && tier < 4 {
        sp.sign = Some(name.clone());
    }
    let b = buildings::build(o, &f, &format!("{}:building", p.id), &sp);
    o.entrance = Some(f.pt(b.entrance.x, b.entrance.y));
    // wings for the modules: a glazed gym to the right, a pool wing to the left
    let wing = |o: &mut Out, name: &str, ww: f64, dd: f64, side: f64, window: &str| {
        let mut s2 = buildings::styled(cat, style, ww, dd, 1, &club.primary, o.night, opts, p.seed + 7);
        s2.roof = if matches!(s2.roof.as_str(), "gable" | "hip") { "hip".into() } else { s2.roof.clone() };
        s2.window = window.into();
        s2.door = "none".into();
        s2.canopy = "none".into();
        s2.clock = false;
        s2.arcade = false;
        let wf = f.sub(side * (w / 2.0 + ww / 2.0 - 0.1), (dd - d) / 2.0 + 0.6, 0.0);
        buildings::build(o, &wf, &format!("{}:{name}", p.id), &s2);
    };
    if has("gym") {
        wing(o, "gym", 16.0, d * 0.8, 1.0, "ribbon");
    } else if has("sports_science") {
        wing(o, "sports_science", 16.0, d * 0.8, 1.0, "ribbon");
    }
    if has("rehab") {
        wing(o, "pool", 18.0, d * 0.75, -1.0, "strip");
    } else if has("classrooms") {
        // the education wing: classrooms with big windows (full-time scholars are schooled on site)
        wing(o, "classrooms", 20.0, d * 0.8, -1.0, "large");
    }
    if tier >= 4 {
        // the flagship: an atrium in front and a rear wing
        let mut atrium = buildings::styled(cat, "ultra_modern", 14.0, 7.0, storeys.min(2), &club.primary, o.night, opts, p.seed + 11);
        atrium.window = "ribbon".into();
        atrium.door = "glass".into();
        atrium.canopy = "none".into();
        atrium.fins = false;
        atrium.mats[buildings::T_WALL as usize] = mats::named("Facade_DarkGlass");
        atrium.roof = "flat".into();
        if signed {
            atrium.sign = Some(name.clone());
        }
        let af = f.sub(0.0, -d / 2.0 - 3.4, 0.0);
        let a = buildings::build(o, &af, &format!("{}:atrium", p.id), &atrium);
        o.entrance = Some(af.pt(a.entrance.x, a.entrance.y));
        let mut rear = buildings::styled(cat, style, 22.0, 26.0, 2, &club.primary, o.night, opts, p.seed + 13);
        rear.door = "none".into();
        rear.canopy = "none".into();
        rear.clock = false;
        rear.arcade = false;
        let rf = f.sub(w / 2.0 - 14.0, d / 2.0 + 13.0 - 0.2, 0.0);
        buildings::build(o, &rf, &format!("{}:rear_wing", p.id), &rear);
    }
    // the canteen terrace on the pitch side (behind), media dishes on the roof
    if has("canteen") || has("parents_lounge") {
        let tf = f.sub(-w / 4.0, d / 2.0 + 5.0, 0.0);
        let mut m = MeshAcc::new();
        kernel::ground(&mut m, &kernel::rect(16.0, 9.0, 0.0, 0.0), COVER_Z + 0.01, 0);
        o.part_at(&tf, &format!("{}:terrace", p.id), m, vec![mats::named("Site_Wood")], 0.0);
        for i in 0..4u64 {
            let x = -5.4 + i as f64 * 3.6;
            o.prop_at(&tf, "prop.picnic_table", x, 0.5, COVER_Z, 0.0, &[]);
            o.prop_at(&tf, "prop.parasol", x, 0.5, COVER_Z, 0.0, &[("TINT_club", &club.primary)]);
        }
    }
    if has("media") {
        let h = buildings::PLINTH + storeys as f64 * sp.storey_h;
        let z = if sp.roof == "flat" { h } else { h + 0.3 };
        o.prop_at(&f, "prop.satellite_dish", w / 2.0 - 4.0, d / 2.0 - 3.0, z, PI * 0.8, &[]);
        o.prop_at(&f, "prop.satellite_dish", w / 2.0 - 7.0, d / 2.0 - 3.0, z, PI * 0.8, &[]);
    }
    // flags, bins, bikes and benches at the entrance
    for (i, s) in [-1.0, 0.0, 1.0].iter().enumerate() {
        let col = if i == 1 { club.secondary.clone() } else { club.primary.clone() };
        o.prop_at(&f, "prop.flagpole", s * 3.0 - w / 2.0 + 6.0, -d / 2.0 - 9.0, COVER_Z, 0.0, &[("TINT_flag", &col)]);
    }
    o.prop_at(&f, "prop.bike_rack", w / 2.0 - 4.0, -d / 2.0 - 3.0, COVER_Z, 0.0, &[]);
    if academy && has("boarding") && tier >= 3 {
        // the scholars' lodge: a two-storey residential block behind, its own front door
        let mut lodge = buildings::styled(cat, style, 26.0, 11.0, 2, &club.primary, o.night, opts, p.seed + 17);
        lodge.window = "punched".into();
        lodge.roof = if matches!(lodge.roof.as_str(), "gable" | "hip" | "shallow" | "barrel") { "hip".into() } else { lodge.roof.clone() };
        lodge.clock = false;
        lodge.arcade = false;
        lodge.canopy = "none".into();
        lodge.sign = None;
        let lf = f.sub(-(w / 2.0 - 13.0), d / 2.0 + 22.0, PI);
        buildings::build(o, &lf, &format!("{}:lodge", p.id), &lodge);
    }
    o.prop_at(&f, "prop.bin_litter", 3.6, -d / 2.0 - 2.4, COVER_Z, 0.0, &[("TINT_bin", "#2a2c2e")]);
    o.prop_at(&f, "prop.bench", -5.0, -d / 2.0 - 3.2, COVER_Z, PI, &[]);
    o.prop_at(&f, "genprop.pigeon_pecking", -3.9, -d / 2.0 - 4.4, COVER_Z, 2.1, &[]);
    o.prop_at(&f, "genprop.bicycle", w / 2.0 - 5.85, -d / 2.0 - 3.0, COVER_Z, FRAC_PI_2, &[]);
    if tier >= 3 {
        // a bust of the club's great manager by the door; a bronze ball at an academy
        o.prop_at(&f, if academy { "genprop.bronze_football_plinth" } else { "genprop.bust_pedestal" }, -9.0, -d / 2.0 - 4.5, COVER_Z, PI, &[]);
    }
}

/// An inflated dome over a half pitch: an ellipsoid of membrane on a ring beam.
fn air_dome(o: &mut Out, f: &Frame, name: &str, a: f64, b: f64, h: f64) {
    let mut m = MeshAcc::new();
    let (nu, nv) = (36usize, 9usize);
    let pt = |i: usize, j: usize| -> DVec3 {
        let phi = TAU * i as f64 / nu as f64;
        let th = FRAC_PI_2 * j as f64 / nv as f64;
        // a flattened ellipsoid: steep sides, a broad top, as air halls are
        let c = th.cos().powf(0.55);
        DVec3::new(a * c * phi.cos(), b * c * phi.sin(), COVER_Z + 0.4 + h * th.sin())
    };
    for j in 0..nv {
        for i in 0..nu {
            let (p0, p1, p2, p3) = (pt(i, j), pt((i + 1) % nu, j), pt((i + 1) % nu, j + 1), pt(i, j + 1));
            if j == nv - 1 {
                kernel::tri(&mut m, p0, p1, p2, 0);
            } else {
                quad(&mut m, p0, p1, p2, p3, 0);
            }
        }
    }
    // the ring beam and an airlock door
    let mut base = MeshAcc::new();
    kernel::prism(&mut base, &(0..nu).map(|i| { let q = pt(i, 0); DVec2::new(q.x * 1.02, q.y * 1.02) }).collect::<Vec<_>>(), COVER_Z, COVER_Z + 0.45, 1, Some(1), None);
    bx(&mut base, -2.0, -b - 4.0, COVER_Z, 2.0, -b + 1.0, COVER_Z + 3.0, 2);
    bx(&mut base, a - 2.0, -6.0, COVER_Z, a + 3.0, -2.0, COVER_Z + 2.2, 2);
    o.part_at(f, &format!("{name}:dome"), m, vec![mats::tint("TINT_roof_membrane", "#f1f1ec")], 25.0);
    o.part_at(f, &format!("{name}:dome_base"), base, vec![mats::tint("TINT_roof_membrane", "#f1f1ec"), mats::named("Facade_Concrete"), mats::tint("TINT_trim", "#3b3e42")], 0.0);
}

fn hall(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize, style: &str, opts: &Value) {
    let t = tier_spec(cat, "indoor_hall", tier).cloned().unwrap_or(Value::Null);
    let club = o.club.clone();
    let size = &t["size"];
    let (bw, bd) = (size[0].as_f64().unwrap_or(64.0), size[1].as_f64().unwrap_or(44.0));
    let h = num(&t, "height", 13.0);
    let f = p.frame.sub(0.0, 0.0, 0.0);
    let (w, d) = (bw.min(p.w - 8.0), bd.min(p.d - 10.0));
    let mut apron = MeshAcc::new();
    kernel::ground(&mut apron, &kernel::rect(w + 6.0, d + 6.0, 0.0, 0.0), COVER_Z - 0.005, 0);
    o.part_at(&f, &format!("{}:apron", p.id), apron, vec![mats::named("Apron_Concrete")], 0.0);
    if tier == 1 {
        air_dome(o, &f, &p.id, w / 2.0, d / 2.0, h);
        return;
    }
    let yaw = p.front.x.atan2(-p.front.y);
    let along_x = p.front.y.abs() > 0.5;
    let (ww, dd) = if along_x { (w, d) } else { (d, w) };
    let hf = p.frame.sub(0.0, 0.0, yaw);
    let mut sp = if tier == 2 {
        let mut s = buildings::styled(cat, "industrial", ww, dd, 1, &club.primary, o.night, opts, p.seed);
        s.mats[buildings::T_WALL as usize] = mats::tint("TINT_corrugated", "#7b8a83");
        s
    } else {
        buildings::styled(cat, style, ww, dd, 1, &club.primary, o.night, opts, p.seed)
    };
    sp.storey_h = h - buildings::PLINTH;
    sp.roof = if style == "industrial" || tier == 2 { "shallow".into() } else { "barrel".into() };
    sp.pitch_deg = 6.0;
    sp.window = "strip".into();
    sp.door = "glass".into();
    sp.canopy = "none".into();
    sp.clock = false;
    sp.arcade = false;
    sp.quoins = false;
    sp.fins = false;
    sp.roller_doors = 2;
    sp.mats[buildings::T_DOOR as usize] = mats::tint("TINT_door", "#6e7377");
    if option(cat, "indoor_hall", tier, opts, "sign").as_bool().unwrap_or(true) {
        sp.sign = Some(club.name.clone());
    }
    buildings::build(o, &hf, &format!("{}:hall", p.id), &sp);
    if tier >= 3 {
        // a two-storey entrance block in the site style against the front
        let mut e = buildings::styled(cat, style, 28.0, 9.0, 2, &club.primary, o.night, opts, p.seed + 3);
        e.roof = if matches!(e.roof.as_str(), "gable" | "hip") { "hip".into() } else { e.roof.clone() };
        e.clock = false;
        e.arcade = false;
        let ef = hf.sub(0.0, -dd / 2.0 - 4.4, 0.0);
        buildings::build(o, &ef, &format!("{}:entrance", p.id), &e);
    }
}

fn recovery(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize, style: &str) {
    let t = tier_spec(cat, "recovery", tier).cloned().unwrap_or(Value::Null);
    let club = o.club.clone();
    if tier == 1 {
        cover(o, p, "deck", 2.0, mats::named("Site_Wood"));
        let f = place(p, 14.0, 6.0, 4.0).f;
        o.prop_at(&f, "prop.site_cabin", -3.0, 1.0, COVER_Z, FRAC_PI_2, &[("TINT_cabin", "#f2f3f0")]);
        let mut m = MeshAcc::new();
        cylinder(&mut m, 6.5, 0.5, COVER_Z, COVER_Z + 2.4, 1.1, 20, 0, true);
        for i in 0..3 {
            bx(&mut m, -9.0 + i as f64 * 2.2, -4.5, COVER_Z, -7.4 + i as f64 * 2.2, -2.6, COVER_Z + 0.75, 1);
        }
        o.part_at(&f, &format!("{}:cryo", p.id), m, vec![mats::named("Galvanised"), mats::tint("TINT_trim", "#e9eae6")], 0.0);
        return;
    }
    let size = &t["size"];
    let pl = place(p, size[0].as_f64().unwrap_or(24.0), size[1].as_f64().unwrap_or(14.0), 8.0);
    let (f, w, d) = (pl.f, pl.w, pl.d);
    forecourt(o, p, &pl);
    let mut sp = buildings::styled(cat, style, w, d, 1, &club.primary, o.night, &Value::Null, p.seed);
    sp.window = if tier == 3 { "large".into() } else { sp.window.clone() };
    sp.roof = if matches!(sp.roof.as_str(), "gable") { "hip".into() } else { sp.roof.clone() };
    sp.clock = false;
    sp.arcade = false;
    buildings::build(o, &f, &format!("{}:building", p.id), &sp);
    // plunge pools on a deck at the side
    let pf = f.sub(w / 2.0 + 4.0, 0.0, 0.0);
    let mut deck = MeshAcc::new();
    bx(&mut deck, -2.6, -d / 2.0 + 1.0, COVER_Z, 2.6, d / 2.0 - 1.0, COVER_Z + 0.5, 0);
    let mut water = MeshAcc::new();
    for i in 0..2 {
        let y = -d / 4.0 + i as f64 * d / 2.0 - 1.4;
        kernel::ground(&mut water, &kernel::rect(3.0, 2.8, 0.0, y + 1.4), COVER_Z + 0.45, 0);
    }
    o.part_at(&pf, &format!("{}:pool_deck", p.id), deck, vec![mats::named("Site_Wood")], 0.0);
    o.part_at(&pf, &format!("{}:pools", p.id), water, vec![mats::named("Site_Water")], 0.0);
}

/// A painted lane arrow pointing along +x (sx = 1) or -x: a 3 m shaft and a head.
fn arrow(m: &mut MeshAcc, x: f64, y: f64, sx: f64, z: f64) {
    kernel::ground(m, &kernel::rect(3.0, 0.22, x - sx * 0.8, y), z, 0);
    let (a, b, c) = (DVec2::new(x + sx * 0.7, y - 0.6), DVec2::new(x + sx * 1.9, y), DVec2::new(x + sx * 0.7, y + 0.6));
    kernel::ground(m, &if sx > 0.0 { [a, b, c] } else { [c, b, a] }, z, 0);
}

/// Arrival for an academy (players are dropped off, not parked): a lay-by for parents along a
/// footway, then a one-way loop round a planted island with minibus bays in yellow boxes, zebra
/// crossings and lane arrows; at the top tier a security hut, an entry barrier and CCTV (plot
/// local, the long side along x, entry at -x).
fn drop_off(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize, _opts: &Value) {
    let t = tier_spec(cat, "drop_off", tier).cloned().unwrap_or(Value::Null);
    cover(o, p, "surface", 1.0, mats::named("Site_Asphalt"));
    // turned round when the road is at +x, so the way in faces it
    let flip = p.front.x > 0.5;
    let f = if flip { p.frame.sub(0.0, 0.0, PI) } else { p.frame };
    let v = DVec2::new;
    let (hl, hd) = ((p.w - 2.0) / 2.0, (p.d - 2.0) / 2.0);
    let z = COVER_Z + 0.004;
    let (mut paint, mut yellow, mut kerb, mut paving, mut beds) = (MeshAcc::new(), MeshAcc::new(), MeshAcc::new(), MeshAcc::new(), MeshAcc::new());
    // kerbed footways along both long edges
    for s in [-1.0, 1.0] {
        kernel::ground(&mut paving, &kernel::rect(2.0 * hl, 2.5, 0.0, s * (hd - 1.25)), COVER_Z + 0.12, 0);
        strip(&mut kerb, &[v(-hl, s * (hd - 2.5)), v(hl, s * (hd - 2.5))], 0.2, COVER_Z + 0.125, 0, None);
    }
    o.part_at(&f, &format!("{}:footway", p.id), paving, vec![mats::named("Plaza_Paving")], 0.0);
    // the parents' lay-by: 6 m bays along the north footway, cars stopping in some of them
    let yb = hd - 2.5 - 2.4;
    let nb = ((2.0 * hl - 24.0).max(6.0) / 6.0) as usize;
    let x0 = -(nb as f64) * 3.0;
    strip(&mut paint, &[v(x0, yb), v(x0 + nb as f64 * 6.0, yb)], 0.1, z, 0, None);
    for k in 0..=nb {
        let x = x0 + k as f64 * 6.0;
        strip(&mut paint, &[v(x, yb), v(x, hd - 2.6)], 0.1, z, 0, None);
    }
    let mut lay: Vec<(DVec2, f64)> = (0..nb).map(|k| (v(x0 + (k as f64 + 0.5) * 6.0, yb + 1.2), -FRAC_PI_2)).collect();
    if flip {
        lay = lay.into_iter().map(|(c, a)| (-c, a + PI)).collect();
    }
    park_dir(o, p, &lay, num(&t, "cars", 0.4), false, 620, false);
    let iw = 8.0;
    if tier >= 2 {
        // the island: kerbed bed with trees, the loop round it one way (east on the north lane)
        let il = 2.0 * hl - 36.0;
        let r = kernel::rect(il, iw, 0.0, 0.0);
        strip(&mut kerb, &[r[0], r[1], r[2], r[3], r[0]], 0.25, COVER_Z + 0.125, 0, None);
        kernel::ground(&mut beds, &kernel::rect(il - 0.4, iw - 0.4, 0.0, 0.0), COVER_Z + 0.11, 0);
        let n = (il / 12.0).max(1.0) as usize;
        for k in 0..n {
            let x = -il / 2.0 + (k as f64 + 0.5) * il / n as f64;
            if tier >= 3 && k == 0 {
                continue; // the security hut's place
            }
            landscape::tree(o, "prop.tree_birch", f.pt(x, 0.0), COVER_Z + 0.11, p.seed + 640, k as u64);
            o.prop_at(&f, "prop.shrub", x + il / n as f64 / 2.0 - 1.5, 1.6, COVER_Z + 0.11, uniform(p.seed + 641, k as u64) * TAU, &[]);
        }
        for (sx, yl) in [(1.0, (iw / 2.0 + yb) / 2.0), (-1.0, -(iw / 2.0 + hd - 2.5) / 2.0)] {
            for x in [-il / 4.0, il / 4.0] {
                arrow(&mut paint, x, yl, sx, z);
            }
        }
        // zebra crossings from each end of the island to both footways
        for x in [-il / 2.0 + 3.0, il / 2.0 - 3.0] {
            for (y_a, y_b) in [(iw / 2.0, yb), (-(hd - 2.5), -iw / 2.0)] {
                let mut y = y_a + 0.3;
                while y + 0.5 <= y_b - 0.2 {
                    kernel::ground(&mut paint, &kernel::rect(2.4, 0.5, x, y + 0.25), z, 0);
                    y += 1.0;
                }
            }
        }
        // minibus bays on the south side, boxed in yellow, the club's minibuses in some
        let yc = -(hd - 2.5 - 1.75);
        let club = o.club.primary.clone();
        for k in 0..3u64 {
            let x = -13.0 + k as f64 * 13.0;
            let q = kernel::rect(10.0, 3.3, x, yc);
            strip(&mut yellow, &[q[0], q[1], q[2], q[3], q[0]], 0.12, z, 0, None);
            if uniform(p.seed + 650, k) < 0.7 {
                o.prop_at(&f, "prop.minibus", x, yc, COVER_Z, if k % 2 == 0 { -FRAC_PI_2 } else { FRAC_PI_2 }, &[("TINT_van_paint", &club)]);
            }
        }
        for x in [-il / 2.0 + 8.0, il / 2.0 - 8.0] {
            o.prop_at(&f, "prop.lamp_post", x, 0.0, COVER_Z + 0.11, 0.0, &[]);
            if o.night {
                let q = f.pt(x, 1.2);
                o.spot(DVec3::new(q.x, q.y, 7.6), DVec3::new(0.0, 0.0, -1.0), 400.0, 120.0, DVec3::new(1.0, 0.86, 0.7));
            }
        }
        for x in [-6.0, 6.0] {
            o.prop_at(&f, "prop.bench", x, hd - 1.0, COVER_Z + 0.12, PI, &[]);
        }
        o.prop_at(&f, "prop.planter", 0.0, hd - 1.0, COVER_Z + 0.12, 0.0, &[]);
        o.prop_at(&f, "prop.sign_post", -hl + 3.0, hd - 2.0, COVER_Z + 0.12, -FRAC_PI_2, &[]);
    }
    if tier >= 3 {
        // controlled entry: a security hut on the island nose, a barrier across the way in, CCTV
        // (on the island's nose, past the zebra's landing; the arm swings across the way in)
        let il = 2.0 * hl - 36.0;
        o.prop_at(&f, "prop.security_hut", -il / 2.0 + 6.5, 0.6, COVER_Z + 0.11, 0.0, &[]);
        o.prop_at(&f, "prop.barrier_arm", -il / 2.0 + 9.5, iw / 2.0 - 0.5, COVER_Z + 0.11, FRAC_PI_2, &[]);
        for (x, y) in [(-hl + 1.0, hd - 0.6), (hl - 1.0, hd - 0.6), (-hl + 1.0, -hd + 0.6), (hl - 1.0, -hd + 0.6)] {
            o.prop_at(&f, "prop.cctv_pole", x, y, COVER_Z + 0.12, (-x).atan2(y), &[]);
        }
        for k in 0..3 {
            o.prop_at(&f, "prop.bike_rack", hl - 10.0 + k as f64, -hd + 1.2, COVER_Z + 0.12, 0.0, &[]);
        }
    }
    o.part_at(&f, &format!("{}:paint", p.id), paint, vec![mats::tint("TINT_road_paint", "#ecebe4")], 0.0);
    o.part_at(&f, &format!("{}:yellow", p.id), yellow, vec![mats::tint("TINT_road_paint", "#e3b92a")], 0.0);
    o.part_at(&f, &format!("{}:kerb", p.id), kerb, vec![mats::named("Site_Kerb")], 0.0);
    o.part_at(&f, &format!("{}:beds", p.id), beds, vec![mats::lawn("amenity", f.pt(0.0, 0.0), f.yaw)], 0.0);
}

fn car_park(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize, opts: &Value) {
    let t = tier_spec(cat, "car_park", tier).cloned().unwrap_or(Value::Null);
    cover(o, p, "surface", 1.0, mats::named(text(&t, "surface", "Site_Asphalt")));
    let b = |k: &str| t.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    let mut paint = MeshAcc::new();
    let mut spots = bays(p.w, p.d, 2.2, &mut paint, COVER_Z + 0.004, &|_| tier == 1);
    if tier >= 2 {
        o.part_at(&p.frame, &format!("{}:paint", p.id), paint, vec![mats::tint("TINT_road_paint", "#ecebe4")], 0.0);
        // kerb round the edge
        let mut k = MeshAcc::new();
        let r = kernel::rect(p.w - 2.0, p.d - 2.0, 0.0, 0.0);
        strip(&mut k, &[r[0], r[1], r[2], r[3], r[0]], 0.25, COVER_Z + 0.1, 0, None);
        o.part_at(&p.frame, &format!("{}:kerb", p.id), k, vec![mats::named("Site_Kerb")], 0.0);
    } else {
        // timber edging and bollards on gravel
        let r = kernel::rect(p.w - 2.0, p.d - 2.0, 0.0, 0.0);
        for (k, (q, _)) in kernel::along(&[r[0], r[1], r[2], r[3], r[0]], 4.0, 0.5).into_iter().enumerate() {
            if k % 2 == 0 {
                o.prop_at(&p.frame, "prop.bollard", q.x, q.y, COVER_Z, 0.0, &[]);
            }
        }
    }
    let mut islands = MeshAcc::new();
    if b("islands") {
        // kerbed planting beds at the ends of each row of bays (and one every 12 bays along a long
        // row): a tree, or under a solar canopy low shrubs
        let mut beds = MeshAcc::new();
        let mut kept = Vec::new();
        let mut rows: Vec<(i64, Vec<usize>)> = Vec::new();
        for (i, s) in spots.iter().enumerate() {
            let key = (s.0.y * 10.0).round() as i64;
            match rows.iter_mut().find(|r| r.0 == key) {
                Some(r) => r.1.push(i),
                None => rows.push((key, vec![i])),
            }
        }
        let mut bed = std::collections::HashSet::new();
        for (_, idx) in &rows {
            if idx.len() >= 6 {
                bed.insert(idx[0]);
                bed.insert(idx[idx.len() - 1]);
                let mut k = 12;
                while k < idx.len() - 6 {
                    bed.insert(idx[k]);
                    k += 12;
                }
            }
        }
        for (i, s) in spots.iter().enumerate() {
            if bed.contains(&i) {
                let (cx, cy) = (s.0.x, s.0.y);
                for (x0, y0, x1, y1) in [(cx - 1.2, cy - 2.4, cx + 1.2, cy - 2.25), (cx - 1.2, cy + 2.25, cx + 1.2, cy + 2.4), (cx - 1.2, cy - 2.25, cx - 1.05, cy + 2.25), (cx + 1.05, cy - 2.25, cx + 1.2, cy + 2.25)] {
                    bx(&mut islands, x0, y0, COVER_Z, x1, y1, COVER_Z + 0.15, 0);
                }
                kernel::ground(&mut beds, &kernel::rect(2.1, 4.5, cx, cy), COVER_Z + 0.11, 0);
                if b("canopy") {
                    for k in [-1.0f64, 1.0] {
                        let yaw = uniform(p.seed + 601, 2 * i as u64 + if k > 0.0 { 1 } else { 0 }) * 6.283;
                        o.prop_at(&p.frame, "prop.shrub", cx, cy + k * 1.15, COVER_Z + 0.11, yaw, &[]);
                    }
                } else {
                    landscape::tree(o, "prop.tree_birch", p.frame.pt(cx, cy), COVER_Z + 0.11, p.seed + 600, i as u64);
                }
            } else {
                kept.push(*s);
            }
        }
        spots = kept;
        o.part_at(&p.frame, &format!("{}:islands", p.id), islands, vec![mats::named("Site_Kerb")], 0.0);
        o.part_at(&p.frame, &format!("{}:beds", p.id), beds, vec![mats::named("Site_Dirt")], 0.0);
    }
    if b("ev") {
        let mut ev = MeshAcc::new();
        for s in spots.iter().take(6) {
            kernel::ground(&mut ev, &kernel::rect(2.3, 4.8, s.0.x, s.0.y), COVER_Z + 0.005, 0);
            let back = DVec2::new(0.0, if s.1 == 0.0 { -2.7 } else { 2.7 });
            o.prop_at(&p.frame, "prop.ev_charger", s.0.x + back.x, s.0.y + back.y, COVER_Z, s.1, &[]);
        }
        o.part_at(&p.frame, &format!("{}:ev_bays", p.id), ev, vec![mats::tint("TINT_road_paint", "#2f8a4f")], 0.0);
    }
    let fill = option(cat, "car_park", tier, opts, "fill").as_f64().unwrap_or(0.6).clamp(0.0, 1.0);
    park(o, p, &spots, fill, false, 610);
    o.parking += spots.len();
    if b("lamps") {
        let n = (p.w / 24.0).floor().max(1.0) as usize;
        for i in 0..n {
            let x = -p.w / 2.0 + (i as f64 + 0.5) * p.w / n as f64;
            for y in [-p.d / 2.0 + 1.4, p.d / 2.0 - 1.4] {
                o.prop_at(&p.frame, "prop.lamp_post", x, y, COVER_Z, if y < 0.0 { 0.0 } else { PI }, &[]);
                if o.night {
                    let q = p.frame.pt(x, y + if y < 0.0 { 1.2 } else { -1.2 });
                    o.spot(DVec3::new(q.x, q.y, 7.6), DVec3::new(0.0, 0.0, -1.0), 400.0, 120.0, DVec3::new(1.0, 0.86, 0.7));
                }
            }
        }
        o.prop_at(&p.frame, "prop.ticket_machine", p.w / 2.0 - 3.0, -p.d / 2.0 + 2.0, COVER_Z, 0.0, &[]);
    }
    if b("canopy") {
        // solar canopies over the bay rows: columns and an inclined deck of panels
        let mut deck = MeshAcc::new();
        let mut cols: Vec<Polyline> = Vec::new();
        let ys: Vec<f64> = {
            let mut v: Vec<f64> = spots.iter().map(|s| (s.0.y * 10.0).round() / 10.0).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.dedup();
            v
        };
        let mut seen = std::collections::HashSet::new();
        for y in ys {
            let row: Vec<&(DVec2, f64)> = spots.iter().filter(|s| (s.0.y - y).abs() < 0.2).collect();
            let x0 = row.iter().map(|s| s.0.x).fold(f64::MAX, f64::min) - 1.25;
            let x1 = row.iter().map(|s| s.0.x).fold(f64::MIN, f64::max) + 1.25;
            quad(&mut deck, DVec3::new(x0, y - 2.6, 3.0), DVec3::new(x1, y - 2.6, 3.0), DVec3::new(x1, y + 2.6, 3.7), DVec3::new(x0, y + 2.6, 3.7), 0);
            // columns at the back of the bays, on the lines between cars (every third), never in a bay
            let yb = if row[0].1 == 0.0 { y - 2.45 } else { y + 2.45 };
            let zb = 3.0 + 0.7 * (yb - (y - 2.6)) / 5.2;
            let n = ((x1 - x0) / 7.5).round().max(1.0) as usize;
            let mut xs: Vec<f64> = (0..n).map(|j| x0 + (j as f64 * 7.5).min(x1 - x0)).collect();
            xs.push(x1);
            for x in xs {
                cols.push((vec![DVec3::new(x, y - 2.6, 2.95), DVec3::new(x, y + 2.6, 3.65)], 0.06));
                if seen.insert(((x * 10.0).round() as i64, (yb * 10.0).round() as i64)) {
                    cols.push((vec![DVec3::new(x, yb, COVER_Z), DVec3::new(x, yb, zb - 0.08)], 0.11));
                }
            }
            cols.push((vec![DVec3::new(x0, yb, zb - 0.1), DVec3::new(x1, yb, zb - 0.1)], 0.07));
        }
        o.part_at(&p.frame, &format!("{}:canopy", p.id), deck, vec![mats::named("Solar_Panel")], 0.0);
        o.tubes_at(&p.frame, &format!("{}:canopy_steel", p.id), cols, mats::named("Galvanised"), 8);
    }
}

fn maintenance(o: &mut Out, cat: &Catalog, p: &Plot, tier: usize, style: &str) {
    let t = tier_spec(cat, "maintenance", tier).cloned().unwrap_or(Value::Null);
    cover(o, p, "yard", 1.0, mats::named(if tier == 1 { "Site_Gravel" } else { "Apron_Concrete" }));
    let mut heaps = MeshAcc::new();
    let mut sand = MeshAcc::new();
    let (f, w, d) = if tier == 1 {
        let f = place(p, 14.0, 6.0, 6.0).f;
        for i in 0..2 {
            o.prop_at(&f, "prop.storage_container", -3.5 + i as f64 * 7.0, 0.0, COVER_Z, FRAC_PI_2 * 0.0, &[("TINT_container", "#3c5a3a")]);
        }
        (f, 14.0, 6.0)
    } else {
        let size = &t["size"];
        let pl = place(p, size[0].as_f64().unwrap_or(20.0), size[1].as_f64().unwrap_or(12.0), 9.0);
        let (f, w, d) = (pl.f, pl.w, pl.d);
        let mut sp = buildings::styled(cat, if style == "ultra_modern" { "industrial" } else { style }, w, d, 1, &o.club.primary.clone(), o.night, &Value::Null, p.seed);
        sp.storey_h = 5.6;
        sp.window = "strip".into();
        sp.roller_doors = if tier == 3 { 3 } else { 2 };
        sp.door = "none".into();
        sp.canopy = "none".into();
        sp.clock = false;
        sp.arcade = false;
        sp.quoins = false;
        sp.mats[buildings::T_DOOR as usize] = mats::tint("TINT_door", "#6e7377");
        sp.roof = if matches!(sp.roof.as_str(), "flat" | "mono") { sp.roof.clone() } else { "shallow".into() };
        buildings::build(o, &f, &format!("{}:shed", p.id), &sp);
        if tier == 3 {
            // covered bays beside the depot
            let cf = f.sub(w / 2.0 + 6.0, 0.0, 0.0);
            let mut c = MeshAcc::new();
            bx(&mut c, -5.0, -d / 2.0, 4.5, 5.0, d / 2.0, 4.7, 0);
            o.part_at(&cf, &format!("{}:bays_roof", p.id), c, vec![mats::tint("TINT_roof", "#5d6166")], 0.0);
            let mut cols: Vec<Polyline> = Vec::new();
            for x in [-4.8, 0.0, 4.8] {
                for y in [-d / 2.0 + 0.2, d / 2.0 - 0.2] {
                    cols.push((vec![DVec3::new(x, y, COVER_Z), DVec3::new(x, y, 4.5)], 0.1));
                }
            }
            o.tubes_at(&cf, &format!("{}:bays_steel", p.id), cols, mats::named("Galvanised"), 8);
            o.prop_at(&cf, "prop.tractor", 0.0, 0.0, COVER_Z, FRAC_PI_2, &[("TINT_tractor", "#2f6b2f")]);
        }
        (f, w, d)
    };
    // the yard in front: mowers, sand and topsoil, a fuel tank
    let yard = f.sub(0.0, -d / 2.0 - 5.5, 0.0);
    o.prop_at(&yard, "prop.ride_mower", -w / 2.0 + 2.0, 0.0, COVER_Z, 0.4, &[]);
    if tier >= 2 {
        o.prop_at(&yard, "prop.ride_mower", -w / 2.0 + 4.5, 0.3, COVER_Z, 0.5, &[]);
        o.prop_at(&yard, "prop.tractor", w / 2.0 - 3.0, -0.5, COVER_Z, -0.3, &[("TINT_tractor", "#2f6b2f")]);
        o.prop_at(&f, "prop.fuel_tank", -w / 2.0 - 2.5, d / 2.0 - 2.0, COVER_Z, 0.0, &[]);
    }
    // the groundsman's hand kit along the front of the shed
    o.prop_at(&yard, "genprop.line_marker", -2.5, 3.9, COVER_Z, 0.3, &[]);
    o.prop_at(&yard, "genprop.wheelbarrow_tools", -0.6, 4.0, COVER_Z, -0.2, &[]);
    o.prop_at(&yard, "genprop.hose_reel", 1.4, 4.4, COVER_Z, 0.0, &[]);
    o.prop_at(&yard, "genprop.divot_bucket", 2.7, 4.5, COVER_Z, 0.0, &[]);
    if tier >= 2 {
        o.prop_at(&yard, "genprop.cylinder_mower", 4.6, 4.0, COVER_Z, 0.2, &[]);
        o.prop_at(&yard, "genprop.pitch_sprinkler", -4.4, 4.2, COVER_Z, 0.0, &[]);
    }
    mound(&mut sand, 2.0, -1.0, 2.6, 2.0, 1.3, p.seed + 700, 0);
    mound(&mut heaps, 6.5, -1.0, 2.4, 2.0, 1.2, p.seed + 701, 0);
    o.part_at(&yard, &format!("{}:sand", p.id), sand, vec![mats::named("Site_Sand")], 75.0);
    o.part_at(&yard, &format!("{}:topsoil", p.id), heaps, vec![mats::named("Site_Dirt")], 75.0);
    // a compound fence round the yard (containers tier)
    if tier == 1 {
        let (hw, hd) = (p.w / 2.0 - 1.5, p.d / 2.0 - 1.5);
        let mid = DVec2::new(p.front.x * hw, p.front.y * hd);
        let tang = DVec2::new(-p.front.y, p.front.x);
        let corners = [DVec2::new(-hw, -hd), DVec2::new(hw, -hd), DVec2::new(hw, hd), DVec2::new(-hw, hd)];
        let start = corners.iter().position(|c| (*c - mid).dot(tang) > 0.0 && (*c - mid).dot(p.front) > -0.1).unwrap_or(0);
        let mut ring = vec![mid + tang * 3.0];
        for k in 0..4 {
            ring.push(corners[(start + k) % 4]);
        }
        ring.push(mid - tang * 3.0);
        let world: Vec<DVec2> = ring.iter().map(|q| p.frame.pt(q.x, q.y)).collect();
        landscape::chainlink(o, &world, 2.4, &format!("{}:fence", p.id));
    }
    let _ = obox;
}
