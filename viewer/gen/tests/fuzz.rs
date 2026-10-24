//! Configuration fuzzer: random layouts across the whole option space (seeded, reproducible),
//! each built by the generator and checked against geometric invariants — nothing buried or
//! floating, nothing on the field of play, furniture clear of the stands, lights on structure.
//!
//!   cargo test --release -p stadium-gen --test fuzz              (60 layouts)
//!   FUZZ_N=400 cargo test --release -p stadium-gen --test fuzz   (a longer sweep)
//! A failure prints the layout JSON so it can be opened in the editor.

use glam::{DVec2, DVec3};
use serde_json::{json, Value};
use stadium_gen::catalog::Catalog;
use stadium_gen::prng::uniform;
use stadium_gen::stadium::{self, Stadium};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Rng {
    seed: u64,
    i: u64,
}

impl Rng {
    fn f(&mut self) -> f64 {
        self.i += 1;
        uniform(self.seed, self.i)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
    fn pick<'a, T>(&mut self, v: &'a [T]) -> &'a T {
        &v[((self.f() * v.len() as f64) as usize).min(v.len() - 1)]
    }
    fn chance(&mut self, p: f64) -> bool {
        self.f() < p
    }
    fn color(&mut self) -> String {
        format!("#{:02x}{:02x}{:02x}", (self.f() * 255.0) as u8, (self.f() * 255.0) as u8, (self.f() * 255.0) as u8)
    }
}

fn keys(m: &serde_json::Map<String, Value>) -> Vec<String> {
    m.keys().cloned().collect()
}

/// A random layout inside the editor's ranges, extremes included.
fn random_layout(cat: &Catalog, seed: u64) -> Value {
    let mut r = Rng { seed, i: 0 };
    let ext = |r: &mut Rng, lo: f64, hi: f64| if r.chance(0.3) { *r.pick(&[lo, hi]) } else { r.range(lo, hi) };
    let mut l = json!({"schema": "stadium-layout/1", "name": format!("fuzz {seed}")});
    l["lighting"] = json!(if r.chance(0.3) { "night" } else { "day" });
    l["pitch"] = json!({
        "length": (ext(&mut r, 90.0, 120.0) * 2.0).round() / 2.0,
        "width": (ext(&mut r, 45.0, 90.0) * 2.0).round() / 2.0,
        "grass": r.pick(&keys(&cat.grass)).clone(),
        "mow": {"pattern": *r.pick(&["none", "stripes", "stripes_long", "checkerboard", "diagonal", "diamond", "circles"]), "width": ext(&mut r, 1.5, 12.0), "angle": ext(&mut r, -90.0, 90.0)},
        "wear": r.f(), "wetness": r.f(), "hand_painted": r.f(), "crown": ext(&mut r, 0.0, 0.3),
    });
    l["goals"] = json!({"style": r.pick(&keys(&cat.goals)).clone(), "net_pattern": *r.pick(&["solid", "bands", "checker"]), "mesh": *r.pick(&["hex", "square"]), "net_colors": [r.color(), r.color()]});
    l["corner_flags"] = json!({"preset": r.pick(&keys(&cat.corner_flags)).clone()});
    let profiles = cat.profile_order.clone();
    let seats = keys(&cat.seats);
    let roofs = keys(&cat.roofs);
    let facades = keys(&cat.facades);
    if r.chance(0.85) {
        let curved = r.chance(0.35);
        let mut sides = serde_json::Map::new();
        let bowl = r.pick(&profiles).clone();
        for side in ["south", "north", "east", "west"] {
            if !curved && r.chance(0.25) {
                continue;
            }
            let mut sd = json!({"profile": if curved { bowl.clone() } else { r.pick(&profiles).clone() }, "seat": r.pick(&seats).clone()});
            if !curved && r.chance(0.3) {
                sd["length"] = json!(ext(&mut r, 10.0, 140.0).round());
            }
            if !curved && r.chance(0.4) {
                sd["roof"] = if r.chance(0.2) { json!(false) } else { json!({"preset": r.pick(&roofs).clone()}) };
            }
            if !curved && r.chance(0.4) {
                sd["facade"] = if r.chance(0.2) { json!(false) } else { json!({"preset": r.pick(&facades).clone()}) };
            }
            sides.insert(side.into(), sd);
        }
        if sides.is_empty() {
            sides.insert("south".into(), json!({"profile": "single_tier_20"}));
        }
        let mut st = json!({
            "corners": if curved { "curved" } else { "open" },
            "corner_radius": ext(&mut r, 6.0, 40.0), "front_gap": ext(&mut r, 1.0, 12.0),
            "apron": *r.pick(&["turf", "concrete", "track"]), "sides": Value::Object(sides),
            "colors": {"pattern": *r.pick(&["solid", "tiers", "rows", "bays", "gradient", "random"]), "colors": [r.color(), r.color(), r.color()]},
        });
        if r.chance(0.7) {
            st["roof"] = json!({"preset": r.pick(&roofs).clone()});
        }
        if r.chance(0.7) {
            st["facade"] = json!({"preset": r.pick(&facades).clone()});
        }
        if r.chance(0.4) {
            let text = *r.pick(&["KIT FC", "THE GREATEST FOOTBALL CLUB IN THE WHOLE WIDE WORLD", "àé!? 1-0 ÄÖ", "COME ON YOU REDS"]);
            st["mosaics"] = json!([{"run": *r.pick(&["south", "north", "east", "west"]), "tier": (r.f() * 3.0) as u32, "text": text}]);
        }
        if curved && r.chance(0.1) {
            // hand-edited: a bowl with a side missing falls back to open corners
            st["sides"].as_object_mut().unwrap().remove("north");
        }
        l["stands"] = st;
    }
    if r.chance(0.8) {
        l["floodlights"] = json!({"preset": r.pick(&keys(&cat.floodlights)).clone()});
    }
    if r.chance(0.8) {
        l["ads"] = json!({"perimeter": {"type": *r.pick(&["led", "static", "none"])}, "carpets": r.chance(0.5)});
        if r.chance(0.4) {
            l["ads"]["ribbons"] = json!({"tiers": [1, 2, 3]});
        }
        if r.chance(0.2) {
            l["ads"]["perimeter"]["end_offset"] = json!(ext(&mut r, 3.0, 12.0));
        }
    }
    if r.chance(0.7) {
        l["screens"] = json!({"placement": *r.pick(&["ends", "corners", "center", "stand_top"]), "width": ext(&mut r, 6.0, 40.0), "height": ext(&mut r, 3.0, 16.0)});
    }
    if r.chance(0.7) {
        l["dugouts"] = json!({"style": *r.pick(&["shell", "box", "shelter"]), "seats": ext(&mut r, 4.0, 30.0).round(), "seat": r.pick(&seats).clone()});
    }
    if r.chance(0.6) {
        l["tunnel"] = json!({"style": *r.pick(&["telescopic", "portal"])});
    }
    // hand-edited files: unknown ids and colours warn and fall back, they never fail the build
    if r.chance(0.15) {
        let k = (r.f() * 6.0) as usize;
        match k {
            0 => l["goals"]["style"] = json!("nope"),
            1 => l["corner_flags"]["preset"] = json!("nope"),
            2 => l["pitch"]["grass"] = json!("astroturf"),
            3 => l["goals"]["net_colors"] = json!(["white", "#zzzzzz"]),
            4 => l["floodlights"] = json!({"preset": "nope"}),
            _ => l["screens"] = json!({"placement": "center"}),
        }
    }
    l
}

struct Bounds {
    lo: DVec3,
    hi: DVec3,
}

fn bounds(st: &Stadium, name_prefix: &str) -> Option<Bounds> {
    let mut b: Option<Bounds> = None;
    for p in st.parts.iter().filter(|p| p.name.starts_with(name_prefix)) {
        for v in &p.mesh.verts {
            let e = b.get_or_insert(Bounds { lo: *v, hi: *v });
            e.lo = e.lo.min(*v);
            e.hi = e.hi.max(*v);
        }
    }
    b
}

/// Every invariant violation of one built stadium.
fn check(st: &Stadium) -> Vec<String> {
    let mut bad = Vec::new();
    let p = &st.pitch;
    let (hl, hw) = (p.length / 2.0, p.width / 2.0);
    for part in &st.parts {
        if part.mesh.verts.iter().any(|v| !v.is_finite()) {
            bad.push(format!("{}: non-finite vertex", part.name));
        }
        // nothing but the pitch, its paint and carpets lies below ground level
        let lowest = part.mesh.verts.iter().map(|v| v.z).fold(f64::MAX, f64::min);
        // steel tubes from the ground run a little into their footings
        let tube = ["Steel", "Masts", "Rails", "Frame", "Hoops", "Legs"].iter().any(|k| part.name.contains(k));
        let floor = if tube { -0.35 } else { -0.05 };
        if lowest < floor && !matches!(part.name.as_str(), "Plaza" | "Apron") {
            bad.push(format!("{}: {lowest:.2} m below ground", part.name));
        }
    }
    // big screens: well clear of the ground and the crowd
    for part in st.parts.iter().filter(|p| p.name.starts_with("Screen_")) {
        let bottom = part.mesh.verts.iter().map(|v| v.z).fold(f64::MAX, f64::min);
        if bottom < 2.0 {
            bad.push(format!("{}: bottom {bottom:.2} m above ground", part.name));
        }
    }
    // the field of play stays clear (except goals, flags, lines and the pitch itself)
    for part in &st.parts {
        let allowed = ["PitchSurface", "PitchLines", "Apron", "Plaza", "TechnicalAreas"];
        if allowed.contains(&part.name.as_str()) || part.name.starts_with("Screen_centre") || part.name == "ScreenCaps" || part.name == "ScreenSteel" {
            continue;
        }
        let on = part.mesh.verts.iter().filter(|v| v.x.abs() < hl - 0.1 && v.y.abs() < hw - 0.1 && v.z < 2.5).count();
        if on > 0 {
            bad.push(format!("{}: {on} vertices on the field of play", part.name));
        }
    }
    // furniture in front of the stands (stand front line at x = ±A, y = ±B)
    if st.report.footprint.0 > 0.0 {
        let (a, b) = st.report.footprint;
        for name in ["BoardHousing", "Boards", "BoardBacks", "Dugout_", "Tunnel", "Carpet_"] {
            if let Some(bb) = bounds(st, name) {
                // only the front zone counts: the tunnel mouth sits in the stand front on purpose
                let lim = if name == "Tunnel" { 0.05 } else { -0.05 };
                // only sides that have a stand have a front to run into
                let has = |side: &str| st.parts.iter().any(|p| p.name == format!("Stand_{side}"));
                let near_y = (bb.lo.y < -(b - lim) && has("south")) || (bb.hi.y > b - lim && has("north"));
                let near_x = (bb.lo.x < -(a - lim) && has("west")) || (bb.hi.x > a - lim && has("east"));
                let inside_footprint = (bb.lo.x.abs().min(bb.hi.x.abs()) < a) && (bb.lo.y.abs().min(bb.hi.y.abs()) < b);
                if inside_footprint && ((near_y && name != "Tunnel") || near_x) {
                    bad.push(format!("{name}: reaches the stand front (x {:.1}..{:.1}, y {:.1}..{:.1}; front at ±{a:.1}, ±{b:.1})", bb.lo.x, bb.hi.x, bb.lo.y, bb.hi.y));
                }
            }
        }
    }
    // floodlights: up high, aimed down at the pitch; roof fixtures on a roof edge
    if let Some(d) = &st.report.lights {
        for (pos, tgt) in &d.fixtures {
            if pos.z < 5.0 {
                bad.push(format!("floodlight fixture at {:.1} m", pos.z));
                break;
            }
            if tgt.z >= pos.z {
                bad.push("floodlight aimed upwards".into());
                break;
            }
        }
        if d.kind == "roof" {
            let edges: Vec<DVec3> = st.sockets.iter().filter(|s| s.kind == "roof_edge").map(|s| s.loc).collect();
            if edges.is_empty() {
                bad.push("roof floodlights without a roof edge".into());
            }
        }
    }
    // the main camera is up on a gantry, unless a low roof caps it
    if st.parts.iter().any(|p| p.name == "Stand_south") && st.camera.0.z < stadium::CAMERA_MIN_HEIGHT - 2.0 - 1e-6 {
        bad.push(format!("main camera only {:.1} m up", st.camera.0.z));
    }
    // carpets: clear of the goal nets and in front of the end boards
    let hl = st.pitch.length / 2.0;
    for part in st.parts.iter().filter(|p| p.name.starts_with("Carpet_")) {
        if part.mesh.verts.iter().any(|v| v.x.abs() > hl && v.x.abs() < hl + 2.0 && v.y.abs() < 3.66 + 0.15) {
            bad.push(format!("{}: under the goal net", part.name));
        }
        let limit = hl + st.report.end_boards_m - 0.4 + 1e-3;
        if part.mesh.verts.iter().any(|v| v.x.abs() > limit) {
            bad.push(format!("{}: runs under the end boards", part.name));
        }
    }
    // corner pylons stand clear of the stands and roofs
    if let Some(d) = st.report.lights.as_ref().filter(|d| d.kind == "corner_pylons") {
        let solid: Vec<DVec2> = st.parts.iter().filter(|p| p.name.starts_with("Stand_") || p.name.starts_with("Roof_")).flat_map(|p| p.mesh.verts.iter().map(|v| v.truncate())).collect();
        for m in d.fixtures.iter().step_by(35) {
            let near = solid.iter().map(|q| q.distance(m.0.truncate())).fold(f64::MAX, f64::min);
            if near < 1.0 {
                bad.push(format!("floodlight mast head {:.1} m from a stand or roof", near));
                break;
            }
        }
    }
    for w in &st.report.warnings {
        if w.contains("mosaic") && w.contains("could not") {
            bad.push(w.clone());
        }
    }
    bad
}

fn run(n: u64, start: u64) {
    let cat = Catalog::load(&root().join("catalog")).unwrap();
    let mut failures = Vec::new();
    for seed in start..start + n {
        let layout = random_layout(&cat, seed);
        let built = std::panic::catch_unwind(|| stadium::build(&layout, &cat));
        let Ok(built) = built else {
            failures.push((seed, vec!["generator panicked".into()], layout));
            continue;
        };
        match built {
            Ok(st) => {
                let bad = check(&st);
                if !bad.is_empty() {
                    failures.push((seed, bad, layout));
                }
            }
            Err(e) => {
                // an error is fine only for genuinely impossible requests
                if !(e.contains("centre-hung screen needs roofs") || e.contains("roof floodlights need roofs")) {
                    failures.push((seed, vec![format!("build error: {e}")], layout));
                }
            }
        }
    }
    if !failures.is_empty() {
        let dir = std::env::temp_dir().join("stadium_fuzz");
        let _ = std::fs::create_dir_all(&dir);
        let mut msg = format!("{} of {n} layouts break invariants (layouts in {}):\n", failures.len(), dir.display());
        for (seed, bad, layout) in &failures {
            let _ = std::fs::write(dir.join(format!("fuzz_{seed}.json")), serde_json::to_string_pretty(layout).unwrap());
            msg += &format!("  seed {seed}: {}\n", bad.join("; "));
        }
        panic!("{msg}");
    }
}

#[test]
fn fuzz_layouts() {
    let n = std::env::var("FUZZ_N").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let start = std::env::var("FUZZ_START").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    run(n, start);
}

