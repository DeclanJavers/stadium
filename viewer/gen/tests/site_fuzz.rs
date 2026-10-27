//! Training-ground fuzzer: random site layouts across the whole option space (seeded,
//! reproducible), junk values included, each built by the generator and checked against
//! invariants — it builds, every number is finite, meshes are well formed, props stand on the
//! ground inside the site and off every field of play, pitches never overlap, the report agrees
//! with what was built, and the same layout always builds the same site.
//!
//!   cargo test --release -p stadium-gen --test site_fuzz              (60 layouts)
//!   FUZZ_N=400 cargo test --release -p stadium-gen --test site_fuzz   (a longer sweep)
//! A failure prints the layout JSON so it can be opened in the editor.

use glam::{DVec2, DVec3};
use serde_json::{json, Value};
use stadium_gen::catalog::Catalog;
use stadium_gen::prng::uniform;
use stadium_gen::site::{self, Site, BOUNDARIES, SCREENING, STATES};
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
        if self.chance(0.05) {
            return "not-a-colour".into();
        }
        format!("#{:02x}{:02x}{:02x}", (self.f() * 255.0) as u8, (self.f() * 255.0) as u8, (self.f() * 255.0) as u8)
    }
}

fn keys(m: &serde_json::Map<String, Value>) -> Vec<String> {
    m.keys().filter(|k| !k.starts_with('_') && k.as_str() != "plot_kinds").cloned().collect()
}

/// A random site layout: every template, state, facility, tier, option and style, plus junk.
fn random_layout(cat: &Catalog, seed: u64) -> Value {
    let mut r = Rng { seed, i: 0 };
    let junk = |r: &mut Rng, v: Vec<String>| if r.chance(0.04) { "nonsense".to_string() } else { r.pick(&v).clone() };
    let templates = keys(&cat.sites);
    let styles = keys(&cat.styles);
    let facilities = keys(&cat.facilities);
    let occupants = keys(&cat.occupants);
    let template = r.pick(&templates).clone();
    let tpl = &cat.sites[&template];
    let mut l = json!({"schema": "site-layout/1", "name": format!("site fuzz {seed}"), "template": template, "seed": (r.f() * 1000.0) as u64});
    l["style"] = json!(junk(&mut r, styles.clone()));
    l["lighting"] = json!(if r.chance(0.3) { "night" } else { "day" });
    if r.chance(0.8) {
        l["club"] = json!({"name": *r.pick(&["KIT FC", "", "A VERY LONG CLUB NAME UNITED AND ATHLETIC"]), "primary": r.color(), "secondary": r.color()});
    }
    let b: Vec<String> = BOUNDARIES.iter().map(|s| s.to_string()).collect();
    let s: Vec<String> = SCREENING.iter().map(|s| s.to_string()).collect();
    l["boundary"] = json!({"type": junk(&mut r, b), "screening": junk(&mut r, s)});
    l["gate"] = json!({"tier": (r.f() * 5.0) as i64});

    // plots: each template plot (or a merge in its place), some left at their default
    let mut ids: Vec<(String, String)> = tpl["plots"].as_array().map(|a| a.iter().map(|p| (p["id"].as_str().unwrap_or("").to_string(), p["kind"].as_str().unwrap_or("").to_string())).collect()).unwrap_or_default();
    for m in tpl["merges"].as_array().cloned().unwrap_or_default() {
        if r.chance(0.4) {
            let parts: Vec<&str> = m["plots"].as_array().unwrap().iter().filter_map(|v| v.as_str()).collect();
            ids.retain(|(id, _)| !parts.contains(&id.as_str()));
            ids.push((m["id"].as_str().unwrap().to_string(), m["kind"].as_str().unwrap_or("").to_string()));
        }
    }
    let mut plots = serde_json::Map::new();
    for (id, _kind) in ids {
        if r.chance(0.1) {
            continue;
        }
        let state = if r.chance(0.03) { "nonsense".to_string() } else { r.pick(&STATES).to_string() };
        let mut p = json!({"state": state});
        if state == "built" || state == "construction" {
            let fac = junk(&mut r, facilities.clone());
            let tiers = cat.facilities.get(&fac).and_then(|f| f["tiers"].as_array()).map(|t| t.len()).unwrap_or(1);
            p["facility"] = json!(fac);
            p["tier"] = json!((r.f() * (tiers as f64 + 1.0)) as i64);
            let mut o = serde_json::Map::new();
            match fac.as_str() {
                "pitch" => {
                    if r.chance(0.5) {
                        o.insert("length".into(), json!(if r.chance(0.3) { *r.pick(&[90.0, 110.0, 40.0]) } else { r.range(90.0, 110.0) }));
                        o.insert("width".into(), json!(if r.chance(0.3) { *r.pick(&[55.0, 72.0, 200.0]) } else { r.range(55.0, 72.0) }));
                    }
                    if r.chance(0.5) {
                        o.insert("mow".into(), json!(*r.pick(&["none", "stripes", "stripes_long", "checkerboard", "diagonal", "diamond", "circles", "zigzag"])));
                    }
                    for k in ["floodlights", "ball_stop", "filming_tower", "viewing_deck", "shelters", "equipment"] {
                        if r.chance(0.5) {
                            o.insert(k.into(), json!(r.chance(0.5)));
                        }
                    }
                }
                "artificial_pitch" => {
                    for k in ["floodlights", "equipment"] {
                        if r.chance(0.5) {
                            o.insert(k.into(), json!(r.chance(0.5)));
                        }
                    }
                }
                "car_park" => {
                    if r.chance(0.7) {
                        o.insert("fill".into(), json!(r.range(-0.2, 1.2)));
                    }
                }
                "training_building" => {
                    let mods: Vec<&str> = ["gym", "rehab", "canteen", "media", "sauna"].into_iter().filter(|_| r.chance(0.5)).collect();
                    o.insert("modules".into(), json!(mods));
                    if r.chance(0.3) {
                        o.insert("sign".into(), json!(r.chance(0.5)));
                    }
                }
                _ => {}
            }
            p["options"] = Value::Object(o);
            if r.chance(0.2) {
                p["style"] = json!(junk(&mut r, styles.clone()));
            }
        } else if state == "occupied" || state == "demolition" {
            p["occupant"] = json!(junk(&mut r, occupants.clone()));
        }
        plots.insert(id, p);
    }
    l["plots"] = Value::Object(plots);
    l
}

/// Training kit that a session lays out on the grass (and the gulls that land on it).
const TRAINING_GEAR: [&str; 11] = [
    "prop.cone_disc", "prop.cone_tall", "prop.ball", "prop.agility_pole", "prop.agility_ladder", "prop.mannequin", "prop.mini_goal",
    "prop.rebounder", "genprop.ball_muddy", "genprop.gk_gloves", "genprop.seagull_standing",
];

fn finite(v: DVec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn inside(poly: &[DVec2], p: DVec2) -> bool {
    let mut c = false;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (b.x - a.x) * (p.y - a.y) / (b.y - a.y) {
            c = !c;
        }
    }
    c
}

fn dist_to_poly(poly: &[DVec2], p: DVec2) -> f64 {
    (0..poly.len())
        .map(|i| {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            let t = ((p - a).dot(b - a) / (b - a).length_squared().max(1e-12)).clamp(0.0, 1.0);
            (a + (b - a) * t).distance(p)
        })
        .fold(f64::INFINITY, f64::min)
}

/// A pitch's field of play in site coordinates: (centre, length axis, half length, half width).
fn field(p: &site::SitePitch) -> (DVec2, DVec2, f64, f64) {
    let l = p.params["length"].as_f64().unwrap_or(105.0);
    let w = p.params["width"].as_f64().unwrap_or(68.0);
    (p.origin, DVec2::new(p.yaw.cos(), p.yaw.sin()), l / 2.0, w / 2.0)
}

fn in_field(f: &(DVec2, DVec2, f64, f64), q: DVec2, inset: f64) -> bool {
    let d = q - f.0;
    let (u, v) = (d.dot(f.1), d.perp_dot(f.1));
    u.abs() < f.2 - inset && v.abs() < f.3 - inset
}

/// Everything wrong with one built site.
fn check(s: &Site) -> Vec<String> {
    let mut bad = Vec::new();
    for part in &s.parts {
        let m = &part.mesh;
        if m.verts.iter().any(|v| !finite(*v)) || m.uvs.iter().any(|v| !v.x.is_finite() || !v.y.is_finite()) {
            bad.push(format!("{}: non-finite vertex or uv", part.name));
        }
        if m.tags.len() != m.faces.len() {
            bad.push(format!("{}: {} tags for {} faces", part.name, m.tags.len(), m.faces.len()));
        }
        let corners: usize = m.faces.iter().map(|f| f.len()).sum();
        if part.tubes.is_empty() && m.uvs.len() != corners {
            bad.push(format!("{}: {} uvs for {} face corners", part.name, m.uvs.len(), corners));
        }
        if let Some(f) = m.faces.iter().find(|f| f.len() < 3 || f.iter().any(|&i| i as usize >= m.verts.len())) {
            bad.push(format!("{}: bad face {:?} ({} verts)", part.name, f, m.verts.len()));
        }
        if let Some(t) = m.tags.iter().find(|&&t| t < 0 || t as usize >= part.mats.len().max(1)) {
            bad.push(format!("{}: tag {} but {} materials", part.name, t, part.mats.len()));
        }
        for (pts, r) in &part.tubes {
            if pts.len() < 2 || !(*r > 0.0) || pts.iter().any(|p| !finite(*p)) {
                bad.push(format!("{}: bad tube ({} points, r {})", part.name, pts.len(), r));
                break;
            }
        }
        if part.mats.is_empty() {
            bad.push(format!("{}: no materials", part.name));
        }
    }
    let fields: Vec<_> = s.pitches.iter().map(field).collect();
    // the farmland or meadow plane the whole site and its surroundings stand on
    let world = s.parts.iter().filter(|p| p.name.starts_with("World:")).flat_map(|p| p.mesh.verts.iter()).fold(
        (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
        |(lo, hi), v| (lo.min(v.truncate()), hi.max(v.truncate())),
    );
    for ps in &s.props {
        // hedgerows, field trees and copses and riverbank reeds make up the surroundings; training gear is laid out on
        // the pitches for a session; floodlight heads sit up their poles
        let landscape = ps.key == "prop.hedge" || ps.key == "prop.reeds" || ps.key == "prop.shrub" || ps.key.starts_with("prop.tree_");
        let session = TRAINING_GEAR.iter().any(|k| ps.key == *k);
        let top = if ps.key.starts_with("floodlight_fixture") { 40.0 } else { 20.0 };
        for it in &ps.items {
            if !finite(it.pos) || !finite(it.scale) || !it.yaw.is_finite() || !it.tilt.is_finite() {
                bad.push(format!("{}: non-finite placement", ps.key));
                continue;
            }
            if it.scale.min_element() <= 0.0 {
                bad.push(format!("{}: scale {:?}", ps.key, it.scale));
            }
            if it.pos.z < -0.3 || it.pos.z > top {
                bad.push(format!("{}: off the ground at z {:.2}", ps.key, it.pos.z));
            }
            let q = it.pos.truncate();
            if world.0.x.is_finite() && (q.cmplt(world.0).any() || q.cmpgt(world.1).any()) {
                bad.push(format!("{}: off the edge of the world at {:?}", ps.key, q));
            }
            if !landscape && !s.boundary.is_empty() && !inside(&s.boundary, q) && dist_to_poly(&s.boundary, q) > 60.0 {
                bad.push(format!("{}: {:.1} m outside the site at {:?}", ps.key, dist_to_poly(&s.boundary, q), q));
            }
            if let Some(i) = fields.iter().position(|f| !session && in_field(f, q, 0.5)) {
                bad.push(format!("{}: on the field of play of {} at {:?}", ps.key, s.pitches[i].plot, q));
            }
        }
    }
    for (i, a) in fields.iter().enumerate() {
        for (j, b) in fields.iter().enumerate().skip(i + 1) {
            // corners of one field inside the other (fields are rectangles of similar size)
            let corners = |f: &(DVec2, DVec2, f64, f64)| {
                let n = f.1.perp();
                [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)].map(|(su, sv)| f.0 + f.1 * (su * f.2) + n * (sv * f.3))
            };
            if corners(a).iter().any(|c| in_field(b, *c, 0.0)) || corners(b).iter().any(|c| in_field(a, *c, 0.0)) || in_field(a, b.0, 0.0) {
                bad.push(format!("pitches {} and {} overlap", s.pitches[i].plot, s.pitches[j].plot));
            }
        }
    }
    for sp in &s.spots {
        if !finite(sp.pos) || !finite(sp.dir) || !(sp.power > 0.0) {
            bad.push("spot light with bad position, direction or power".into());
        }
    }
    for so in &s.sockets {
        if !finite(so.loc) {
            bad.push(format!("socket {}:{} not finite", so.kind, so.id));
        }
    }
    let r = &s.report;
    // goalkeeper areas ("<plot>-gk") are built with the pitch generator but are not pitches
    let pitches: Vec<_> = s.pitches.iter().filter(|p| !p.plot.ends_with("-gk")).collect();
    if r.pitches != pitches.len() {
        bad.push(format!("report: {} pitches, built {}", r.pitches, pitches.len()));
    }
    if r.grass + r.hybrid + r.artificial != r.pitches {
        bad.push(format!("report: surfaces {}+{}+{} != {} pitches", r.grass, r.hybrid, r.artificial, r.pitches));
    }
    let lit = pitches.iter().filter(|p| p.lit).count();
    if r.lit != lit {
        bad.push(format!("report: {} lit, built {}", r.lit, lit));
    }
    bad
}

/// Cheap fingerprint of a site, to check builds are deterministic.
fn digest(s: &Site) -> (usize, usize, f64, f64) {
    let v: f64 = s.parts.iter().flat_map(|p| p.mesh.verts.iter()).map(|v| v.x + 2.0 * v.y + 3.0 * v.z).sum();
    let p: f64 = s.props.iter().flat_map(|p| p.items.iter()).map(|i| i.pos.x - i.pos.y + i.yaw).sum();
    (s.parts.len(), s.props.iter().map(|p| p.items.len()).sum(), v, p)
}

#[test]
fn random_sites_hold_invariants() {
    let cat = Catalog::load(&root().join("catalog")).unwrap();
    let n: u64 = std::env::var("FUZZ_N").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let mut failures = Vec::new();
    for seed in 0..n {
        let layout = random_layout(&cat, 9000 + seed);
        let s = match site::build(&layout, &cat) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("seed {}: build failed: {e}\n{layout}", 9000 + seed));
                continue;
            }
        };
        let bad = check(&s);
        if !bad.is_empty() {
            let mut uniq: Vec<String> = Vec::new();
            for b in bad {
                if !uniq.contains(&b) && uniq.len() < 12 {
                    uniq.push(b);
                }
            }
            failures.push(format!("seed {}:\n  {}\n{layout}", 9000 + seed, uniq.join("\n  ")));
            continue;
        }
        let again = site::build(&layout, &cat).unwrap();
        if digest(&s) != digest(&again) {
            failures.push(format!("seed {}: not deterministic\n{layout}", 9000 + seed));
        }
    }
    assert!(failures.is_empty(), "{} of {n} layouts failed:\n{}", failures.len(), failures.join("\n\n"));
}

#[test]
fn preset_sites_hold_invariants() {
    let cat = Catalog::load(&root().join("catalog")).unwrap();
    let mut n = 0;
    for e in std::fs::read_dir(root().join("layouts")).unwrap() {
        let path = e.unwrap().path();
        let layout: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        if layout["schema"] != "site-layout/1" {
            continue;
        }
        n += 1;
        let s = site::build(&layout, &cat).unwrap();
        assert!(s.report.warnings.is_empty(), "{}: warnings {:?}", path.display(), s.report.warnings);
        let bad = check(&s);
        assert!(bad.is_empty(), "{}:\n  {}", path.display(), bad.join("\n  "));
    }
    assert!(n >= 5, "expected the five training-ground presets, found {n}");
}
