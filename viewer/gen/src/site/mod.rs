//! Training grounds (port of kit/site): a club-owned site from a site-layout/1 document.
//!
//! The land is fixed (a template from catalog/sites.json) and split into plots. Every plot is in
//! one state: occupied by something the club inherited (an old warehouse, a car park, scrub),
//! being demolished, cleared, under construction, or built on with a facility at a tier, in a style,
//! with options. The output is renderer-neutral like a stadium's: meshes with material references,
//! instanced props, pitches (built by the pitch generator), spot lights, sockets and a report.

pub mod academy;
pub mod buildings;
pub mod construction;
pub mod facilities;
pub mod kernel;
pub mod landscape;
pub mod interior;
pub mod mats;
pub mod occupants;

use crate::catalog::{num, text, Catalog};
use crate::mesh::MeshAcc;
use crate::patterns::hex_rgb;
use crate::prng::{crc32, uniform};
use crate::stadium::{Mat, Placement, PropSet, Spot};
use crate::stands::{Polyline, Socket};
use glam::{DVec2, DVec3};
use kernel::Frame;
use serde_json::{json, Map, Value};

/// Heights of the layered ground (no holes are cut: covers sit on the lawn).
pub const FIELD_Z: f64 = -0.06;
pub const LAWN_Z: f64 = -0.03;
pub const COVER_Z: f64 = 0.02;
pub const PATH_Z: f64 = 0.035;
pub const ROAD_Z: f64 = 0.05;

pub const STATES: [&str; 5] = ["occupied", "demolition", "cleared", "construction", "built"];
pub const BOUNDARIES: [&str; 5] = ["chainlink", "palisade", "railings", "brick_wall", "hedge"];
pub const SCREENING: [&str; 3] = ["none", "trees", "berm"];

/// A generated mesh (or a set of tubes) with one material per face tag.
#[derive(Clone, Debug)]
pub struct SitePart {
    pub name: String,
    pub mesh: MeshAcc,
    pub tubes: Vec<Polyline>,
    pub sides: usize,
    pub mats: Vec<Mat>,
    pub smooth_deg: f64,
}

/// A pitch on the site: the pitch generator's parameters and where it lies.
#[derive(Clone, Debug)]
pub struct SitePitch {
    pub plot: String,
    pub params: Value,
    pub origin: DVec2,
    /// radians: the pitch's length runs along this direction
    pub yaw: f64,
    pub goals: Option<String>,
    /// goals at both ends (false: the east end only, a goalkeeper area)
    pub both_goals: bool,
    pub flags: Option<String>,
    pub lit: bool,
    /// "grass", "hybrid" or "artificial"
    pub surface: String,
}

#[derive(Clone, Debug)]
pub struct Club {
    pub name: String,
    pub primary: String,
    pub secondary: String,
}

/// A plot ready to generate: its rectangle, frame, frontage and resolved state.
#[derive(Clone, Debug)]
pub struct Plot {
    pub id: String,
    pub kind: String,
    pub w: f64,
    pub d: f64,
    pub frame: Frame,
    /// local unit vector towards the road or path
    pub front: DVec2,
    pub cfg: Value,
    pub seed: u64,
}

impl Plot {
    pub fn state(&self) -> &str {
        text(&self.cfg, "state", "occupied")
    }
    pub fn corners(&self) -> Vec<DVec2> {
        kernel::rect(self.w, self.d, 0.0, 0.0).iter().map(|p| self.frame.pt(p.x, p.y)).collect()
    }
    pub fn r(&self, i: u64) -> f64 {
        uniform(self.seed, i)
    }
    /// yaw (radians, about z) that turns a prop's local +y to face the frontage
    pub fn front_yaw(&self) -> f64 {
        (-self.front.x).atan2(self.front.y)
    }
}

#[derive(Clone, Debug, Default)]
pub struct PlotReport {
    pub id: String,
    pub kind: String,
    pub state: String,
    pub label: String,
    pub facility: Option<String>,
    pub tier: usize,
    pub style: Option<String>,
    pub area_m2: f64,
}

#[derive(Clone, Debug, Default)]
pub struct SiteReport {
    pub template: String,
    pub plots: Vec<PlotReport>,
    pub pitches: usize,
    pub grass: usize,
    pub hybrid: usize,
    pub artificial: usize,
    pub lit: usize,
    pub replica: bool,
    pub parking: usize,
    pub area_m2: f64,
    pub developed_m2: f64,
    /// from the training building's entrance to the furthest pitch centre (m)
    pub walk_m: Option<f64>,
    /// the academy category the built facilities reach (kit/site/academy.py)
    pub academy: academy::Grade,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Site {
    pub name: String,
    pub night: bool,
    pub parts: Vec<SitePart>,
    pub props: Vec<PropSet>,
    pub pitches: Vec<SitePitch>,
    pub spots: Vec<Spot>,
    pub sockets: Vec<Socket>,
    /// overview camera (position, target)
    pub camera: (DVec3, DVec3),
    pub boundary: Vec<DVec2>,
    pub report: SiteReport,
}

/// Collects a site's output (parts in order, props grouped by asset and recolour).
pub struct Out {
    pub parts: Vec<SitePart>,
    pub props: Vec<PropSet>,
    pub pitches: Vec<SitePitch>,
    pub spots: Vec<Spot>,
    pub sockets: Vec<Socket>,
    pub warnings: Vec<String>,
    pub parking: usize,
    /// training building entrance (site xy)
    pub entrance: Option<DVec2>,
    pub night: bool,
    pub club: Club,
}

impl Out {
    pub fn part(&mut self, name: &str, mesh: MeshAcc, mats: Vec<Mat>, smooth_deg: f64) {
        if !mesh.faces.is_empty() {
            self.parts.push(SitePart { name: name.into(), mesh, tubes: vec![], sides: 0, mats, smooth_deg });
        }
    }

    /// A plot-local mesh moved into the site.
    pub fn part_at(&mut self, f: &Frame, name: &str, mut mesh: MeshAcc, mats: Vec<Mat>, smooth_deg: f64) {
        f.mesh(&mut mesh);
        self.part(name, mesh, mats, smooth_deg);
    }

    pub fn tubes(&mut self, name: &str, lines: Vec<Polyline>, mat: Mat, sides: usize) {
        if !lines.is_empty() {
            self.parts.push(SitePart { name: name.into(), mesh: MeshAcc::new(), tubes: lines, sides, mats: vec![mat], smooth_deg: 60.0 });
        }
    }

    pub fn tubes_at(&mut self, f: &Frame, name: &str, lines: Vec<Polyline>, mat: Mat, sides: usize) {
        let l = f.tubes(&lines);
        self.tubes(name, l, mat, sides);
    }

    /// An instance of a library prop; tints are (material, #rrggbb).
    pub fn prop(&mut self, key: &str, pl: Placement, tints: &[(&str, &str)]) {
        let t: Vec<(String, DVec3)> = tints.iter().map(|(n, h)| (n.to_string(), hex_rgb(h))).collect();
        match self.props.iter_mut().find(|p| p.key == key && p.tints == t) {
            Some(ps) => ps.items.push(pl),
            None => self.props.push(PropSet { key: key.into(), items: vec![pl], tints: t, net: None }),
        }
    }

    /// A prop at plot-local (u, v, z) turned by a local yaw.
    #[allow(clippy::too_many_arguments)]
    pub fn prop_at(&mut self, f: &Frame, key: &str, u: f64, v: f64, z: f64, yaw: f64, tints: &[(&str, &str)]) {
        let p = f.pt(u, v);
        self.prop(key, Placement::at(DVec3::new(p.x, p.y, z), f.yaw + yaw), tints);
    }

    /// Same with a tilt (local x) and scale.
    #[allow(clippy::too_many_arguments)]
    pub fn prop_full(&mut self, f: &Frame, key: &str, u: f64, v: f64, z: f64, yaw: f64, tilt: f64, scale: DVec3, tints: &[(&str, &str)]) {
        let p = f.pt(u, v);
        let mut pl = Placement::at(DVec3::new(p.x, p.y, z), f.yaw + yaw);
        pl.tilt = tilt;
        pl.scale = scale;
        self.prop(key, pl, tints);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn socket_at(&mut self, f: &Frame, kind: &'static str, id: &str, u: f64, v: f64, z: f64, dir: DVec2, props: Vec<(&'static str, f64)>) {
        let p = f.pt(u, v);
        let n = f.vec(dir.x, dir.y);
        self.sockets.push(Socket { kind, id: id.into(), loc: DVec3::new(p.x, p.y, z), n, props });
    }

    pub fn spot(&mut self, pos: DVec3, dir: DVec3, power: f64, angle_deg: f64, color: DVec3) {
        self.spots.push(Spot { pos, dir: dir.normalize(), power, angle_deg, blend: 0.6, color });
    }
}

pub fn facility_spec<'a>(cat: &'a Catalog, facility: &str) -> Option<&'a Value> {
    cat.facilities.get(facility)
}

pub fn tier_spec<'a>(cat: &'a Catalog, facility: &str, tier: usize) -> Option<&'a Value> {
    cat.facilities.get(facility)?.get("tiers")?.as_array()?.get(tier.max(1) - 1)
}

/// Does tier `tier` of `facility` fit a plot of `kind`?
pub fn fits(cat: &Catalog, facility: &str, tier: usize, kind: &str) -> bool {
    tier_spec(cat, facility, tier).and_then(|t| t.get("fits")).and_then(|f| f.as_array()).is_some_and(|a| a.iter().any(|k| k.as_str() == Some(kind)))
}

pub fn tier_count(cat: &Catalog, facility: &str) -> usize {
    cat.facilities.get(facility).and_then(|f| f.get("tiers")).and_then(|t| t.as_array()).map_or(0, |a| a.len())
}

/// A facility option: the plot's value, else the option default, else the tier's value.
pub fn option(cat: &Catalog, facility: &str, tier: usize, opts: &Value, key: &str) -> Value {
    if let Some(v) = opts.get(key).filter(|v| !v.is_null()) {
        return v.clone();
    }
    let d = cat.facilities.get(facility).and_then(|f| f.get("options")).and_then(|o| o.get(key)).and_then(|o| o.get("default")).cloned().unwrap_or(Value::Null);
    if !d.is_null() {
        return d;
    }
    tier_spec(cat, facility, tier).and_then(|t| t.get(key)).cloned().unwrap_or(Value::Null)
}

pub fn option_bool(cat: &Catalog, facility: &str, tier: usize, opts: &Value, key: &str) -> bool {
    let min_tier = cat.facilities.get(facility).and_then(|f| f.get("options")).and_then(|o| o.get(key)).and_then(|o| o.get("min_tier")).and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    tier >= min_tier && option(cat, facility, tier, opts, key).as_bool().unwrap_or(false)
}

fn is_hex(v: &Value) -> bool {
    crate::sanitize::is_hex(v)
}

/// A sanitised copy of the layout (template, club, style, boundary, plots) and the warnings for
/// what was changed. Every plot of the template gets an entry; merged plots replace their parts.
pub fn sanitize(layout: &Value, cat: &Catalog) -> (Value, Vec<String>) {
    let mut w = Vec::new();
    let mut l = if layout.is_object() { layout.clone() } else { json!({}) };
    let tpl_name = text(&l, "template", "edge_of_town").to_string();
    let tpl_name = if cat.sites.get(&tpl_name).is_some_and(|t| t.is_object()) && tpl_name != "plot_kinds" {
        tpl_name
    } else {
        w.push(format!("unknown site template {tpl_name:?}; using edge_of_town"));
        "edge_of_town".to_string()
    };
    l["template"] = json!(tpl_name);
    let tpl = &cat.sites[&tpl_name];
    // club and style
    let club = l.get("club").cloned().filter(|c| c.is_object()).unwrap_or(json!({}));
    let mut c2 = Map::new();
    c2.insert("name".into(), json!(text(&club, "name", "KIT FC")));
    for (k, d) in [("primary", "#c8102e"), ("secondary", "#f2f2f2")] {
        let v = club.get(k).cloned().unwrap_or(json!(d));
        if !is_hex(&v) {
            w.push(format!("club {k} colour {v} is not a #rrggbb colour; using {d}"));
            c2.insert(k.into(), json!(d));
        } else {
            c2.insert(k.into(), v);
        }
    }
    l["club"] = Value::Object(c2);
    let style = text(&l, "style", "classic").to_string();
    if !cat.styles.contains_key(&style) {
        w.push(format!("unknown style {style:?}; using classic"));
        l["style"] = json!("classic");
    } else {
        l["style"] = json!(style);
    }
    let style = l["style"].as_str().unwrap().to_string();
    // boundary, gate, landscape
    let b = l.get("boundary").cloned().filter(|v| v.is_object()).unwrap_or(json!({}));
    let mut b2 = json!({"type": text(&b, "type", "chainlink"), "screening": text(&b, "screening", "none")});
    if !BOUNDARIES.contains(&b2["type"].as_str().unwrap()) {
        w.push(format!("unknown boundary {}; using chainlink", b2["type"]));
        b2["type"] = json!("chainlink");
    }
    if !SCREENING.contains(&b2["screening"].as_str().unwrap()) {
        w.push(format!("unknown screening {}; using none", b2["screening"]));
        b2["screening"] = json!("none");
    }
    l["boundary"] = b2;
    let g = l.get("gate").cloned().filter(|v| v.is_object()).unwrap_or(json!({}));
    l["gate"] = json!({"tier": num(&g, "tier", 1.0).round().clamp(1.0, 3.0) as u64, "style": g.get("style").and_then(|s| s.as_str()).filter(|s| cat.styles.contains_key(*s)).unwrap_or(&style)});
    let ls = l.get("landscape").cloned().filter(|v| v.is_object()).unwrap_or(json!({}));
    l["landscape"] = json!({"trees": num(&ls, "trees", 0.5).clamp(0.0, 1.0)});
    // plots: the template's, with merges
    let given = l.get("plots").cloned().filter(|v| v.is_object()).unwrap_or(json!({}));
    let tpl_ids: Vec<String> = tpl["plots"].as_array().map(|a| a.iter().filter_map(|p| p["id"].as_str().map(String::from)).collect()).unwrap_or_default();
    let merges: Vec<&Value> = tpl.get("merges").and_then(|m| m.as_array()).map(|a| a.iter().collect()).unwrap_or_default();
    let mut out_plots = Map::new();
    let mut taken: Vec<String> = Vec::new();
    for m in &merges {
        let id = m["id"].as_str().unwrap_or("");
        if given.get(id).is_some_and(|v| v.is_object()) {
            let parts: Vec<String> = m["plots"].as_array().map(|a| a.iter().filter_map(|p| p.as_str().map(String::from)).collect()).unwrap_or_default();
            for p in &parts {
                if given.get(p).is_some() {
                    w.push(format!("plot {p} is part of {id}; its own entry is ignored"));
                }
            }
            taken.extend(parts);
            out_plots.insert(id.into(), given[id].clone());
        }
    }
    for (k, _) in given.as_object().unwrap() {
        if !tpl_ids.contains(k) && !merges.iter().any(|m| m["id"].as_str() == Some(k)) {
            w.push(format!("unknown plot {k:?} in template {tpl_name}"));
        }
    }
    for id in &tpl_ids {
        if !taken.contains(id) {
            out_plots.insert(id.clone(), given.get(id).cloned().unwrap_or(Value::Null));
        }
    }
    // each plot's state
    let kind_of = |id: &str| -> String {
        tpl["plots"].as_array().and_then(|a| a.iter().find(|p| p["id"].as_str() == Some(id))).or_else(|| merges.iter().find(|m| m["id"].as_str() == Some(id)).copied()).and_then(|p| p["kind"].as_str()).unwrap_or("medium").to_string()
    };
    let seed = num(&l, "seed", 1.0) as u64;
    let occ_keys: Vec<String> = cat.occupants.keys().cloned().collect();
    for (id, cfg) in out_plots.iter_mut() {
        let kind = kind_of(id);
        let mut c = cfg.clone();
        if !c.is_object() {
            c = json!({});
        }
        let mut state = text(&c, "state", "occupied").to_string();
        if !STATES.contains(&state.as_str()) {
            w.push(format!("{id}: unknown state {state:?}; using occupied"));
            state = "occupied".into();
        }
        let occ_fits = |o: &str| cat.occupants.get(o).and_then(|v| v["fits"].as_array()).is_some_and(|a| a.iter().any(|k| k.as_str() == Some(kind.as_str())));
        if state == "occupied" || state == "demolition" {
            let occ = text(&c, "occupant", "").to_string();
            if !occ_fits(&occ) {
                let fitting: Vec<&String> = occ_keys.iter().filter(|o| occ_fits(o)).collect();
                let pick = fitting[(uniform(seed + crc32(id) as u64, 0) * fitting.len() as f64) as usize % fitting.len()].clone();
                if !occ.is_empty() {
                    w.push(format!("{id}: occupant {occ:?} does not fit a {kind} plot; using {pick}"));
                }
                c["occupant"] = json!(pick);
            }
        }
        if state == "construction" || state == "built" {
            let fac = text(&c, "facility", "").to_string();
            let n = tier_count(cat, &fac);
            if n == 0 {
                w.push(format!("{id}: unknown facility {fac:?}; plot left cleared"));
                state = "cleared".into();
            } else {
                let want = (num(&c, "tier", 1.0).round().max(1.0) as usize).min(n);
                let tier = (1..=want).rev().find(|&t| fits(cat, &fac, t, &kind));
                match tier {
                    Some(t) => {
                        if t != want {
                            w.push(format!("{id}: tier {want} {fac} does not fit a {kind} plot; using tier {t}"));
                        }
                        c["tier"] = json!(t);
                    }
                    None => {
                        w.push(format!("{id}: a {fac} does not fit a {kind} plot; plot left cleared"));
                        state = "cleared".into();
                    }
                }
                if let Some(s) = c.get("style").and_then(|s| s.as_str()) {
                    if !cat.styles.contains_key(s) {
                        w.push(format!("{id}: unknown style {s:?}; using the site style"));
                        c.as_object_mut().unwrap().remove("style");
                    }
                }
                if !c.get("options").is_some_and(|o| o.is_object()) {
                    c["options"] = json!({});
                }
            }
        }
        c["state"] = json!(state);
        *cfg = c;
    }
    l["plots"] = Value::Object(out_plots);
    (l, w)
}

fn front_vec(f: &str) -> DVec2 {
    match f {
        "x" => DVec2::new(1.0, 0.0),
        "-x" => DVec2::new(-1.0, 0.0),
        "y" => DVec2::new(0.0, 1.0),
        _ => DVec2::new(0.0, -1.0),
    }
}

/// The plots to generate, in template order (a merge takes its first part's place).
pub fn plots(layout: &Value, cat: &Catalog) -> Vec<Plot> {
    let tpl = &cat.sites[layout["template"].as_str().unwrap()];
    let kinds = &cat.sites["plot_kinds"];
    let seed = num(layout, "seed", 1.0) as u64;
    let mut out = Vec::new();
    let mut done: Vec<String> = Vec::new();
    let defs: Vec<&Value> = tpl["plots"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
    for p in defs {
        let id = p["id"].as_str().unwrap_or("").to_string();
        let merged = tpl.get("merges").and_then(|m| m.as_array()).and_then(|a| a.iter().find(|m| m["plots"].as_array().is_some_and(|ps| ps.iter().any(|x| x.as_str() == Some(&id))) && layout["plots"].get(m["id"].as_str().unwrap_or("")).is_some()));
        let def = merged.unwrap_or(p);
        let pid = def["id"].as_str().unwrap_or("").to_string();
        if done.contains(&pid) {
            continue;
        }
        done.push(pid.clone());
        let kind = def["kind"].as_str().unwrap_or("medium").to_string();
        let size = &kinds[&kind]["size"];
        out.push(Plot {
            id: pid.clone(),
            kind,
            w: size[0].as_f64().unwrap_or(50.0),
            d: size[1].as_f64().unwrap_or(40.0),
            frame: Frame::new(num(def, "x", 0.0), num(def, "y", 0.0), num(def, "yaw", 0.0)),
            front: front_vec(text(def, "front", "-y")),
            cfg: layout["plots"].get(&pid).cloned().unwrap_or(json!({"state": "occupied"})),
            seed: seed.wrapping_mul(1_000_003).wrapping_add(crc32(&pid) as u64),
        });
    }
    out
}

/// Build a training ground from a site-layout/1 document.
pub fn build(layout: &Value, cat: &Catalog) -> Result<Site, String> {
    let (l, warnings) = sanitize(layout, cat);
    let tpl_name = l["template"].as_str().unwrap().to_string();
    let tpl = cat.sites[&tpl_name].clone();
    let night = text(&l, "lighting", "day") == "night";
    let club = Club { name: text(&l["club"], "name", "KIT FC").to_string(), primary: text(&l["club"], "primary", "#c8102e").to_string(), secondary: text(&l["club"], "secondary", "#f2f2f2").to_string() };
    let mut o = Out { parts: vec![], props: vec![], pitches: vec![], spots: vec![], sockets: vec![], warnings, parking: 0, entrance: None, night, club };
    let boundary: Vec<DVec2> = tpl["boundary"].as_array().map(|a| a.iter().map(|p| DVec2::new(p[0].as_f64().unwrap_or(0.0), p[1].as_f64().unwrap_or(0.0))).collect()).unwrap_or_default();
    let ps = plots(&l, cat);
    let style = l["style"].as_str().unwrap().to_string();
    let seed = num(&l, "seed", 1.0) as u64;

    landscape::context(&mut o, &tpl, &boundary, seed);
    landscape::site_ground(&mut o, &boundary);
    landscape::roads(&mut o, &tpl, &ps);
    landscape::boundary(&mut o, &tpl, &boundary, &l["boundary"], seed);
    landscape::gate(&mut o, cat, &tpl, &l["gate"], seed);

    let mut report = SiteReport { template: tpl_name.clone(), ..Default::default() };
    for p in &ps {
        let state = p.state().to_string();
        let mut pr = PlotReport { id: p.id.clone(), kind: p.kind.clone(), state: state.clone(), area_m2: p.w * p.d, ..Default::default() };
        match state.as_str() {
            "occupied" | "demolition" => {
                let occ = text(&p.cfg, "occupant", "scrubland").to_string();
                pr.label = cat.occupants.get(&occ).map(|v| text(v, "label", &occ).to_string()).unwrap_or(occ.clone());
                occupants::build(&mut o, p, &occ, state == "demolition");
                if state == "demolition" {
                    construction::demolition(&mut o, p);
                    pr.label = format!("Demolishing {}", pr.label.to_lowercase());
                }
            }
            "cleared" => {
                pr.label = "Cleared".into();
                construction::cleared(&mut o, p);
            }
            "construction" | "built" => {
                let fac = text(&p.cfg, "facility", "").to_string();
                let tier = num(&p.cfg, "tier", 1.0) as usize;
                let st = p.cfg.get("style").and_then(|v| v.as_str()).unwrap_or(&style).to_string();
                let tl = tier_spec(cat, &fac, tier).map(|t| text(t, "label", "").to_string()).unwrap_or_default();
                let fl = facility_spec(cat, &fac).map(|t| text(t, "label", &fac).to_string()).unwrap_or(fac.clone());
                pr.facility = Some(fac.clone());
                pr.tier = tier;
                pr.style = Some(st.clone());
                if state == "construction" {
                    pr.label = format!("Building: {fl} ({tl})");
                    construction::site(&mut o, cat, p, &fac, tier, &st);
                } else {
                    pr.label = format!("{fl}: {tl}");
                    facilities::build(&mut o, cat, p, &fac, tier, &st);
                    report.developed_m2 += p.w * p.d;
                }
            }
            _ => {}
        }
        o.socket_at(&p.frame, "plot", &p.id, 0.0, 0.0, 0.0, p.front, vec![("width", p.w), ("depth", p.d)]);
        report.plots.push(pr);
    }
    landscape::trees(&mut o, &tpl, &boundary, &ps, num(&l["landscape"], "trees", 0.5), seed);

    // report
    for sp in &o.pitches {
        if sp.plot.contains("gk") {
            continue;
        }
        report.pitches += 1;
        match sp.surface.as_str() {
            "artificial" => report.artificial += 1,
            "hybrid" => report.hybrid += 1,
            _ => report.grass += 1,
        }
        report.lit += sp.lit as usize;
    }
    report.replica = ps.iter().any(|p| p.state() == "built" && text(&p.cfg, "facility", "") == "pitch" && num(&p.cfg, "tier", 1.0) as usize >= 4);
    report.parking = o.parking;
    report.academy = academy::grade(&ps, cat, &o.pitches);
    report.area_m2 = kernel::area2(&boundary).abs() / 2.0;
    if let Some(e) = o.entrance {
        let far = o.pitches.iter().map(|sp| (sp.origin - e).length()).fold(0.0, f64::max);
        report.walk_m = (!o.pitches.is_empty()).then_some(far);
    }
    report.warnings = std::mem::take(&mut o.warnings);
    // overview camera: from the south-west, above the gate side
    let (lo, hi) = boundary.iter().fold((DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let c = (lo + hi) / 2.0;
    let span = (hi - lo).length();
    let camera = (DVec3::new(c.x - span * 0.45, c.y - span * 0.62, span * 0.42), DVec3::new(c.x, c.y + span * 0.02, 0.0));
    Ok(Site {
        name: text(&l, "name", "Training ground").to_string(),
        night,
        parts: o.parts,
        props: o.props,
        pitches: o.pitches,
        spots: o.spots,
        sockets: o.sockets,
        camera,
        boundary,
        report,
    })
}

/// Everything the renderer draws: the site's parts (tubes turned into meshes) and props, plus each
/// pitch built by the pitch generator (surface, lines, goals, corner flags) in its place.
pub fn realize(site: &Site, cat: &Catalog) -> (Vec<crate::stadium::Part>, Vec<PropSet>) {
    use crate::mesh::tube_lines;
    use crate::pitch::{self, Pitch};
    use crate::stadium::Part;
    let mut parts = Vec::new();
    let mut props = site.props.clone();
    for p in &site.parts {
        let mesh = if p.tubes.is_empty() { p.mesh.clone() } else { tube_lines(&p.tubes, p.sides.max(3)) };
        parts.push(Part { name: p.name.clone(), mesh, mats: p.mats.clone(), smooth_deg: p.smooth_deg });
    }
    for sp in &site.pitches {
        let pt = Pitch::resolve(&sp.params, Some(cat));
        let f = Frame::new(sp.origin.x, sp.origin.y, sp.yaw.to_degrees());
        let spec = pt.grass_spec(sp.origin, sp.yaw);
        let grass = Mat { key: "Pitch_Grass".into(), color: None, emission: None, texture: Some(crate::stadium::Tex::Grass(spec.clone())) };
        let paint = Mat { key: "TINT_Pitch_Paint".into(), color: Some(hex_rgb(&pt.line_color)), emission: None, texture: Some(crate::stadium::Tex::Grass(spec)) };
        let mut surf = pitch::surface_mesh(&pt);
        f.mesh(&mut surf);
        let mut lines = pitch::markings_mesh(&pt);
        f.mesh(&mut lines);
        parts.push(Part { name: format!("{}:pitch", sp.plot), mesh: surf, mats: vec![grass], smooth_deg: 30.0 });
        parts.push(Part { name: format!("{}:lines", sp.plot), mesh: lines, mats: vec![paint], smooth_deg: 30.0 });
        let goals = sp.goals.as_ref().map(|g| json!({"style": g})).unwrap_or(Value::Null);
        let flags = sp.flags.as_ref().map(|g| json!({"preset": g})).unwrap_or(Value::Null);
        if let Ok(sets) = crate::stadium::goals_and_flags(&pt, &goals, &flags, cat) {
            for mut ps in sets {
                if !sp.both_goals && ps.key.starts_with("goal.") {
                    ps.items.retain(|it| it.pos.x > 0.0);
                }
                for it in ps.items.iter_mut() {
                    let q = f.pt(it.pos.x, it.pos.y);
                    it.pos = DVec3::new(q.x, q.y, it.pos.z);
                    it.yaw += sp.yaw;
                }
                match props.iter_mut().find(|p| p.key == ps.key && p.tints == ps.tints && p.net == ps.net) {
                    Some(e) => e.items.extend(ps.items),
                    None => props.push(ps),
                }
            }
        }
    }
    (parts, props)
}
