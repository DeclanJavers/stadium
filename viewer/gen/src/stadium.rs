//! A whole stadium from a stadium-layout/1 document (port of kit/assembler.py and the Blender-side
//! assembly in kit/assets/{stand,roof,facade,ads,screen,dugout,tunnel,floodlight}.py).
//!
//! The output is renderer-neutral: generated meshes with material references (library material
//! names plus recolours), instanced props (seats, goals, flags, fixtures), spot lights, sockets and
//! a report with capacity, ad inventory, floodlight design and warnings. Coordinates are the kit's
//! (metres, Z up, centre mark at the origin).

use crate::catalog::{merge, num, resolve_preset, text, Catalog};
use crate::facades::{self, RoofInfo};
use crate::floodlights::{self, Design, StandInfo};
use crate::mesh::{tube_lines, MeshAcc, Section};
use crate::patterns::{self, Picture};
use crate::pitch::{self, Pitch};
use crate::placement;
use crate::prng::crc32;
use crate::py::{linspace, round_to};
use crate::roofs;
use crate::runs::{perimeter_runs, Run};
use crate::sightlines::{report as tier_report, solve_profile, Kind, Tier, TierReport, WALL};
use crate::stands::{self, stand_depth, vom_rows, Anchor, Polyline, Socket};
use glam::{DVec2, DVec3, DVec4};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::f64::consts::PI;

pub const STANDING_PER_M: f64 = 1.8;
pub const CREATIVE_M: f64 = 8.0;
pub const DUGOUT_FRONT: f64 = 4.3;
pub const DUGOUT_DEPTH: f64 = 2.5;
pub const DUGOUT_GAP: f64 = 6.0;

/// A texture the renderer composes at runtime.
#[derive(Clone, Debug, PartialEq)]
pub enum Tex {
    /// horizontal atlas of board creatives, in this order (the LED rotation)
    BoardAtlas(Vec<String>),
    Carpet(String),
    Scoreboard,
    /// goal net card: cord pattern regenerated at runtime
    Net(NetSpec),
}

/// Net options for the game's net card (the hero net's cords are a Blender render asset).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NetSpec {
    /// "hex" or "square"
    pub mesh: String,
    pub mesh_size_mm: u32,
    /// cord diameter in tenths of a millimetre
    pub cord_tenths_mm: u32,
    /// "solid", "bands" or "checker"
    pub pattern: String,
    pub band_cells: u32,
    /// sRGB 0-255
    pub colors: Vec<[u8; 3]>,
}

/// Material reference: a library material by name, optionally recoloured / emissive / textured.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub key: String,
    /// sRGB recolour (TINT_ materials scale their detail texture; others replace the base colour)
    pub color: Option<DVec3>,
    /// (sRGB colour, strength)
    pub emission: Option<(DVec3, f64)>,
    pub texture: Option<Tex>,
}

impl Mat {
    pub fn named(key: &str) -> Mat {
        Mat { key: key.into(), color: None, emission: None, texture: None }
    }
    pub fn tint(key: &str, hex: &str) -> Mat {
        Mat { key: key.into(), color: Some(patterns::hex_rgb(hex)), emission: None, texture: None }
    }
}

/// A generated mesh: one material per face tag.
#[derive(Clone, Debug)]
pub struct Part {
    pub name: String,
    pub mesh: MeshAcc,
    pub mats: Vec<Mat>,
    /// smooth normals across edges below this angle (0 = flat shading)
    pub smooth_deg: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub pos: DVec3,
    /// rotation about Z (radians), then `tilt` about the rotated local X
    pub yaw: f64,
    pub tilt: f64,
    pub scale: DVec3,
    /// per-instance colour override (linear RGBA)
    pub color: Option<DVec4>,
}

impl Placement {
    pub fn at(pos: DVec3, yaw: f64) -> Self {
        Placement { pos, yaw, tilt: 0.0, scale: DVec3::ONE, color: None }
    }
    /// Local +Y pointing along `dir` (Blender's track-to Y, up Z).
    pub fn aim(pos: DVec3, dir: DVec3) -> Self {
        let d = dir.normalize();
        Placement { pos, yaw: (-d.x).atan2(d.y), tilt: d.z.clamp(-1.0, 1.0).asin(), scale: DVec3::ONE, color: None }
    }
}

/// Instances of one library asset (e.g. "seat.tipup_classic.std", "goal.box").
#[derive(Clone, Debug)]
pub struct PropSet {
    pub key: String,
    pub items: Vec<Placement>,
    /// material recolours for this asset: (material name, sRGB)
    pub tints: Vec<(String, DVec3)>,
    /// goal nets: the card's cord pattern
    pub net: Option<NetSpec>,
}

#[derive(Clone, Copy, Debug)]
pub struct Spot {
    pub pos: DVec3,
    pub dir: DVec3,
    /// Blender spot power (W): radiant intensity is power / 4 pi W/sr
    pub power: f64,
    pub angle_deg: f64,
    pub blend: f64,
    /// linear RGB
    pub color: DVec3,
}

#[derive(Clone, Debug, Default)]
pub struct Capacity {
    pub seated: usize,
    pub standing: usize,
    pub total: usize,
    pub by_style: BTreeMap<String, usize>,
}

#[derive(Clone, Debug)]
pub struct AdSlot {
    pub kind: &'static str,
    pub id: String,
    pub area_m2: f64,
    pub tv_weight: f64,
    pub brand: String,
}

#[derive(Clone, Debug, Default)]
pub struct AdTotal {
    pub slots: usize,
    pub area_m2: f64,
    pub tv_weighted_m2: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub capacity: Capacity,
    pub warnings: Vec<String>,
    pub tiers: Vec<(String, Vec<TierReport>)>,
    pub roofs: Vec<(String, f64, f64, f64)>, // run, raised for sightline, raised for headroom, clearance
    pub ads: Vec<AdSlot>,
    pub ad_totals: BTreeMap<String, AdTotal>,
    pub lights: Option<Design>,
    pub screens: usize,
    pub footprint: (f64, f64),
    /// end boards: metres behind the goal line
    pub end_boards_m: f64,
    pub grass: Value,
}

#[derive(Clone, Debug)]
pub struct Stadium {
    pub name: String,
    pub pitch: Pitch,
    pub night: bool,
    pub parts: Vec<Part>,
    pub props: Vec<PropSet>,
    pub spots: Vec<Spot>,
    pub sockets: Vec<Socket>,
    /// broadcast camera (position, target)
    pub camera: (DVec3, DVec3),
    pub report: Report,
}

/// Stand facts other families need (ads, screens, tunnel, floodlights).
struct StandsOut {
    a: f64,
    b: f64,
    runs: Vec<(Run, Vec<Tier>)>,
    roofs: Vec<(String, roofs::RoofOut)>,
    facade_lines: Vec<(String, f64)>,
    camera_mount: Option<DVec3>,
}

/// Python truthiness of a JSON value (`if cfg.get("carpets"):` in the Blender code).
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn hexv(v: &Value, key: &str, default: &str) -> String {
    v.get(key).and_then(|x| x.as_str()).unwrap_or(default).to_string()
}

fn seat_spec<'a>(cat: &'a Catalog, style: &str) -> Result<&'a Value, String> {
    cat.seats.get(style).ok_or_else(|| format!("unknown seat style {style:?}; options: {:?}", cat.seats.keys().collect::<Vec<_>>()))
}

pub fn variant_index(spec: &Value, seat: usize, of: usize) -> usize {
    if let Some(re) = spec.get("row_end").and_then(|v| v.as_u64()) {
        if seat + 1 == of {
            return re as usize;
        }
    }
    if let Some(ev) = spec.get("every").and_then(|v| v.as_array()) {
        let (v, p) = (ev[0].as_u64().unwrap_or(0) as usize, ev[1].as_u64().unwrap_or(1).max(1) as usize);
        if seat % p == 0 {
            return v;
        }
    }
    0
}

fn variants(spec: &Value) -> Vec<String> {
    spec["variants"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_else(|| vec!["std".into()])
}

fn lin(c: DVec3) -> DVec3 {
    let f = |v: f64| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    DVec3::new(f(c.x), f(c.y), f(c.z))
}

/// Box from the dugout module: (x0..x1, y0..y1, z0..z1), outward-facing quads.
fn add_box(acc: &mut MeshAcc, lo: DVec3, hi: DVec3, tag: i32) {
    let base = acc.verts.len() as u32;
    for z in [lo.z, hi.z] {
        for y in [lo.y, hi.y] {
            for x in [lo.x, hi.x] {
                acc.verts.push(DVec3::new(x, y, z));
            }
        }
    }
    for f in [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]] {
        let q: Vec<u32> = f.iter().map(|i| base + i).collect();
        // metric UVs: the two axes the face spans
        let pts: Vec<DVec3> = q.iter().map(|&i| acc.verts[i as usize]).collect();
        let n = (pts[1] - pts[0]).cross(pts[2] - pts[0]).abs();
        let uv = pts
            .iter()
            .map(|p| if n.x >= n.y && n.x >= n.z { DVec2::new(p.y, p.z) } else if n.y >= n.z { DVec2::new(p.x, p.z) } else { DVec2::new(p.x, p.y) })
            .collect();
        acc.add_face(q, tag, Some(uv));
    }
}

fn transform(acc: &mut MeshAcc, p: &Placement) {
    let (s, c) = p.yaw.sin_cos();
    let (st, ct) = p.tilt.sin_cos();
    for v in acc.verts.iter_mut() {
        let l = *v * p.scale;
        let l = DVec3::new(l.x, l.y * ct - l.z * st, l.y * st + l.z * ct);
        *v = DVec3::new(l.x * c - l.y * s, l.x * s + l.y * c, l.z) + p.pos;
    }
}

fn tubes_part(name: &str, lines: &[Polyline], sides: usize, mat: Mat) -> Option<Part> {
    let m = tube_lines(lines, sides);
    if m.faces.is_empty() {
        return None;
    }
    Some(Part { name: name.into(), mesh: m, mats: vec![mat], smooth_deg: 60.0 })
}

/// Build a stadium from a layout (stadium-layout/1). Image mosaics need `build_with_images`.
pub fn build(layout: &Value, cat: &Catalog) -> Result<Stadium, String> {
    build_with_images(layout, cat, &|_| None)
}

/// Loads a mosaic image ("image" path in the layout, relative to the repository root).
pub type ImageLoader<'a> = dyn Fn(&str) -> Option<Picture> + 'a;

pub fn build_with_images(layout: &Value, cat: &Catalog, images: &ImageLoader) -> Result<Stadium, String> {
    let night = layout.get("lighting").and_then(|v| v.as_str()) == Some("night");
    let pitch = Pitch::resolve(layout.get("pitch").unwrap_or(&Value::Null), Some(cat));
    let mut st = Stadium {
        name: text(layout, "name", "Stadium").to_string(),
        pitch: pitch.clone(),
        night,
        parts: Vec::new(),
        props: Vec::new(),
        spots: Vec::new(),
        sockets: Vec::new(),
        camera: placement::main_camera(&pitch, None),
        report: Report { grass: pitch.grass_spec.clone(), ..Default::default() },
    };
    build_pitch(&mut st);
    build_goals_and_flags(&mut st, layout, cat)?;
    let stands = match layout.get("stands") {
        Some(cfg) if cfg.is_object() => Some(build_stands(&mut st, cfg, cat, images)?),
        _ => None,
    };
    if let Some(s) = &stands {
        st.camera = placement::main_camera(&pitch, s.camera_mount);
        st.report.footprint = (s.a, s.b);
    }
    let dug = layout.get("dugouts").filter(|v| v.is_object());
    let tun = layout.get("tunnel").filter(|v| v.is_object());
    let gap = match dug {
        Some(d) => 2.0 * dugout_length(d, cat)? + DUGOUT_GAP + 3.0,
        None if tun.is_some() => 6.0,
        None => 0.0,
    };
    if let Some(ads) = layout.get("ads").filter(|v| v.is_object()) {
        build_ads(&mut st, ads, cat, stands.as_ref(), gap)?;
    }
    if let Some(sc) = layout.get("screens").filter(|v| v.is_object()) {
        build_screens(&mut st, sc, stands.as_ref())?;
    }
    if let Some(d) = dug {
        build_dugouts(&mut st, d, cat)?;
    }
    if let Some(t) = tun {
        build_tunnel(&mut st, t, stands.as_ref());
    }
    if let Some(fl) = layout.get("floodlights").filter(|v| v.is_object()) {
        build_floodlights(&mut st, fl, cat, stands.as_ref())?;
    }
    Ok(st)
}

// -- pitch, goals, corner flags -------------------------------------------------------------------

fn build_pitch(st: &mut Stadium) {
    let p = &st.pitch;
    st.parts.push(Part { name: "PitchSurface".into(), mesh: pitch::surface_mesh(p), mats: vec![Mat::named("Pitch_Grass")], smooth_deg: 30.0 });
    st.parts.push(Part { name: "PitchLines".into(), mesh: pitch::markings_mesh(p), mats: vec![Mat::tint("TINT_Pitch_Paint", &p.line_color)], smooth_deg: 30.0 });
    for (kind, id, pos, yaw) in pitch::sockets(p) {
        st.sockets.push(Socket { kind, id: id.into(), loc: pos, n: DVec2::new(yaw.to_radians().cos(), yaw.to_radians().sin()), props: vec![("yaw_deg", yaw)] });
    }
}

fn build_goals_and_flags(st: &mut Stadium, layout: &Value, cat: &Catalog) -> Result<(), String> {
    let goals = layout.get("goals").cloned().unwrap_or(json!({"style": "classic"}));
    if !goals.is_null() {
        let style = text(&goals, "style", "classic").to_string();
        if !cat.goals.contains_key(&style) {
            return Err(format!("unknown goal style {style:?}"));
        }
        let mut tints = vec![("TINT_frame".to_string(), patterns::hex_rgb(&hexv(&goals, "frame_color", "#f4f4f0")))];
        let nets: Vec<String> = goals.get("net_colors").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|c| c.as_str().map(String::from)).collect()).unwrap_or_else(|| vec!["#f7f7f5".into()]);
        if let Some(c) = nets.first() {
            tints.push(("TINT_net_card".into(), patterns::hex_rgb(c)));
            tints.push(("TINT_net_a".into(), patterns::hex_rgb(c)));
        }
        let to8 = |c: &str| {
            let v = patterns::hex_rgb(c) * 255.0;
            [v.x.round() as u8, v.y.round() as u8, v.z.round() as u8]
        };
        let net = NetSpec {
            mesh: text(&goals, "mesh", "hex").to_string(),
            mesh_size_mm: (num(&goals, "mesh_size", 0.12) * 1000.0).round() as u32,
            cord_tenths_mm: (num(&goals, "cord_mm", 4.0) * 10.0).round() as u32,
            pattern: text(&goals, "net_pattern", "solid").to_string(),
            band_cells: num(&goals, "band_cells", 3.0).max(1.0) as u32,
            colors: nets.iter().map(|c| to8(c)).collect(),
        };
        let items = pitch::sockets(&st.pitch).into_iter().filter(|s| s.0 == "goal").map(|s| Placement::at(s.2, s.3.to_radians())).collect();
        st.props.push(PropSet { key: format!("goal.{style}"), items, tints, net: Some(net) });
    }
    let flags = layout.get("corner_flags").cloned().unwrap_or(json!({"preset": "pro_spring"}));
    if !flags.is_null() {
        let preset = text(&flags, "preset", "pro_spring").to_string();
        let spec = cat.corner_flags.get(&preset).ok_or_else(|| format!("unknown corner flag preset {preset:?}"))?;
        let spec = merge(spec, &flags);
        let mut tints = Vec::new();
        if let Some(c) = spec["pole_colors"].as_array().and_then(|a| a.first()).and_then(|c| c.as_str()) {
            tints.push(("TINT_pole_a".to_string(), patterns::hex_rgb(c)));
        }
        let items = pitch::sockets(&st.pitch).into_iter().filter(|s| s.0 == "corner_flag").map(|s| Placement::at(s.2, s.3.to_radians())).collect();
        st.props.push(PropSet { key: format!("corner_flag.{preset}"), items, tints, net: None });
    }
    Ok(())
}

// -- stands -------------------------------------------------------------------------------------

struct Side {
    name: String,
    cfg: Value,
    profile: String,
}

fn barrier_lines(barriers: &[Vec<DVec3>]) -> Vec<Polyline> {
    let mut out = Vec::new();
    for pts in barriers {
        out.push((pts.clone(), 0.03));
        for p in pts.iter().step_by(2) {
            out.push((vec![*p - DVec3::new(0.0, 0.0, 1.1), *p], 0.03));
        }
    }
    out
}

fn nosings_mesh(segs: &[(DVec3, DVec3)]) -> MeshAcc {
    let mut acc = MeshAcc::new();
    for (p0, p1) in segs {
        let d = *p1 - *p0;
        let mut n = DVec3::new(d.y, -d.x, 0.0);
        n /= n.length().max(1e-9);
        let lift = DVec3::new(0.0, 0.0, 0.002);
        let b = acc.verts.len() as u32;
        acc.verts.extend([*p0 + lift, *p1 + lift, *p1 + n * 0.055 + lift, *p0 + n * 0.055 + lift]);
        acc.add_face(vec![b, b + 1, b + 2, b + 3], 0, None);
    }
    acc
}

/// Point on a rounded rectangle along a ray from the centre (bisection, as the Python does it).
fn rect_hit(d: DVec2, hx: f64, hy: f64, r: f64) -> DVec2 {
    let (mut lo, mut hi) = (0.0, 4.0 * (hx + hy));
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let (x, y) = ((d.x * mid).abs(), (d.y * mid).abs());
        let (qx, qy) = ((x - (hx - r)).max(0.0), (y - (hy - r)).max(0.0));
        let inside = x <= hx && y <= hy && qx * qx + qy * qy <= r * r;
        if inside {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    d * lo
}

fn facade_mats(spec: &Value, night: bool) -> Vec<Mat> {
    let kind = text(spec, "type", "concrete_frame");
    let color = hexv(spec, "color", "#b8b3a8");
    let accent = hexv(spec, "accent", &color);
    let (main, trim) = match kind {
        "brick" => (Mat::tint("Facade_Brick", &color), Mat::named("Facade_Stone")),
        "concrete_frame" => (Mat::tint("Facade_Concrete", &color), Mat::named("Facade_Trim")),
        _ => (Mat::tint("TINT_facade", &color), Mat::named("Facade_Trim")),
    };
    let glow = spec.get("glow").and_then(|v| v.as_str());
    let mut etfe = Mat::tint("TINT_facade_etfe", &color);
    if night {
        if let Some(g) = glow {
            etfe.emission = Some((patterns::hex_rgb(g), 1.6));
        }
    }
    vec![main, Mat::tint("TINT_facade_b", &accent), Mat::named("Facade_DarkGlass"), trim, etfe, Mat::named("Facade_Core")]
}

fn build_stands(st: &mut Stadium, cfg: &Value, cat: &Catalog, images: &ImageLoader) -> Result<StandsOut, String> {
    let p = st.pitch.clone();
    let mut sides: Vec<Side> = Vec::new();
    if let Some(obj) = cfg.get("sides").and_then(|v| v.as_object()) {
        for (name, sc) in obj {
            let Some(prof) = sc.get("profile").and_then(|v| v.as_str()) else { continue };
            if !cat.profiles.contains_key(prof) {
                return Err(format!("unknown stand profile {prof:?}; options: {:?}", cat.profile_order));
            }
            sides.push(Side { name: name.clone(), cfg: sc.clone(), profile: prof.to_string() });
        }
    }
    let gap = num(cfg, "front_gap", 3.0);
    let curved = text(cfg, "corners", "open") == "curved";
    let r_corner = num(cfg, "corner_radius", 14.0);
    let a = p.length / 2.0 + p.runoff_end + gap;
    let b = p.width / 2.0 + p.runoff_side + gap;
    if curved {
        let profs: std::collections::BTreeSet<&str> = sides.iter().map(|s| s.profile.as_str()).collect();
        if sides.len() < 4 || profs.len() != 1 {
            return Err(format!("curved corners need all four sides to share one profile (a bowl); got {:?} on {:?}", profs, sides.iter().map(|s| &s.name).collect::<Vec<_>>()));
        }
    }
    let concrete = Mat::tint("Stand_Concrete", &hexv(cfg, "concrete", "#9d998f"));
    let mats = vec![concrete, Mat::tint("TINT_fascia", &hexv(cfg, "fascia_color", "#1b1d20")), Mat::named("Stand_Soffit"), Mat::named("Stand_Block")];
    let side_list: Vec<(String, Option<f64>)> = sides.iter().map(|s| (s.name.clone(), s.cfg.get("length").and_then(|v| v.as_f64()))).collect();
    let run_defs = perimeter_runs(a, b, &side_list, curved, r_corner);
    let focal = |n: &str| match n {
        "south" | "north" => p.runoff_side + gap,
        _ => p.runoff_end + gap,
    };
    let hosp_seat = text(cfg, "hospitality_seat", "padded_vip").to_string();
    let mut all: Vec<Anchor> = Vec::new();
    let mut styles: Vec<String> = Vec::new(); // per anchor
    let mut run_lengths: Vec<f64> = Vec::new();
    let mut tier_rows: Vec<Vec<usize>> = Vec::new();
    let mut vom_top: Vec<Vec<Option<usize>>> = Vec::new();
    let mut standing = 0.0;
    let mut out = StandsOut { a, b, runs: Vec::new(), roofs: Vec::new(), facade_lines: Vec::new(), camera_mount: None };
    let mut warnings: Vec<String> = Vec::new();
    for (ri, run) in run_defs.iter().enumerate() {
        let sc = sides.iter().find(|s| s.name == run.name).unwrap_or(&sides[0]);
        let df = if curved { focal("south").min(focal("east")) } else { focal(&run.name) };
        let mut tiers = solve_profile(&cat.profiles[&sc.profile], df);
        let style = text(&sc.cfg, "seat", "tipup_classic").to_string();
        let mut tier_styles = Vec::new();
        for t in tiers.iter_mut() {
            let s = if t.hospitality { hosp_seat.clone() } else { style.clone() };
            let spec = seat_spec(cat, &s)?;
            t.seat_pitch = t.seat_pitch.max(num(spec, "width", 0.5));
            let min_tread = num(spec, "min_tread", 0.7);
            if t.kind == Kind::Seated && t.tread < min_tread - 1e-9 {
                let w = format!("{} tier {}: {s} needs {min_tread} m rows for a 305 mm seatway; tier has {} m", run.name, t.index, t.tread);
                if !warnings.contains(&w) {
                    warnings.push(w);
                }
            }
            tier_styles.push(s);
        }
        let o = stands::build_run(run, ri, &tiers);
        st.parts.push(Part { name: format!("Stand_{}", run.name), mesh: o.mesh.clone(), mats: mats.clone(), smooth_deg: if run.curved() { 35.0 } else { 0.0 } });
        let far = if run.name == "north" || run.name == "south" || run.curved() { p.width } else { p.length };
        let roof_spec = sc.cfg.get("roof").or_else(|| cfg.get("roof")).cloned().unwrap_or(Value::Null);
        let roof_spec = resolve_preset(&cat.roofs, &roof_spec)?;
        let mut roof_info = None;
        if let Some(rs) = &roof_spec {
            let roof = roofs::solve(&tiers, rs, df, far);
            let ro = roofs::build(run, &roof, &o.bay_list, !run.curved());
            let membrane = text(rs, "deck", "metal") == "membrane";
            let rmats = vec![
                Mat::tint(if membrane { "TINT_roof_membrane" } else { "TINT_roof" }, &hexv(rs, "color", "#e3e4e2")),
                Mat::named("Roof_Polycarbonate"),
                Mat::tint("TINT_roof_fascia", &hexv(rs, "fascia_color", "#1b1d20")),
                Mat::tint("Roof_Underside", &hexv(rs, "underside_color", "#d8d8d4")),
            ];
            if !ro.mesh.faces.is_empty() {
                st.parts.push(Part { name: format!("Roof_{}", run.name), mesh: ro.mesh.clone(), mats: rmats, smooth_deg: 35.0 });
            }
            if let Some(t) = tubes_part(&format!("RoofSteel_{}", run.name), &ro.tubes, 8, Mat::tint("Roof_Steel", &hexv(rs, "steel_color", "#eceeee"))) {
                st.parts.push(t);
            }
            st.sockets.extend(ro.sockets.iter().cloned());
            st.report.roofs.push((run.name.clone(), roof.raised_for_sightline, roof.raised_for_headroom, roof.clearance_to_sightline));
            roof_info = Some(RoofInfo { tail_s: ro.tail_s, tail_top: ro.tail_top });
            out.roofs.push((run.name.clone(), ro));
        }
        let facade_spec = sc.cfg.get("facade").or_else(|| cfg.get("facade")).cloned().unwrap_or(Value::Null);
        if let Some(fs) = resolve_preset(&cat.facades, &facade_spec)? {
            let fo = facades::build(run, &tiers, &o.bay_list, &fs, roof_info, (!curved, !curved));
            st.parts.push(Part { name: format!("Facade_{}", run.name), mesh: fo.mesh, mats: facade_mats(&fs, st.night), smooth_deg: 35.0 });
            if let Some(t) = tubes_part(&format!("FacadeFrame_{}", run.name), &fo.tubes, 6, Mat::named("Facade_Frame")) {
                st.parts.push(t);
            }
            st.sockets.extend(fo.sockets);
            out.facade_lines.push((run.name.clone(), fo.line_s));
        }
        // balustrade glass on the tier fronts
        let mut glass = MeshAcc::new();
        for t in &tiers {
            if t.glass <= 0.0 {
                continue;
            }
            let (s, z) = (t.s0 - WALL + 0.12, t.h[0] + t.parapet);
            let sec = Section::uniform(vec![DVec2::new(s, z), DVec2::new(s + 0.02, z), DVec2::new(s + 0.02, z + t.glass), DVec2::new(s, z + t.glass)], 0);
            glass.add_prism(run, run.t0, run.t1, &sec, 0);
        }
        if !glass.faces.is_empty() {
            st.parts.push(Part { name: format!("Glass_{}", run.name), mesh: glass, mats: vec![Mat::named("Balustrade_Glass")], smooth_deg: 35.0 });
        }
        let mut rails = o.rails.clone();
        rails.extend(barrier_lines(&o.barriers));
        if let Some(t) = tubes_part(&format!("Rails_{}", run.name), &rails, 8, Mat::named("Galvanised")) {
            st.parts.push(t);
        }
        if !o.nosings.is_empty() {
            st.parts.push(Part { name: format!("Nosings_{}", run.name), mesh: nosings_mesh(&o.nosings), mats: vec![Mat::named("Step_Nosing")], smooth_deg: 0.0 });
        }
        st.sockets.extend(o.sockets.iter().cloned());
        for a_ in &o.anchors {
            styles.push(tier_styles[a_.tier].clone());
        }
        all.extend(o.anchors.iter().cloned());
        run_lengths.push(run.length());
        tier_rows.push(tiers.iter().map(|t| t.rows).collect());
        vom_top.push(tiers.iter().map(|t| vom_rows(t).map(|(r0, k)| r0 + k)).collect());
        for t in &tiers {
            if t.kind == Kind::Standing {
                standing += STANDING_PER_M * t.rows as f64 * (run.t1 - run.t0) * run.metric(t.s0) * 0.88;
            }
        }
        st.report.tiers.push((run.name.clone(), tier_report(&tiers)));
        if run.name == "south" {
            let t = tiers.iter().min_by(|x, y| (x.h[0] - 16.0).abs().partial_cmp(&(y.h[0] - 16.0).abs()).unwrap()).unwrap();
            let (pnt, n) = run.frame(0.5 * (run.t0 + run.t1));
            let q = pnt + n * (t.s0 - 1.2);
            let m = DVec3::new(q.x, q.y, t.h[0] + t.parapet + 1.6);
            out.camera_mount = Some(m);
            st.sockets.push(Socket { kind: "camera", id: "main".into(), loc: m, n: DVec2::new(0.0, -1.0), props: vec![] });
        }
        out.runs.push((run.clone(), tiers));
    }

    // seat colours: base pattern per run, mosaics, finish
    let run_index = |name: &str| out.runs.iter().position(|(r, _)| r.name == name);
    let default = cfg.get("colors").cloned().unwrap_or(json!({"pattern": "solid", "color": "#c8102e"}));
    let mut colors = vec![DVec3::ZERO; all.len()];
    for (ri, (run, _)) in out.runs.iter().enumerate() {
        let idx: Vec<usize> = (0..all.len()).filter(|&i| all[i].run == ri).collect();
        if idx.is_empty() {
            continue;
        }
        let spec = sides.iter().find(|s| s.name == run.name).and_then(|s| s.cfg.get("colors")).cloned().unwrap_or_else(|| default.clone());
        let sub: Vec<&Anchor> = idx.iter().map(|&i| &all[i]).collect();
        let cols = patterns::base_colors(&sub, &spec, &tier_rows[ri], (crc32(&run.name) % 1000) as u64)?;
        for (k, &i) in idx.iter().enumerate() {
            colors[i] = cols[k];
        }
    }
    for mz in cfg.get("mosaics").and_then(|v| v.as_array()).cloned().unwrap_or_default() {
        let rname = text(&mz, "run", "");
        let Some(ri) = run_index(rname) else {
            warnings.push(format!("mosaic on {rname:?}: no such stand"));
            continue;
        };
        let tier = mz.get("tier").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        if tier >= tier_rows[ri].len() {
            warnings.push(format!("mosaic on {rname} tier {tier}: the stand has {} tiers", tier_rows[ri].len()));
            continue;
        }
        let mut mz = mz.clone();
        let rows = tier_rows[ri][tier];
        if mz.get("rows").is_none() {
            if let Some(top) = vom_top[ri][tier] {
                if rows - top >= 7 {
                    mz["rows"] = json!([top, rows - 1]); // keep letters clear of the vomitory mouths
                }
            }
        }
        let pic = match mz.get("image").and_then(|v| v.as_str()) {
            Some(path) => match images(path) {
                Some(p) => Some(p),
                None => {
                    warnings.push(format!("mosaic image {path:?} could not be read"));
                    continue;
                }
            },
            None => mz.get("pixels").and_then(picture_from_json),
        };
        patterns::apply_mosaic(&mut colors, &all, &mz, ri, run_lengths[ri], &tier_rows[ri], pic.as_ref());
    }
    let finished = patterns::finish(&colors, num(cfg, "seat_fade", 0.0), 0);
    let mut by_style: BTreeMap<String, usize> = BTreeMap::new();
    let mut groups: BTreeMap<String, Vec<Placement>> = BTreeMap::new();
    for (i, a_) in all.iter().enumerate() {
        let spec = seat_spec(cat, &styles[i])?;
        let vars = variants(spec);
        let vi = variant_index(spec, a_.seat, a_.of).min(vars.len() - 1);
        let sx = if spec.get("stretch").and_then(|v| v.as_bool()).unwrap_or(false) { a_.pitch / num(spec, "width", 0.5) } else { 1.0 };
        *by_style.entry(styles[i].clone()).or_default() += 1;
        groups.entry(format!("seat.{}.{}", styles[i], vars[vi])).or_default().push(Placement {
            pos: a_.pos,
            yaw: a_.yaw,
            tilt: 0.0,
            scale: DVec3::new(sx, 1.0, 1.0),
            color: Some(finished[i]),
        });
    }
    for (key, items) in groups {
        st.props.push(PropSet { key, items, tints: vec![], net: None });
    }

    // apron between the run-off and the stand fronts, and the plaza outside
    let apron_kind = text(cfg, "apron", "turf");
    let (ix, iy) = (p.length / 2.0 + p.runoff_end, p.width / 2.0 + p.runoff_side);
    let n = 360;
    let mut apron = MeshAcc::new();
    for k in 0..n {
        let ang = 2.0 * PI * k as f64 / n as f64;
        let d = DVec2::new(ang.cos(), ang.sin());
        let q = rect_hit(d, ix, iy, 0.0);
        apron.verts.push(DVec3::new(q.x, q.y, -0.005));
    }
    for k in 0..n {
        let ang = 2.0 * PI * k as f64 / n as f64;
        let d = DVec2::new(ang.cos(), ang.sin());
        let q = rect_hit(d, a + 0.3, b + 0.3, if curved { r_corner } else { 0.0 });
        apron.verts.push(DVec3::new(q.x, q.y, -0.005));
    }
    for j in 0..n as u32 {
        let jn = (j + 1) % n as u32;
        let f = vec![j, jn, n as u32 + jn, n as u32 + j];
        let uv = f.iter().map(|&i| apron.verts[i as usize].truncate()).collect();
        apron.add_face(f, 0, Some(uv));
    }
    let apron_mat = match apron_kind {
        "track" => Mat::named("Apron_Track"),
        "concrete" => Mat::named("Apron_Concrete"),
        _ => Mat::named("Pitch_Grass"),
    };
    st.parts.push(Part { name: "Apron".into(), mesh: apron, mats: vec![apron_mat], smooth_deg: 0.0 });
    let plaza_default = !out.facade_lines.is_empty();
    if cfg.get("plaza").and_then(|v| v.as_bool()).unwrap_or(plaza_default) {
        let line = |names: [&str; 2]| out.facade_lines.iter().filter(|(n, _)| names.contains(&n.as_str())).map(|(_, s)| *s).fold(20.0, f64::max);
        let (reach_x, reach_y) = (line(["east", "west"]), line(["north", "south"]));
        let (hx, hy, r) = (a + reach_x + 28.0, b + reach_y + 28.0, (if curved { r_corner } else { 0.0 }).max(20.0) + 28.0);
        let n = 256;
        let mut plaza = MeshAcc::new();
        plaza.verts.push(DVec3::new(0.0, 0.0, -0.012));
        for k in 0..n {
            let ang = 2.0 * PI * k as f64 / n as f64;
            let q = rect_hit(DVec2::new(ang.cos(), ang.sin()), hx, hy, r);
            plaza.verts.push(DVec3::new(q.x, q.y, -0.012));
        }
        for j in 0..n as u32 {
            let f = vec![0, 1 + j, 1 + (j + 1) % n as u32];
            let uv = f.iter().map(|&i| plaza.verts[i as usize].truncate()).collect();
            plaza.add_face(f, 0, Some(uv));
        }
        st.parts.push(Part { name: "Plaza".into(), mesh: plaza, mats: vec![Mat::named("Plaza_Paving")], smooth_deg: 0.0 });
    }
    let seated = all.len();
    st.report.capacity = Capacity { seated, standing: standing as usize, total: seated + standing as usize, by_style };
    st.report.warnings.extend(warnings);
    Ok(out)
}

fn picture_from_json(v: &Value) -> Option<Picture> {
    // {"w": W, "h": H, "rgb": ["#rrggbb" or null, ...]} (row 0 = top)
    let w = v.get("w")?.as_u64()? as usize;
    let h = v.get("h")?.as_u64()? as usize;
    let px = v.get("rgb")?.as_array()?.iter().map(|c| c.as_str().map(patterns::hex_rgb)).collect::<Vec<_>>();
    (px.len() == w * h).then_some(Picture { w, h, px })
}

// -- ads ----------------------------------------------------------------------------------------

/// Quad strip between two section points along a run, offset toward the pitch; per-loop UVs with
/// u running against t (seen from the pitch a run reads right-to-left).
#[allow(clippy::too_many_arguments)]
fn ad_strip(run: &Run, t0: f64, t1: f64, s_bot: f64, z_bot: f64, s_top: f64, z_top: f64, u0: f64, u_per_m: f64, lift: f64) -> (Vec<DVec3>, Vec<[u32; 4]>, Vec<DVec2>) {
    let d = DVec2::new(s_top - s_bot, z_top - z_bot);
    let mut nrm = DVec2::new(-d.y, d.x) / d.length();
    if nrm.x > 0.0 {
        nrm = -nrm;
    }
    let (sb, zb) = (s_bot + nrm.x * lift, z_bot + nrm.y * lift);
    let (stp, zt) = (s_top + nrm.x * lift, z_top + nrm.y * lift);
    let ts = run.samples(t0, t1, s_bot.max(s_top));
    let mut v = Vec::new();
    let mut uv = Vec::new();
    for &t in &ts {
        let (p, n) = run.frame(t);
        let along = (t - run.t0) * run.metric(s_bot);
        for (s, z, vv) in [(sb, zb, 0.0), (stp, zt, 1.0)] {
            let q = p + n * s;
            v.push(DVec3::new(q.x, q.y, z));
            uv.push(DVec2::new(u0 - along * u_per_m, vv));
        }
    }
    let f = (0..ts.len() as u32 - 1).map(|j| [2 * j, 2 * j + 2, 2 * j + 3, 2 * j + 1]).collect();
    (v, f, uv)
}

fn add_vertex_uv_quads(acc: &mut MeshAcc, v: &[DVec3], f: &[[u32; 4]], uv: &[DVec2], tag: i32) {
    let base = acc.verts.len() as u32;
    acc.verts.extend_from_slice(v);
    for q in f {
        acc.add_face(q.iter().map(|i| i + base).collect(), tag, Some(q.iter().map(|&i| uv[i as usize]).collect()));
    }
}

fn build_ads(st: &mut Stadium, cfg: &Value, cat: &Catalog, stands: Option<&StandsOut>, gap: f64) -> Result<(), String> {
    let p = st.pitch.clone();
    let keys: Vec<String> = match cfg.get("brands").and_then(|v| v.as_array()) {
        Some(a) if !a.is_empty() => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => cat.brands.iter().filter(|(_, b)| b.get("tier").and_then(|t| t.as_i64()).unwrap_or(1) > 0).map(|(k, _)| k.clone()).collect(),
    };
    for k in &keys {
        if !cat.brands.contains_key(k) {
            return Err(format!("unknown brand {k:?}"));
        }
    }
    if keys.is_empty() {
        return Ok(());
    }
    let n_atlas = keys.len();
    let cam = st.camera.0;
    let strength = if st.night { 1.2 } else { 4.0 };
    let mut slots: Vec<AdSlot> = Vec::new();
    let mut slot = |st: &mut Stadium, kind: &'static str, id: String, centre: DVec3, facing: DVec3, area: f64, brand: String| {
        let w = placement::tv_weight(centre, facing, cam);
        let yaw = (-facing.x).atan2(facing.y);
        st.sockets.push(Socket { kind: "ad_slot", id: id.clone(), loc: centre, n: DVec2::new(yaw.cos(), yaw.sin()), props: vec![("area_m2", round_to(area, 2)), ("tv_weight", w)] });
        slots.push(AdSlot { kind, id, area_m2: round_to(area, 2), tv_weight: w, brand });
    };
    let led_mat = |key: &str, s: f64| Mat { key: key.into(), color: None, emission: Some((DVec3::ONE, s)), texture: Some(Tex::BoardAtlas(keys.clone())) };
    // with 3D carpets the end boards stand further back so the carpets have room for a ~1 m
    // virtual board ("end_offset": metres behind the goal line, or "auto")
    let per = cfg.get("perimeter").cloned().unwrap_or(json!({"type": "led"}));
    let kind = text(&per, "type", "led").to_string();
    let carpets_on = cfg.get("carpets").is_some_and(truthy);
    let carpet_width = cfg.get("carpets").map(|c| num(c, "width", 5.0)).unwrap_or(5.0);
    let end_off = match per.get("end_offset") {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(placement::BOARD_END_OFFSET),
        _ if carpets_on => {
            let reach = stands.map(|s| s.a - p.length / 2.0 - 0.8).unwrap_or(p.runoff_end + 3.0);
            placement::end_board_offset(cam, &p, carpet_width, reach, placement::CARPET_HEIGHT)
        }
        _ => placement::BOARD_END_OFFSET,
    };
    st.report.end_boards_m = end_off;
    if kind != "none" {
        let runs = placement::board_runs(&p, gap, placement::BOARD_SIDE_OFFSET, end_off);
        if kind == "led" {
            let mut housing = MeshAcc::new();
            let mut faces = MeshAcc::new();
            for run in &runs {
                let poly = vec![DVec2::new(0.0, 0.0), DVec2::new(0.42, 0.0), DVec2::new(0.42, 0.92), DVec2::new(0.2, 1.0), DVec2::new(0.17, 0.97), DVec2::new(0.0, 0.1)];
                housing.add_prism(run, run.t0, run.t1, &Section::uniform(poly, 0), 0);
                let offset = (crc32(&run.name) as usize % n_atlas) as f64 / n_atlas as f64;
                let (v, f, uv) = ad_strip(run, run.t0, run.t1, 0.02, 0.12, 0.165, 0.95, offset, 1.0 / (CREATIVE_M * n_atlas as f64), 0.004);
                add_vertex_uv_quads(&mut faces, &v, &f, &uv, 0);
                let length = run.t1 - run.t0;
                for k in 0..(length / CREATIVE_M).floor() as usize {
                    let (q, n) = run.frame(run.t0 + (k as f64 + 0.5) * CREATIVE_M);
                    slot(st, "perimeter_led", format!("{}_{k}", run.name), DVec3::new(q.x, q.y, 0.5), DVec3::new(-n.x, -n.y, 0.0), CREATIVE_M * 0.85, "rotation".into());
                }
            }
            st.parts.push(Part { name: "BoardHousing".into(), mesh: housing, mats: vec![Mat::named("Board_Housing")], smooth_deg: 0.0 });
            st.parts.push(Part { name: "BoardLED".into(), mesh: faces, mats: vec![led_mat("Ads_LED", strength)], smooth_deg: 0.0 });
        } else {
            let panel = 7.2;
            let mut legs = Vec::new();
            let mut boards = MeshAcc::new();
            let mut backs = MeshAcc::new();
            for run in &runs {
                let length = run.t1 - run.t0;
                let n_p = 1usize.max((length / (panel + 0.2)).floor() as usize);
                let pad = (length - n_p as f64 * (panel + 0.2)) / 2.0;
                for k in 0..n_p {
                    let ta = run.t0 + pad + k as f64 * (panel + 0.2);
                    let bk = &keys[(k + crc32(&run.name) as usize) % keys.len()];
                    let bi = keys.iter().position(|x| x == bk).unwrap();
                    let (v, f, mut uv) = ad_strip(run, ta, ta + panel, 0.0, 0.05, 0.24, 0.92, 0.0, 0.0, 0.0);
                    for (i, u) in uv.iter_mut().enumerate() {
                        u.x = if i / 2 == 0 { (bi + 1) as f64 / n_atlas as f64 } else { bi as f64 / n_atlas as f64 };
                    }
                    add_vertex_uv_quads(&mut boards, &v, &f, &uv, 0);
                    // the panel's back: a printed board is a sheet on a frame, not a see-through quad
                    let back = vec![DVec2::new(0.02, 0.05), DVec2::new(0.035, 0.05), DVec2::new(0.275, 0.92), DVec2::new(0.26, 0.92)];
                    backs.add_prism(run, ta, ta + panel, &Section::uniform(back, 0), 0);
                    for t in [ta + 0.4, ta + panel / 2.0, ta + panel - 0.4] {
                        let (q, nrm) = run.frame(t);
                        let top = q + nrm * 0.24;
                        let foot = q + nrm * 0.75;
                        legs.push((vec![DVec3::new(top.x, top.y, 0.9), DVec3::new(foot.x, foot.y, 0.0)], 0.02));
                    }
                    let (q, nrm) = run.frame(ta + panel / 2.0);
                    slot(st, "perimeter_static", format!("{}_{k}", run.name), DVec3::new(q.x, q.y, 0.5), DVec3::new(-nrm.x, -nrm.y, 0.0), panel * 0.9, bk.clone());
                }
            }
            st.parts.push(Part { name: "Boards".into(), mesh: boards, mats: vec![Mat { key: "Ads_Print".into(), color: None, emission: None, texture: Some(Tex::BoardAtlas(keys.clone())) }], smooth_deg: 0.0 });
            if let Some(t) = tubes_part("BoardLegs", &legs, 6, Mat::named("Galvanised")) {
                st.parts.push(t);
            }
            st.parts.push(Part { name: "BoardBacks".into(), mesh: backs, mats: vec![Mat::named("Board_Housing")], smooth_deg: 0.0 });
        }
    }
    if let Some(cp) = cfg.get("carpets").filter(|v| truthy(v)) {
        let width = num(cp, "width", 5.0);
        let ckeys: Vec<String> = cp.get("brands").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).filter(|v: &Vec<String>| !v.is_empty()).unwrap_or_else(|| keys.clone());
        let limit = p.length / 2.0 + end_off - 0.4;
        for (k, (_, _, base)) in placement::carpet_spots(&p, width, 0.8).into_iter().enumerate() {
            let bk = ckeys[k % ckeys.len()].clone();
            let (base, h) = placement::place_carpet(cam, base, width, 1.1, &p, limit);
            let pc = p.clone();
            let (v, f, uv) = placement::carpet(cam, base, width, h, 32, 10, &move |x, y| pc.crown_z(x, y));
            let mut m = MeshAcc::new();
            add_vertex_uv_quads(&mut m, &v, &f, &uv, 0);
            st.parts.push(Part { name: format!("Carpet_{k}"), mesh: m, mats: vec![Mat { key: "Carpet".into(), color: None, emission: None, texture: Some(Tex::Carpet(bk.clone())) }], smooth_deg: 0.0 });
            let to_cam = cam - base;
            slot(st, "carpet_3d", format!("carpet_{k}"), DVec3::new(base.x, base.y, 0.0), to_cam / to_cam.length(), width * h, bk);
        }
    }
    if let (Some(rb), Some(stands)) = (cfg.get("ribbons").filter(|v| v.is_object()), stands) {
        let wanted: Vec<u64> = rb.get("tiers").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|t| t.as_u64()).collect()).unwrap_or_else(|| vec![1, 2, 3]);
        let mut ribbons = MeshAcc::new();
        for (run, tiers) in &stands.runs {
            for t in tiers {
                if !wanted.contains(&(t.index as u64)) || t.index == 0 {
                    continue;
                }
                let top = t.h[0] + t.parapet;
                let s_face = t.s0 - WALL;
                let (v, f, uv) = ad_strip(run, run.t0, run.t1, s_face, top - 0.98, s_face, top - 0.06, (crc32(&run.name) as usize % n_atlas) as f64 / n_atlas as f64, 1.0 / (CREATIVE_M * n_atlas as f64), 0.03);
                add_vertex_uv_quads(&mut ribbons, &v, &f, &uv, 0);
                let length = (run.t1 - run.t0) * run.metric(s_face);
                let (q, n) = run.frame(0.5 * (run.t0 + run.t1));
                let c = q + n * s_face;
                slot(st, "led_ribbon", format!("{}_T{}", run.name, t.index), DVec3::new(c.x, c.y, top - 0.5), DVec3::new(-n.x, -n.y, 0.0), length * 0.92, "rotation".into());
            }
        }
        if !ribbons.faces.is_empty() {
            st.parts.push(Part { name: "Ribbons".into(), mesh: ribbons, mats: vec![led_mat("Ads_Ribbon", strength * 0.8)], smooth_deg: 35.0 });
        }
    }
    for s in &slots {
        let t = st.report.ad_totals.entry(s.kind.to_string()).or_default();
        t.slots += 1;
        t.area_m2 = round_to(t.area_m2 + s.area_m2, 1);
        t.tv_weighted_m2 = round_to(t.tv_weighted_m2 + s.area_m2 * s.tv_weight, 1);
    }
    st.report.ads.extend(slots);
    Ok(())
}

// -- screens ------------------------------------------------------------------------------------

fn screen_model(w: f64, h: f64) -> MeshAcc {
    let depth = 0.9;
    let mut m = MeshAcc::new();
    add_box(&mut m, DVec3::new(-(w + 0.5) / 2.0, -depth, -(h + 0.5) / 2.0), DVec3::new((w + 0.5) / 2.0, 0.0, (h + 0.5) / 2.0), 0);
    let b = m.verts.len() as u32;
    m.verts.extend([DVec3::new(-w / 2.0, 0.01, -h / 2.0), DVec3::new(w / 2.0, 0.01, -h / 2.0), DVec3::new(w / 2.0, 0.01, h / 2.0), DVec3::new(-w / 2.0, 0.01, h / 2.0)]);
    // wound to face +Y; seen from the front local -X is the viewer's right, so u runs +X -> -X
    m.add_face(vec![b, b + 3, b + 2, b + 1], 1, Some(vec![DVec2::new(1.0, 0.0), DVec2::new(1.0, 1.0), DVec2::new(0.0, 1.0), DVec2::new(0.0, 0.0)]));
    m
}

fn build_screens(st: &mut Stadium, cfg: &Value, stands: Option<&StandsOut>) -> Result<(), String> {
    let p = st.pitch.clone();
    let (w, h) = (num(cfg, "width", 18.0), num(cfg, "height", 8.0));
    let (l, wp) = (p.length, p.width);
    let led = Mat { key: "Screen_LED".into(), color: None, emission: Some((DVec3::ONE, if st.night { 1.6 } else { 3.5 })), texture: Some(Tex::Scoreboard) };
    let mats = vec![Mat::named("Screen_Body"), led];
    let mut tubes: Vec<Polyline> = Vec::new();
    let mut count = 0;
    let mut put = |st: &mut Stadium, name: String, pos: DVec3, yaw: f64, tilt: f64| {
        let mut m = screen_model(w, h);
        transform(&mut m, &Placement { pos, yaw, tilt, scale: DVec3::ONE, color: None });
        st.parts.push(Part { name: name.clone(), mesh: m, mats: mats.clone(), smooth_deg: 0.0 });
        st.sockets.push(Socket { kind: "screen", id: name, loc: pos, n: DVec2::new(yaw.cos(), yaw.sin()), props: vec![("width", w), ("height", h)] });
        count += 1;
    };
    let yaw_to_centre = |x: f64, y: f64| x.atan2(-y);
    let roof_line = |name: &str| stands.and_then(|s| s.roofs.iter().find(|r| r.0 == name)).map(|r| r.1.edge_line.clone());
    let hang = |st: &mut Stadium, tubes: &mut Vec<Polyline>, put: &mut dyn FnMut(&mut Stadium, String, DVec3, f64, f64), name: String, line: &[DVec3]| {
        let mid = line[line.len() / 2];
        let inward = -DVec2::new(mid.x, mid.y) / mid.x.hypot(mid.y);
        let top = mid.z - 0.6;
        let centre = DVec3::new(mid.x + inward.x * 1.5, mid.y + inward.y * 1.5, top - h / 2.0);
        let yaw = yaw_to_centre(centre.x, centre.y);
        put(st, name, centre, yaw, (-8.0f64).to_radians());
        let side = DVec2::new(-inward.y, inward.x);
        for sgn in [-1.0, 1.0] {
            let a = centre.truncate() + side * sgn * w * 0.4;
            tubes.push((vec![DVec3::new(a.x, a.y, top + 0.2), DVec3::new(a.x, a.y, mid.z)], 0.04));
        }
    };
    let on_frame = |st: &mut Stadium, tubes: &mut Vec<Polyline>, put: &mut dyn FnMut(&mut Stadium, String, DVec3, f64, f64), name: String, xy: DVec2, bottom: f64, ground: f64| {
        let yaw = yaw_to_centre(xy.x, xy.y);
        put(st, name, DVec3::new(xy.x, xy.y, bottom + h / 2.0 + 0.25), yaw, (-4.0f64).to_radians());
        let side = DVec2::new(yaw.cos(), yaw.sin());
        let back = -DVec2::new(-yaw.sin(), yaw.cos());
        for sgn in [-0.38, 0.38] {
            let q = xy + side * sgn * w + back * 1.2;
            let r = q - back * 1.1;
            tubes.push((vec![DVec3::new(q.x, q.y, ground), DVec3::new(q.x, q.y, bottom + h)], 0.22));
            tubes.push((vec![DVec3::new(q.x, q.y, bottom + 0.2), DVec3::new(r.x, r.y, bottom + 0.2)], 0.08));
            tubes.push((vec![DVec3::new(q.x, q.y, bottom + h), DVec3::new(r.x, r.y, bottom + h)], 0.08));
        }
    };
    let placement_kind = text(cfg, "placement", "ends").to_string();
    match placement_kind.as_str() {
        "ends" | "stand_top" => {
            for (side, x) in [("east", 1.0), ("west", -1.0)] {
                let name = format!("Screen_{side}");
                let run = stands.and_then(|s| s.runs.iter().find(|r| r.0.name == side));
                if placement_kind == "ends" && roof_line(side).is_some() {
                    hang(st, &mut tubes, &mut put, name, &roof_line(side).unwrap());
                } else if let Some((run, tiers)) = run {
                    let t_mid = 0.5 * (run.t0 + run.t1);
                    let (q, n) = run.frame(t_mid);
                    let s = stand_depth(tiers) - 1.0;
                    let top = tiers.iter().map(|t| t.h_last()).fold(f64::MIN, f64::max) + 1.1;
                    let xy = q + n * s;
                    on_frame(st, &mut tubes, &mut put, name, xy, top + 1.5, top);
                } else {
                    on_frame(st, &mut tubes, &mut put, name, DVec2::new(x * (l / 2.0 + p.runoff_end + 4.0), 0.0), 4.0, 0.0);
                }
            }
        }
        "corners" => {
            for (sx, sy, arc) in [(1.0, 1.0, "NE"), (-1.0, -1.0, "SW")] {
                let name = format!("Screen_{arc}");
                if let Some(line) = roof_line(arc) {
                    hang(st, &mut tubes, &mut put, name, &line);
                } else {
                    let (a, b) = stands.map(|s| (s.a, s.b)).unwrap_or((l / 2.0 + 10.0, wp / 2.0 + 8.0));
                    on_frame(st, &mut tubes, &mut put, name, DVec2::new(sx * (a + 4.0), sy * (b + 4.0)), 8.0, 0.0);
                }
            }
        }
        "center" => {
            let roofs = stands.map(|s| &s.roofs[..]).unwrap_or(&[]);
            if roofs.is_empty() {
                return Err("a centre-hung screen needs roofs to hang from".into());
            }
            let z_edge = roofs.iter().map(|r| r.1.edge_line.iter().map(|q| q.z).fold(f64::MAX, f64::min)).fold(f64::MAX, f64::min);
            let cz = (z_edge - 6.0 - h / 2.0).min(25.0 + h / 2.0);
            let half = w / 2.0 + 0.3;
            let mut caps = MeshAcc::new();
            for z in [cz - h / 2.0 - 0.4, cz + h / 2.0 + 0.4] {
                let e = half + 0.3;
                add_box(&mut caps, DVec3::new(-e, -e, z - 0.3), DVec3::new(e, e, z + 0.3), 0);
            }
            st.parts.push(Part { name: "ScreenCaps".into(), mesh: caps, mats: vec![Mat::named("Screen_Body")], smooth_deg: 0.0 });
            for (k, (ux, uy)) in [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)].into_iter().enumerate() {
                put(st, format!("Screen_centre_{k}"), DVec3::new(ux * half, uy * half, cz), (-ux).atan2(uy), 0.0);
            }
            for r in roofs {
                let mid = r.1.edge_line[r.1.edge_line.len() / 2];
                tubes.push((vec![DVec3::new(0.0, 0.0, cz + h / 2.0 + 0.5), mid], 0.035));
            }
        }
        k => return Err(format!("unknown screen placement {k:?}")),
    }
    if let Some(t) = tubes_part("ScreenSteel", &tubes, 8, Mat::named("Galvanised")) {
        st.parts.push(t);
    }
    st.report.screens = count;
    Ok(())
}

// -- dugouts and tunnel ------------------------------------------------------------------------

struct DugoutCfg {
    style: String,
    seats: usize,
    seat: String,
    color: String,
}

fn dugout_cfg(cfg: &Value) -> DugoutCfg {
    let style = text(cfg, "style", "shell").to_string();
    let seat = match cfg.get("seat").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None if style == "shelter" => "bench".into(),
        None => "padded_vip".into(),
    };
    DugoutCfg { seats: num(cfg, "seats", 14.0) as usize, seat, color: hexv(cfg, "color", "#c8102e"), style }
}

pub fn dugout_length(cfg: &Value, cat: &Catalog) -> Result<f64, String> {
    let d = dugout_cfg(cfg);
    let per_row = d.seats.div_ceil(2);
    Ok(per_row as f64 * num(seat_spec(cat, &d.seat)?, "width", 0.5) + 1.0)
}

fn build_dugouts(st: &mut Stadium, cfg: &Value, cat: &Catalog) -> Result<(), String> {
    let d = dugout_cfg(cfg);
    let lx = dugout_length(cfg, cat)?;
    let w_p = st.pitch.width;
    let y = -(w_p / 2.0 + DUGOUT_FRONT);
    let xs = [-(DUGOUT_GAP / 2.0 + lx / 2.0), DUGOUT_GAP / 2.0 + lx / 2.0];
    let depth = DUGOUT_DEPTH;
    // one model, front centre at the origin, +Y towards the pitch
    let mut m = MeshAcc::new();
    add_box(&mut m, DVec3::new(-lx / 2.0, -depth, 0.0), DVec3::new(lx / 2.0, 0.0, 0.12), 0);
    add_box(&mut m, DVec3::new(-lx / 2.0, -depth, 0.12), DVec3::new(lx / 2.0, -1.15, 0.42), 0);
    let mut frame_lines: Vec<Polyline> = Vec::new();
    let mut mats = vec![Mat::named("Dugout_Slab"), Mat::named("Dugout_Polycarbonate"), Mat::tint("Dugout_Frame", if d.style == "shell" { "#d8dadb" } else { "#2b2d30" }), Mat::named("Dugout_Brick"), Mat::tint("TINT_dugout", &d.color)];
    match d.style.as_str() {
        "shell" => {
            let prof: Vec<DVec2> = linspace(PI, PI / 2.0 + 0.25, 14).iter().map(|a| DVec2::new(-depth + 0.1 + 2.35 * (1.0 + a.cos()), 0.12 + 2.05 * a.sin())).collect();
            let xs_c = linspace(-lx / 2.0, lx / 2.0, 13);
            let k = prof.len() as u32;
            let base = m.verts.len() as u32;
            for &x in &xs_c {
                for q in &prof {
                    m.verts.push(DVec3::new(x, q.x, q.y));
                }
            }
            for i in 0..12u32 {
                for j in 0..k - 1 {
                    let f = vec![base + i * k + j, base + i * k + j + 1, base + (i + 1) * k + j + 1, base + (i + 1) * k + j];
                    let uv = f.iter().map(|&v| { let p = m.verts[v as usize]; DVec2::new(p.x, p.z) }).collect();
                    m.add_face(f, 1, Some(uv));
                }
            }
            for x in [-lx / 2.0, lx / 2.0] {
                let b = m.verts.len() as u32;
                m.verts.push(DVec3::new(x, -depth + 0.1, 0.12));
                for q in &prof {
                    m.verts.push(DVec3::new(x, q.x, q.y));
                }
                m.verts.push(DVec3::new(x, prof[prof.len() - 1].x, 0.12));
                let n = m.verts.len() as u32 - b;
                m.add_face((b..b + n).collect(), 1, None);
            }
            for x in linspace(-lx / 2.0, lx / 2.0, 5) {
                frame_lines.push((prof.iter().map(|q| DVec3::new(x, q.x, q.y)).collect(), 0.03));
            }
        }
        "box" => {
            add_box(&mut m, DVec3::new(-lx / 2.0, -depth - 0.25, 0.0), DVec3::new(lx / 2.0, -depth, 2.2), 3);
            add_box(&mut m, DVec3::new(-lx / 2.0 - 0.25, -depth - 0.25, 0.0), DVec3::new(-lx / 2.0, 0.0, 2.2), 3);
            add_box(&mut m, DVec3::new(lx / 2.0, -depth - 0.25, 0.0), DVec3::new(lx / 2.0 + 0.25, 0.0, 2.2), 3);
            add_box(&mut m, DVec3::new(-lx / 2.0 - 0.3, -depth - 0.3, 2.2), DVec3::new(lx / 2.0 + 0.3, 0.35, 2.38), 4);
        }
        _ => {
            mats[1] = Mat::tint("Dugout_Polycarbonate", "#dfe8ea");
            for x in [-lx / 2.0, lx / 2.0] {
                for yy in [-depth + 0.05, -0.05] {
                    frame_lines.push((vec![DVec3::new(x, yy, 0.12), DVec3::new(x, yy, 2.15)], 0.04));
                }
            }
            add_box(&mut m, DVec3::new(-lx / 2.0, -depth + 0.02, 0.3), DVec3::new(lx / 2.0, -depth + 0.04, 2.1), 1);
            add_box(&mut m, DVec3::new(-lx / 2.0 - 0.01, -depth, 0.3), DVec3::new(-lx / 2.0 + 0.01, -0.1, 2.1), 1);
            add_box(&mut m, DVec3::new(lx / 2.0 - 0.01, -depth, 0.3), DVec3::new(lx / 2.0 + 0.01, -0.1, 2.1), 1);
            add_box(&mut m, DVec3::new(-lx / 2.0 - 0.15, -depth - 0.15, 2.15), DVec3::new(lx / 2.0 + 0.15, 0.25, 2.22), 4);
        }
    }
    let spec = seat_spec(cat, &d.seat)?.clone();
    let vars = variants(&spec);
    let per_row = d.seats.div_ceil(2);
    let sw = num(&spec, "width", 0.5);
    let mut seat_groups: BTreeMap<String, Vec<Placement>> = BTreeMap::new();
    for (side, &x) in ["home", "away"].iter().zip(&xs) {
        let place = Placement::at(DVec3::new(x, y, 0.0), 0.0);
        let mut mm = m.clone();
        transform(&mut mm, &place);
        st.parts.push(Part { name: format!("Dugout_{side}"), mesh: mm, mats: mats.clone(), smooth_deg: 35.0 });
        let lines: Vec<Polyline> = frame_lines.iter().map(|(pts, r)| (pts.iter().map(|q| *q + place.pos).collect(), *r)).collect();
        if let Some(t) = tubes_part(&format!("DugoutFrame_{side}"), &lines, 8, mats[2].clone()) {
            st.parts.push(t);
        }
        for (ry, rz) in [(-1.15, 0.12), (-depth + 0.05, 0.42)] {
            for j in 0..per_row {
                let vi = variant_index(&spec, j, per_row).min(vars.len() - 1);
                let local = DVec3::new(-lx / 2.0 + 0.5 + (j as f64 + 0.5) * sw, ry, rz);
                seat_groups.entry(format!("seat.{}.{}", d.seat, vars[vi])).or_default().push(Placement::at(place.pos + local, 0.0));
            }
        }
        st.sockets.push(Socket { kind: "dugout", id: side.to_string(), loc: place.pos, n: DVec2::new(0.0, 1.0), props: vec![("seats", d.seats as f64), ("length", round_to(lx, 2))] });
    }
    let seat_color = patterns::finish(&[patterns::hex_rgb(text(&spec, "color", "#3b3f46"))], 0.0, 0)[0];
    for (key, mut items) in seat_groups {
        for it in items.iter_mut() {
            it.color = Some(seat_color);
        }
        match st.props.iter_mut().find(|p| p.key == key) {
            Some(ps) => ps.items.extend(items),
            None => st.props.push(PropSet { key, items, tints: vec![], net: None }),
        }
    }
    // technical areas: dashed lines 1 m beyond the seats each side, to 1 m from the touchline
    let y_front = -(w_p / 2.0 + 1.0);
    let y_back = -(w_p / 2.0 + DUGOUT_FRONT);
    let mut paint = MeshAcc::new();
    for xc in xs {
        let (x0, x1) = (xc - lx / 2.0 - 1.0, xc + lx / 2.0 + 1.0);
        for (a, b) in [((x0, y_back), (x0, y_front)), ((x0, y_front), (x1, y_front)), ((x1, y_front), (x1, y_back))] {
            let (a, b) = (DVec2::new(a.0, a.1), DVec2::new(b.0, b.1));
            let n = ((b - a).length() / 0.6) as usize;
            for k in (0..n).step_by(2) {
                let p0 = a + (b - a) * (k as f64 / n as f64);
                let p1 = a + (b - a) * ((k + 1) as f64 / n as f64);
                let (v, q, uv) = pitch::strip(&[p0, p1], 0.1, false);
                let base = paint.verts.len() as u32;
                paint.verts.extend(v.iter().map(|q| DVec3::new(q.x, q.y, st.pitch.crown_z(q.x, q.y) + 0.004)));
                for (i, f) in q.iter().enumerate() {
                    paint.add_face(f.iter().map(|x| x + base).collect(), 0, Some(uv[4 * i..4 * i + 4].to_vec()));
                }
            }
        }
    }
    st.parts.push(Part { name: "TechnicalAreas".into(), mesh: paint, mats: vec![Mat::tint("TINT_Pitch_Paint", &st.pitch.line_color)], smooth_deg: 0.0 });
    Ok(())
}

fn build_tunnel(st: &mut Stadium, cfg: &Value, stands: Option<&StandsOut>) {
    let style = text(cfg, "style", "telescopic").to_string();
    let color = hexv(cfg, "color", "#c8102e");
    let (w, h) = (num(cfg, "width", 3.6), num(cfg, "height", 2.7));
    let y_end = -(st.pitch.width / 2.0 + DUGOUT_FRONT);
    let has_stand = stands.is_some_and(|s| s.runs.iter().any(|r| r.0.name == "south"));
    let y_wall = if has_stand { -stands.unwrap().b } else { y_end - 3.0 };
    let mut m = MeshAcc::new();
    add_box(&mut m, DVec3::new(-w / 2.0, y_wall - 0.02, 0.0), DVec3::new(w / 2.0, y_wall + 0.01, h), 0);
    add_box(&mut m, DVec3::new(-w / 2.0 - 0.25, y_wall, 0.0), DVec3::new(-w / 2.0, y_wall + 0.12, h + 0.25), 1);
    add_box(&mut m, DVec3::new(w / 2.0, y_wall, 0.0), DVec3::new(w / 2.0 + 0.25, y_wall + 0.12, h + 0.25), 1);
    add_box(&mut m, DVec3::new(-w / 2.0 - 0.25, y_wall, h), DVec3::new(w / 2.0 + 0.25, y_wall + 0.12, h + 0.25), 1);
    if !has_stand {
        add_box(&mut m, DVec3::new(-w / 2.0 - 1.2, y_wall - 4.0, 0.0), DVec3::new(w / 2.0 + 1.2, y_wall, h + 0.8), 2);
    }
    let mats = vec![Mat::named("Tunnel_Mouth"), Mat::named("Tunnel_Trim"), Mat::named("Tunnel_Building"), Mat::tint("TINT_tunnel", &color), Mat::named("Tunnel_Mat")];
    if style == "telescopic" {
        let n_hoops = 3usize.max(((y_end - y_wall) / 1.1) as usize + 1);
        let ys = linspace(y_wall + 0.15, y_end, n_hoops);
        let hw = 0.5f64.max(h - w / 2.0);
        let mut prof = vec![DVec2::new(-w / 2.0, 0.0)];
        prof.extend(linspace(PI, 0.0, 17).iter().map(|a| DVec2::new((w / 2.0) * a.cos(), hw + (w / 2.0) * a.sin())));
        prof.push(DVec2::new(w / 2.0, 0.0));
        let hoops: Vec<Polyline> = ys.iter().map(|&y| (prof.iter().map(|q| DVec3::new(q.x, y, q.y)).collect(), 0.035)).collect();
        if let Some(t) = tubes_part("TunnelHoops", &hoops, 8, Mat::named("Tunnel_Hoop")) {
            st.parts.push(t);
        }
        let k = prof.len() as u32;
        let n = (n_hoops - 1) * 4 + 1;
        let base = m.verts.len() as u32;
        for (i, y) in linspace(ys[0], ys[ys.len() - 1], n).into_iter().enumerate() {
            let sag = 0.04 * (PI * ((i % 4) as f64 / 4.0)).sin();
            for q in &prof {
                if q.y <= 1e-6 {
                    m.verts.push(DVec3::new(q.x, y, q.y));
                    continue;
                }
                let r = DVec2::new(q.x, q.y - hw);
                let qq = DVec2::new(0.0, hw) + r * (1.0 - sag / r.length().max(1e-3));
                m.verts.push(DVec3::new(qq.x, y, qq.y));
            }
        }
        for i in 0..n as u32 - 1 {
            for j in 0..k - 1 {
                let f = vec![base + i * k + j, base + (i + 1) * k + j, base + (i + 1) * k + j + 1, base + i * k + j + 1];
                let uv = f.iter().map(|&v| { let p = m.verts[v as usize]; DVec2::new(p.y, p.x + p.z) }).collect();
                m.add_face(f, 3, Some(uv));
            }
        }
        add_box(&mut m, DVec3::new(-w / 2.0, ys[0], 0.0), DVec3::new(w / 2.0, ys[ys.len() - 1], 0.03), 4);
    }
    st.parts.push(Part { name: "Tunnel".into(), mesh: m, mats, smooth_deg: 40.0 });
    st.sockets.push(Socket { kind: "tunnel", id: "players".into(), loc: DVec3::new(0.0, y_end, 0.0), n: DVec2::new(0.0, 1.0), props: vec![("width", w), ("height", h)] });
}

// -- floodlights -------------------------------------------------------------------------------

fn build_floodlights(st: &mut Stadium, cfg: &Value, cat: &Catalog, stands: Option<&StandsOut>) -> Result<(), String> {
    let spec = resolve_preset(&cat.floodlights, cfg)?.unwrap_or_default();
    let on = spec.get("on").and_then(|v| v.as_bool()).unwrap_or(st.night);
    let info = stands.map(|s| StandInfo {
        a: s.a,
        b: s.b,
        reach: s.facade_lines.iter().map(|f| f.1).fold(10.0, f64::max),
        roof_top: s.roofs.iter().map(|r| r.1.tail_top).fold(0.0, f64::max),
        roof_edges: s.roofs.iter().map(|r| r.1.edge_line.clone()).collect(),
    });
    let p = &st.pitch;
    let d = floodlights::design(&spec, p.length, p.width, p.runoff_side, info.as_ref())?;
    let fixture_key = format!("floodlight_fixture.{}", d.fixture);
    let items = d.fixtures.iter().map(|(pos, t)| Placement::aim(*pos, *t - *pos)).collect();
    st.props.push(PropSet { key: fixture_key, items, tints: vec![], net: None });
    if let Some(t) = tubes_part("FloodlightSteel", &d.steel, 6, Mat::named("Galvanised")) {
        st.parts.push(t);
    }
    let masts: Vec<Polyline> = d.tube_masts.iter().map(|(a, b, r)| (vec![*a, *b], *r)).collect();
    if let Some(t) = tubes_part("FloodlightMasts", &masts, 16, Mat::named("Galvanised")) {
        st.parts.push(t);
    }
    if on {
        let color = floodlights::kelvin_rgb(d.kelvin);
        for (pos, t) in &d.fixtures {
            let dir = (*t - *pos).normalize();
            st.spots.push(Spot { pos: *pos + dir * 0.3, dir, power: d.power, angle_deg: d.beam, blend: 0.9, color });
        }
        if let Some(i) = &info {
            if !i.roof_edges.is_empty() {
                let (lights, power) = floodlights::bowl_lights(&i.roof_edges);
                for (pos, dir) in lights {
                    st.spots.push(Spot { pos, dir, power, angle_deg: 120.0, blend: 1.0, color });
                }
            }
        }
    }
    st.report.lights = Some(d);
    Ok(())
}

/// Seat models a stadium needs (for preloading), e.g. ["seat.tipup_classic.std", ...].
pub fn prop_keys(st: &Stadium) -> Vec<String> {
    st.props.iter().map(|p| p.key.clone()).collect()
}

/// sRGB -> linear, exposed for renderers.
pub fn srgb_to_linear(c: DVec3) -> DVec3 {
    lin(c)
}
