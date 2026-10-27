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
/// a screen hung under a roof edge keeps at least this much air below it (m)
pub const SCREEN_HANG_CLEARANCE: f64 = 6.0;
/// lowest edge of a centre-hung screen over the pitch (m)
pub const SCREEN_CENTRE_CLEARANCE: f64 = 18.0;
/// lowest roof edge a roof floodlight ribbon is mounted on (m)
pub const ROOF_LIGHTS_MIN_HEIGHT: f64 = 10.0;
/// The main TV camera sits at least this high; a lower tier front gets a gantry under the roof
/// edge or a platform at the back of the stand instead.
pub const CAMERA_MIN_HEIGHT: f64 = 12.0;
/// Highest front girder a goalpost roof is built with; taller stands get a cantilever.
pub const GOALPOST_MAX_HEIGHT: f64 = 25.0;
/// Grounds this big get a paved plaza round them by default.
pub const PLAZA_MIN_CAPACITY: f64 = 5000.0;

/// A texture the renderer composes at runtime.
#[derive(Clone, Debug, PartialEq)]
pub enum Tex {
    /// horizontal atlas of board creatives, in this order (the LED rotation)
    BoardAtlas(Vec<String>),
    Carpet(String),
    Scoreboard,
    /// goal net card: cord pattern regenerated at runtime
    Net(NetSpec),
    /// procedural grass: the turf and the pitch frame it is laid out in
    Grass(pitch::GrassSpec),
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
    /// the mesh's UVs are in metres (generated fences and netting) rather than the library goal
    /// card's units of one 120 mm hex repeat
    pub metric: bool,
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
    /// facade height per run
    facade_heights: Vec<(String, f64)>,
    camera_mount: Option<DVec3>,
    /// Some(radius) for a curved bowl
    corner_r: Option<f64>,
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
    let (layout, fixes) = crate::sanitize::sanitize(layout, cat);
    let layout = &layout;
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
        report: Report { grass: pitch.grass_spec.clone(), warnings: fixes, ..Default::default() },
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
        build_dugouts(&mut st, d, cat, stands.as_ref())?;
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

/// The pitch's procedural grass (centred on the origin).
pub fn grass_mat(p: &Pitch) -> Mat {
    Mat { key: "Pitch_Grass".into(), color: None, emission: None, texture: Some(Tex::Grass(p.grass_spec(DVec2::ZERO, 0.0))) }
}

/// Line paint on that grass (the paint lets the grass show through).
pub fn paint_mat(p: &Pitch) -> Mat {
    Mat { key: "TINT_Pitch_Paint".into(), color: Some(patterns::hex_rgb(&p.line_color)), emission: None, texture: Some(Tex::Grass(p.grass_spec(DVec2::ZERO, 0.0))) }
}

fn build_pitch(st: &mut Stadium) {
    let p = &st.pitch;
    st.parts.push(Part { name: "PitchSurface".into(), mesh: pitch::surface_mesh(p), mats: vec![grass_mat(p)], smooth_deg: 30.0 });
    st.parts.push(Part { name: "PitchLines".into(), mesh: pitch::markings_mesh(p), mats: vec![paint_mat(p)], smooth_deg: 30.0 });
    for (kind, id, pos, yaw) in pitch::sockets(p) {
        st.sockets.push(Socket { kind, id: id.into(), loc: pos, n: DVec2::new(yaw.to_radians().cos(), yaw.to_radians().sin()), props: vec![("yaw_deg", yaw)] });
    }
}

fn build_goals_and_flags(st: &mut Stadium, layout: &Value, cat: &Catalog) -> Result<(), String> {
    let goals = layout.get("goals").cloned().unwrap_or(json!({"style": "classic"}));
    let flags = layout.get("corner_flags").cloned().unwrap_or(json!({"preset": "pro_spring"}));
    let sets = goals_and_flags(&st.pitch, &goals, &flags, cat)?;
    st.props.extend(sets);
    Ok(())
}

/// Goal and corner-flag props on a pitch's sockets (pitch-local coordinates). `goals` / `flags`
/// are the layout sections (null = none).
pub fn goals_and_flags(pitch: &Pitch, goals: &Value, flags: &Value, cat: &Catalog) -> Result<Vec<PropSet>, String> {
    let mut props = Vec::new();
    let goals = goals.clone();
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
        // net mesh: the layout's, else the style's (youth goals carry a square 100 mm mesh)
        let preset = &cat.goals[&style];
        let net = NetSpec {
            mesh: text(&goals, "mesh", text(preset, "mesh", "hex")).to_string(),
            mesh_size_mm: (num(&goals, "mesh_size", num(preset, "mesh_size", 0.12)) * 1000.0).round() as u32,
            cord_tenths_mm: (num(&goals, "cord_mm", num(preset, "cord_mm", 4.0)) * 10.0).round() as u32,
            pattern: text(&goals, "net_pattern", "solid").to_string(),
            band_cells: num(&goals, "band_cells", 3.0).max(1.0) as u32,
            colors: nets.iter().map(|c| to8(c)).collect(),
            metric: false,
        };
        let items = pitch::sockets(pitch).into_iter().filter(|s| s.0 == "goal").map(|s| Placement::at(s.2, s.3.to_radians())).collect();
        props.push(PropSet { key: format!("goal.{style}"), items, tints, net: Some(net) });
    }
    let flags = flags.clone();
    if !flags.is_null() {
        let preset = text(&flags, "preset", "pro_spring").to_string();
        let spec = cat.corner_flags.get(&preset).ok_or_else(|| format!("unknown corner flag preset {preset:?}"))?;
        let spec = merge(spec, &flags);
        let mut tints = Vec::new();
        let poles = spec["pole_colors"].as_array().cloned().unwrap_or_default();
        for (slot, c) in ["TINT_pole_a", "TINT_pole_b"].iter().zip(poles.iter().filter_map(|c| c.as_str())) {
            tints.push((slot.to_string(), patterns::hex_rgb(c)));
        }
        let items = pitch::sockets(pitch).into_iter().filter(|s| s.0 == "corner_flag").map(|s| Placement::at(s.2, s.3.to_radians())).collect();
        props.push(PropSet { key: format!("corner_flag.{preset}"), items, tints, net: None });
    }
    Ok(props)
}

// -- stands -------------------------------------------------------------------------------------

/// Floor height of the stand at depth `s` (the highest tread in front of it).
fn floor_at(tiers: &[Tier], s: f64) -> f64 {
    let mut z: f64 = 0.0;
    for t in tiers {
        for (i, &si) in t.s.iter().enumerate() {
            if si - t.tread <= s {
                z = z.max(t.h[i]);
            }
        }
    }
    z
}

enum Gantry {
    /// hung from the roof's front edge at this height
    Hung(f64),
    /// a scaffold platform standing on the treads at this height
    Standing(f64),
    /// a camera hut on a clad base standing on a low roof's leading edge (its top at this height)
    OnRoof(f64),
}

/// Where the main camera goes when the tier front is too low: (depth, eye height, structure).
/// roof_edge: (depth of the leading edge, its underside, its top, where a gantry may hang).
fn high_camera(tiers: &[Tier], roof_edge: Option<(f64, f64, f64, f64)>) -> (f64, f64, Gantry) {
    if let Some((s_front, z_edge, edge_top, s_hang)) = roof_edge {
        if z_edge - 2.2 >= CAMERA_MIN_HEIGHT - 2.0 {
            // a gantry slung under the roof's leading edge, clear of the treads below and low
            // enough that the edge and any floodlights on it stay above the shot
            let s = s_hang;
            let z = (z_edge - 2.2).max(floor_at(tiers, s) + 2.2).min(z_edge - 0.3);
            return (s, z, Gantry::Hung(z_edge));
        }
        // a low roof: a gantry standing on its leading edge
        return (s_front + 0.6, (edge_top + 1.6).max(CAMERA_MIN_HEIGHT), Gantry::OnRoof(edge_top));
    }
    // a platform over the back rows, reaching the minimum height
    let top = tiers.last().unwrap();
    let s = top.s_back - 1.2;
    let floor = floor_at(tiers, s);
    (s, (floor + 1.6).max(CAMERA_MIN_HEIGHT), Gantry::Standing(floor))
}

/// The camera platform: a 3 x 2.4 m deck 1.6 m below the lens, the lens at its open front edge
/// (rails on the sides and back only, out of shot), and hangers to the roof edge or legs down to
/// the treads or the roof.
fn gantry_part(run: &Run, t: f64, s_cam: f64, z: f64, g: Gantry) -> Part {
    let deck = z - 1.6;
    let s = s_cam + 1.0;
    let mut lines: Vec<Polyline> = Vec::new();
    let corners = [(-1.5, -1.2), (1.5, -1.2), (1.5, 1.2), (-1.5, 1.2)];
    let at = |dt: f64, ds: f64, zz: f64| run.world(t + dt, s + ds, zz);
    let mut ring: Vec<DVec3> = corners.iter().map(|&(a, b)| at(a, b, deck)).collect();
    ring.push(ring[0]);
    lines.push((ring, 0.05));
    lines.push(([1, 2, 3, 0].iter().map(|&k| at(corners[k].0, corners[k].1, deck + 1.1)).collect(), 0.025));
    for &(a, b) in &corners {
        lines.push((vec![at(a, b, deck), at(a, b, deck + 1.1)], 0.025));
        match g {
            Gantry::Hung(top) => lines.push((vec![at(a, b, deck), at(a, b, top)], 0.04)),
            Gantry::Standing(floor) | Gantry::OnRoof(floor) if deck - floor > 0.3 => lines.push((vec![at(a, b, floor), at(a, b, deck)], 0.05)),
            _ => {}
        }
    }
    if let Gantry::Standing(floor) | Gantry::OnRoof(floor) = g {
        // cross bracing every 2 m on the legs
        let mut zz = floor;
        while zz + 2.0 < deck {
            for k in 0..4 {
                let (a0, b0) = corners[k];
                let (a1, b1) = corners[(k + 1) % 4];
                lines.push((vec![at(a0, b0, zz), at(a1, b1, zz + 2.0)], 0.03));
            }
            zz += 2.0;
        }
    }
    let mut mesh = tube_lines(&lines, 8);
    let sec = Section::uniform(vec![DVec2::new(s - 1.2, deck - 0.12), DVec2::new(s + 1.2, deck - 0.12), DVec2::new(s + 1.2, deck), DVec2::new(s - 1.2, deck)], 0);
    mesh.add_prism(run, t - 1.5, t + 1.5, &sec, 0);
    if let Gantry::OnRoof(floor) = g {
        // a hut with an open front for the camera on a braced steel frame, and a ladder
        let m = run.metric(s);
        let rect = |s0: f64, s1: f64, z0: f64, z1: f64| Section::uniform(vec![DVec2::new(s0, z0), DVec2::new(s1, z0), DVec2::new(s1, z1), DVec2::new(s0, z1)], 1);
        mesh.add_prism(run, t - 1.5, t + 1.5, &rect(s + 1.1, s + 1.2, deck, deck + 2.3), 1);
        for (ta, tb) in [(t - 1.5, t - 1.5 + 0.1 / m), (t + 1.5 - 0.1 / m, t + 1.5)] {
            mesh.add_prism(run, ta, tb, &rect(s - 1.2, s + 1.2, deck, deck + 2.3), 1);
        }
        mesh.add_prism(run, t - 1.6, t + 1.6, &rect(s - 1.45, s + 1.3, deck + 2.3, deck + 2.45), 1);
        let mut ladder: Vec<Polyline> = Vec::new();
        for dt in [-0.22, 0.22] {
            ladder.push((vec![run.world(t + dt / m, s + 1.45, floor), run.world(t + dt / m, s + 1.45, deck + 1.0)], 0.025));
        }
        let mut z = floor + 0.3;
        while z < deck {
            ladder.push((vec![run.world(t - 0.22 / m, s + 1.45, z), run.world(t + 0.22 / m, s + 1.45, z)], 0.012));
            z += 0.3;
        }
        mesh.append(&tube_lines(&ladder, 6), 0);
    }
    Part { name: "CameraGantry".into(), mesh, mats: vec![Mat::named("Galvanised"), Mat::tint("Screen_Body", "#6b6f75")], smooth_deg: 60.0 }
}

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
        acc.verts.extend([*p0 + lift, *p1 + lift, *p1 + n * 0.07 + lift, *p0 + n * 0.07 + lift]);
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
    let mut r_corner = num(cfg, "corner_radius", 14.0);
    let a = p.length / 2.0 + p.runoff_end + gap;
    let b = p.width / 2.0 + p.runoff_side + gap;
    let mut early_warnings = Vec::new();
    if curved {
        let r_max = crate::runs::max_corner_radius(a, b, p.length / 2.0, p.width / 2.0, 2.0);
        if r_corner > r_max {
            early_warnings.push(format!("corner radius {} m would put the corner stands on the pitch; using {:.1} m", r_corner, r_max));
            r_corner = r_max;
        }
    }
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
    let mut out = StandsOut { a, b, runs: Vec::new(), roofs: Vec::new(), facade_lines: Vec::new(), facade_heights: Vec::new(), camera_mount: None, corner_r: curved.then_some(r_corner) };
    let mut warnings: Vec<String> = early_warnings;
    let mut tall_goalposts: Vec<String> = Vec::new();
    let mut plaza_items: Vec<(&'static str, DVec3, f64)> = Vec::new();
    for (ri, run) in run_defs.iter().enumerate() {
        let sc = sides.iter().find(|s| s.name == run.name).unwrap_or(&sides[0]);
        let df = if curved { focal("south").min(focal("east")) } else { focal(&run.name) };
        let style = text(&sc.cfg, "seat", "tipup_classic").to_string();
        // rows as deep as the seat needs for a 305 mm seatway (padded seats need wider treads)
        let mut prof = cat.profiles[&sc.profile].clone();
        for ts in prof.tiers.iter_mut().filter(|ts| ts.kind != "standing") {
            let s = if ts.hospitality.unwrap_or(false) { &hosp_seat } else { &style };
            let min_tread = num(seat_spec(cat, s)?, "min_tread", 0.7);
            if ts.tread.unwrap_or(0.80) < min_tread - 1e-9 {
                ts.tread = Some(min_tread);
            }
        }
        let mut tiers = solve_profile(&prof, df);
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
        // a bowl's ring roof sights the longer span all round, so its edge is one level
        let far = if curved { p.width.max(p.length) } else if run.name == "north" || run.name == "south" { p.width } else { p.length };
        let roof_spec = sc.cfg.get("roof").or_else(|| cfg.get("roof")).cloned().unwrap_or(Value::Null);
        let mut roof_spec = resolve_preset(&cat.roofs, &roof_spec)?;
        // goalpost roofs suit single low stands; over a tall stand the posts would be 30-60 m
        // columns in the front row, so such stands get a cantilever instead
        if let Some(rs) = roof_spec.as_mut() {
            if roofs::kind_of(rs) == roofs::RoofKind::Goalpost && roofs::solve(&tiers, rs, df, far).z_front > GOALPOST_MAX_HEIGHT {
                tall_goalposts.push(run.name.clone());
                rs["type"] = json!("cantilever");
            }
        }
        let mut roof_info = None;
        let mut roof_edge = None;
        let mut roof_posts = None;
        if let Some(rs) = &roof_spec {
            let roof = roofs::solve(&tiers, rs, df, far);
            let ro = roofs::build(run, &roof, &o.bay_list, !run.curved());
            let membrane = text(rs, "deck", "metal") == "membrane";
            let rmats = vec![
                Mat::tint(if membrane { "TINT_roof_membrane" } else { "TINT_roof" }, &hexv(rs, "color", "#e3e4e2")),
                Mat::named("Roof_Polycarbonate"),
                Mat::tint("TINT_roof_fascia", &hexv(rs, "fascia_color", "#1b1d20")),
                Mat::tint("Roof_Underside", &hexv(rs, "underside_color", "#d8d8d4")),
                Mat::tint("Roof_Steel", &hexv(rs, "steel_color", "#eceeee")),
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
            let z_edge = ro.edge_line.iter().map(|q| q.z).fold(f64::MAX, f64::min);
            if z_edge.is_finite() {
                // (a goalpost roof's posts stand in the front row: a gantry goes in front of them)
                let s_hang = if roof.kind == roofs::RoofKind::Goalpost { roof.s_post - 0.8 } else { roof.s_front + 1.0 };
                roof_edge = Some((roof.s_front, z_edge, ro.edge_top, s_hang));
                roof_posts = (roof.kind == roofs::RoofKind::Goalpost).then_some(roof.s_post);
            }
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
            out.facade_heights.push((run.name.clone(), fo.height));
            plaza_items.extend(fo.props);
        } else if !curved && tiers.len() > 1 {
            // no facade gables: close the exposed ends of a multi-tier stand with a concrete wall,
            // or the concourse void under the upper tier shows as a hole
            let hull = Section::uniform(stands::convex_hull(stands::section_hull(&tiers)), 0);
            let d = 0.3 / run.metric(0.0);
            let mut ends = MeshAcc::new();
            ends.add_prism(run, run.t0 - d, run.t0, &hull, 0);
            ends.add_prism(run, run.t1, run.t1 + d, &hull, 0);
            st.parts.push(Part { name: format!("StandEnds_{}", run.name), mesh: ends, mats: vec![mats[0].clone()], smooth_deg: 0.0 });
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
            let mut m = DVec3::new(q.x, q.y, t.h[0] + t.parapet + 1.6);
            // a goalpost roof's front-row posts would stand in the shot of a camera behind them
            let behind_posts = roof_posts.is_some_and(|sp| t.s0 - 1.2 > sp);
            if m.z < CAMERA_MIN_HEIGHT || behind_posts {
                let (s, z, gantry) = high_camera(&tiers, roof_edge);
                if z > m.z || behind_posts {
                    let q = pnt + n * s;
                    m = DVec3::new(q.x, q.y, z);
                    st.parts.push(gantry_part(run, 0.5 * (run.t0 + run.t1), s, z, gantry));
                }
            }
            out.camera_mount = Some(m);
            st.sockets.push(Socket { kind: "camera", id: "main".into(), loc: m, n: DVec2::new(0.0, -1.0), props: vec![] });
        }
        out.runs.push((run.clone(), tiers));
    }

    if !tall_goalposts.is_empty() {
        warnings.push(format!("a goalpost roof would stand over {GOALPOST_MAX_HEIGHT:.0} m on {}; cantilever used there", tall_goalposts.join(", ")));
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
        patterns::apply_mosaic(&mut colors, &all, &mz, ri, run_lengths[ri], &tier_rows[ri], pic.as_ref(), &mut warnings);
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
    let front_r = if curved { r_corner } else { 0.0 };
    for k in 0..n {
        let ang = 2.0 * PI * k as f64 / n as f64;
        let d = DVec2::new(ang.cos(), ang.sin());
        // (a tight bowl's corner stands come in over the run-off corners: stop at their front)
        let q = rect_hit(d, ix, iy, 0.0);
        let f = rect_hit(d, a + 0.3, b + 0.3, front_r);
        let q = if q.length() > f.length() { f } else { q };
        apron.verts.push(DVec3::new(q.x, q.y, -0.005));
    }
    // ... and the run-off turf stops there too instead of poking out behind them
    if curved {
        if let Some(surf) = st.parts.iter_mut().find(|pt| pt.name == "PitchSurface") {
            for v in surf.mesh.verts.iter_mut() {
                let r = v.truncate().length();
                if r > 1e-9 {
                    let f = rect_hit(v.truncate() / r, a + 0.3, b + 0.3, front_r);
                    if r > f.length() {
                        *v = DVec3::new(f.x, f.y, p.crown_z(f.x, f.y));
                    }
                }
            }
        }
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
        _ => grass_mat(&p),
    };
    st.parts.push(Part { name: "Apron".into(), mesh: apron, mats: vec![apron_mat], smooth_deg: 0.0 });
    // a paved plaza (and its street furniture) goes round a stadium, not a club ground
    let plaza_default = !out.facade_lines.is_empty() && all.len() as f64 + standing >= PLAZA_MIN_CAPACITY;
    if cfg.get("plaza").and_then(|v| v.as_bool()).unwrap_or(plaza_default) {
        let mut sets: BTreeMap<&str, Vec<Placement>> = BTreeMap::new();
        for (key, pos, yaw) in &plaza_items {
            sets.entry(key).or_default().push(Placement::at(*pos, *yaw));
        }
        for (key, items) in sets {
            st.props.push(PropSet { key: key.to_string(), items, tints: vec![], net: None });
        }
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
        // charcoal granite bands following the plaza's outline, 6 m and 14 m in from its edge
        let mut bands = MeshAcc::new();
        for d in [6.0, 14.0] {
            let base = bands.verts.len() as u32;
            for w in [0.0, 0.9] {
                for k in 0..n {
                    let ang = 2.0 * PI * k as f64 / n as f64;
                    let q = rect_hit(DVec2::new(ang.cos(), ang.sin()), hx - d - w, hy - d - w, r - d - w);
                    bands.verts.push(DVec3::new(q.x, q.y, -0.002));
                }
            }
            for k in 0..n as u32 {
                let k1 = (k + 1) % n as u32;
                let f = vec![base + k, base + k1, base + n as u32 + k1, base + n as u32 + k];
                let uv = f.iter().map(|&i| bands.verts[i as usize].truncate()).collect();
                bands.add_face(f, 0, Some(uv));
            }
        }
        st.parts.push(Part { name: "PlazaBands".into(), mesh: bands, mats: vec![Mat::named("Plaza_Band")], smooth_deg: 0.0 });
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

/// Facade signs stand this far out from the facade line (clear of fins and cushions).
pub const FACADE_SIGN_OUT: f64 = 1.9;

/// Bottom and top of a facade sign: a tenth of the facade tall (1-4.5 m), its top just under the
/// coping.
pub fn facade_sign_band(height: f64) -> (f64, f64) {
    let h = (0.1 * height).clamp(1.0, 4.5);
    let top = height - 1.2 - 0.06 * height;
    (top - h, top)
}

/// Run of the LED board face (s per metre of height): the face leans back ~11 deg.
pub const LED_FACE_RAKE: f64 = 0.17 / 0.87;

/// Kickstands behind an LED board run: a raking strut and a ground runner every 1.92 m.
fn board_struts(run: &Run) -> Vec<Polyline> {
    let mut out = Vec::new();
    let mut t = run.t0 + 0.96;
    while t < run.t1 - 0.3 {
        out.push((vec![run.world(t, 0.25, 0.75), run.world(t, 0.78, 0.03)], 0.02));
        out.push((vec![run.world(t, 0.08, 0.03), run.world(t, 0.84, 0.03)], 0.022));
        t += 1.92;
    }
    out
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
    // the end boards stay behind the goal nets and in front of the end stands
    let reach = stands.map(|s| s.a - p.length / 2.0 - 0.8).unwrap_or(p.runoff_end + 3.0);
    let end_off = match per.get("end_offset") {
        Some(Value::Number(n)) => {
            let want = n.as_f64().unwrap_or(placement::BOARD_END_OFFSET);
            let got = want.min(reach).max(placement::BOARD_END_MIN.min(reach));
            if (got - want).abs() > 1e-9 {
                st.report.warnings.push(format!("end boards {want} m behind the goal line don't fit; using {got:.2} m"));
            }
            got
        }
        _ if carpets_on => placement::end_board_offset(cam, &p, carpet_width, reach, placement::CARPET_HEIGHT),
        _ => placement::BOARD_END_OFFSET,
    };
    st.report.end_boards_m = end_off;
    if kind != "none" {
        let runs = placement::board_runs(&p, gap, placement::BOARD_SIDE_OFFSET, end_off);
        if kind == "led" {
            let mut housing = MeshAcc::new();
            let mut faces = MeshAcc::new();
            let mut struts: Vec<Polyline> = Vec::new();
            for run in &runs {
                // a slim cabinet leaning back ~11 deg; the LED face sits on its front, framed by the
                // cabinet's edges above and below, and a kickstand behind every 1.92 m module pair
                let poly = vec![DVec2::new(0.0, 0.04), DVec2::new(0.13, 0.04), DVec2::new(0.30, 0.985), DVec2::new(LED_FACE_RAKE * 0.885, 0.985), DVec2::new(0.0, 0.1)];
                housing.add_prism(run, run.t0, run.t1, &Section::uniform(poly, 0), 0);
                let offset = (crc32(&run.name) as usize % n_atlas) as f64 / n_atlas as f64;
                let (v, f, uv) = ad_strip(run, run.t0, run.t1, LED_FACE_RAKE * 0.02, 0.12, LED_FACE_RAKE * 0.85, 0.95, offset, 1.0 / (CREATIVE_M * n_atlas as f64), 0.006);
                add_vertex_uv_quads(&mut faces, &v, &f, &uv, 0);
                struts.extend(board_struts(run));
                let length = run.t1 - run.t0;
                for k in 0..(length / CREATIVE_M).floor() as usize {
                    let (q, n) = run.frame(run.t0 + (k as f64 + 0.5) * CREATIVE_M);
                    slot(st, "perimeter_led", format!("{}_{k}", run.name), DVec3::new(q.x, q.y, 0.5), DVec3::new(-n.x, -n.y, 0.0), CREATIVE_M * 0.85, "rotation".into());
                }
            }
            st.parts.push(Part { name: "BoardHousing".into(), mesh: housing, mats: vec![Mat::named("Board_Housing")], smooth_deg: 0.0 });
            st.parts.push(Part { name: "BoardLED".into(), mesh: faces, mats: vec![led_mat("Ads_LED", strength)], smooth_deg: 0.0 });
            if let Some(t) = tubes_part("BoardStruts", &struts, 6, Mat::named("Board_Housing")) {
                st.parts.push(t);
            }
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
            if !placement::carpet_fits(cam, base, width, h, limit) {
                let w = "no room for 3D carpets between the goals and the end boards; carpets left out".to_string();
                if !st.report.warnings.contains(&w) {
                    st.report.warnings.push(w);
                }
                continue;
            }
            let pc = p.clone();
            let (v, f, uv) = placement::carpet(cam, base, width, h, 32, 10, &move |x, y| pc.crown_z(x, y));
            let mut m = MeshAcc::new();
            add_vertex_uv_quads(&mut m, &v, &f, &uv, 0);
            st.parts.push(Part { name: format!("Carpet_{k}"), mesh: m, mats: vec![Mat { key: "Carpet".into(), color: None, emission: None, texture: Some(Tex::Carpet(bk.clone())) }], smooth_deg: 0.0 });
            let to_cam = cam - base;
            slot(st, "carpet_3d", format!("carpet_{k}"), DVec3::new(base.x, base.y, 0.0), to_cam / to_cam.length(), width * h, bk);
        }
    }
    // a sponsor sign high on the middle of each side's facade, on a dark frame
    if let Some(stands) = stands {
        let mut signs = MeshAcc::new();
        let mut frames = MeshAcc::new();
        for ((name, sw), (_, fh)) in stands.facade_lines.iter().zip(&stands.facade_heights) {
            let Some((run, _)) = stands.runs.iter().find(|(r, _)| &r.name == name && !r.curved()) else { continue };
            let (z0, z1) = facade_sign_band(*fh);
            let s = sw + FACADE_SIGN_OUT;
            let half = (z1 - z0) * CREATIVE_M / 2.0 / run.metric(s);
            let tm = 0.5 * (run.t0 + run.t1);
            let bk = &keys[(crc32(name) as usize + 1) % keys.len()];
            let bi = keys.iter().position(|x| x == bk).unwrap();
            let (v, f, mut uv) = ad_strip(run, tm - half, tm + half, s, z0, s, z1, 0.0, 0.0, 0.0);
            for (i, u) in uv.iter_mut().enumerate() {
                // read from outside, the creative runs with t
                u.x = if i / 2 == 0 { bi as f64 / n_atlas as f64 } else { (bi + 1) as f64 / n_atlas as f64 };
            }
            add_vertex_uv_quads(&mut signs, &v, &f, &uv, 0);
            let pad = 0.3 / run.metric(s);
            frames.add_prism(run, tm - half - pad, tm + half + pad, &Section::uniform(vec![DVec2::new(s - 0.3, z0 - 0.3), DVec2::new(s - 0.02, z0 - 0.3), DVec2::new(s - 0.02, z1 + 0.3), DVec2::new(s - 0.3, z1 + 0.3)], 0), 0);
            let (q, n) = run.frame(tm);
            slot(st, "facade_sign", format!("{name}_facade"), DVec3::new(q.x + n.x * s, q.y + n.y * s, 0.5 * (z0 + z1)), DVec3::new(n.x, n.y, 0.0), (z1 - z0) * (z1 - z0) * CREATIVE_M, bk.clone());
        }
        if !signs.faces.is_empty() {
            st.parts.push(Part { name: "FacadeSigns".into(), mesh: signs, mats: vec![Mat { key: "Ads_Print".into(), color: None, emission: None, texture: Some(Tex::BoardAtlas(keys.clone())) }], smooth_deg: 0.0 });
            st.parts.push(Part { name: "FacadeSignFrames".into(), mesh: frames, mats: vec![Mat::named("Board_Housing")], smooth_deg: 0.0 });
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
    // service rails across the back every metre: the back reads as a cabinet, not a hole
    let mut z = -h / 2.0 + 0.5;
    while z < h / 2.0 {
        add_box(&mut m, DVec3::new(-w / 2.0, -depth - 0.08, z - 0.06), DVec3::new(w / 2.0, -depth, z + 0.06), 2);
        z += 1.0;
    }
    // vertical stiffeners behind, a bezel standing proud of the face, and a rain hood on top
    for x in [-(w / 2.0 - 0.3), -w / 4.0, w / 4.0, w / 2.0 - 0.3] {
        add_box(&mut m, DVec3::new(x - 0.07, -depth - 0.16, -h / 2.0), DVec3::new(x + 0.07, -depth - 0.08, h / 2.0), 2);
    }
    let (ew, eh) = (w / 2.0 + 0.25, h / 2.0 + 0.25);
    for (lo, hi) in [
        (DVec3::new(-ew, 0.0, h / 2.0), DVec3::new(ew, 0.12, eh)),
        (DVec3::new(-ew, 0.0, -eh), DVec3::new(ew, 0.12, -h / 2.0)),
        (DVec3::new(-ew, 0.0, -h / 2.0), DVec3::new(-w / 2.0, 0.12, h / 2.0)),
        (DVec3::new(w / 2.0, 0.0, -h / 2.0), DVec3::new(ew, 0.12, h / 2.0)),
        (DVec3::new(-ew - 0.1, -depth - 0.1, eh), DVec3::new(ew + 0.1, 0.4, eh + 0.12)),
    ] {
        add_box(&mut m, lo, hi, 3);
    }
    let b = m.verts.len() as u32;
    m.verts.extend([DVec3::new(-w / 2.0, 0.01, -h / 2.0), DVec3::new(w / 2.0, 0.01, -h / 2.0), DVec3::new(w / 2.0, 0.01, h / 2.0), DVec3::new(-w / 2.0, 0.01, h / 2.0)]);
    // wound to face +Y; seen from the front local -X is the viewer's right, so u runs +X -> -X
    m.add_face(vec![b, b + 3, b + 2, b + 1], 1, Some(vec![DVec2::new(1.0, 0.0), DVec2::new(1.0, 1.0), DVec2::new(0.0, 1.0), DVec2::new(0.0, 0.0)]));
    m
}

/// The point halfway along a polyline (by length).
fn midpoint(line: &[DVec3]) -> DVec3 {
    let total: f64 = line.windows(2).map(|w| (w[1] - w[0]).length()).sum();
    let mut left = total / 2.0;
    for w in line.windows(2) {
        let d = (w[1] - w[0]).length();
        if d >= left && d > 0.0 {
            return w[0] + (w[1] - w[0]) * (left / d);
        }
        left -= d;
    }
    line[line.len() / 2]
}

fn build_screens(st: &mut Stadium, cfg: &Value, stands: Option<&StandsOut>) -> Result<(), String> {
    let p = st.pitch.clone();
    let (w, h) = (num(cfg, "width", 18.0), num(cfg, "height", 8.0));
    let (l, wp) = (p.length, p.width);
    let led = Mat { key: "Screen_LED".into(), color: None, emission: Some((DVec3::ONE, if st.night { 1.6 } else { 3.5 })), texture: Some(Tex::Scoreboard) };
    // a mid-grey cabinet (its back is what the far end of the ground sees), dark bezels
    let mats = vec![Mat::tint("Screen_Body", "#6b6f75"), led, Mat::named("Galvanised"), Mat::named("Board_Housing")];
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
    let hang = |st: &mut Stadium, tubes: &mut Vec<Polyline>, put: &mut dyn FnMut(&mut Stadium, String, DVec3, f64, f64), name: String, line: &[DVec3]| {
        let mid = midpoint(line);
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
        // a leg every ~8 m with struts to the screen, tied by rails and cross braces behind it
        let n_legs = 2usize.max((w / 8.0).ceil() as usize + 1);
        let legs: Vec<DVec2> = linspace(-0.42 * w, 0.42 * w, n_legs).iter().map(|&u| xy + side * u + back * 1.2).collect();
        for &q in &legs {
            let r = q - back * 1.1;
            tubes.push((vec![DVec3::new(q.x, q.y, ground), DVec3::new(q.x, q.y, bottom + h)], 0.22));
            tubes.push((vec![DVec3::new(q.x, q.y, bottom + 0.2), DVec3::new(r.x, r.y, bottom + 0.2)], 0.08));
            tubes.push((vec![DVec3::new(q.x, q.y, bottom + h), DVec3::new(r.x, r.y, bottom + h)], 0.08));
        }
        for pair in legs.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            for z in [bottom + 0.2, bottom + h] {
                tubes.push((vec![DVec3::new(a.x, a.y, z), DVec3::new(b.x, b.y, z)], 0.07));
            }
            tubes.push((vec![DVec3::new(a.x, a.y, bottom + 0.2), DVec3::new(b.x, b.y, bottom + h)], 0.06));
            if bottom - ground > 2.0 {
                tubes.push((vec![DVec3::new(a.x, a.y, ground + 0.3), DVec3::new(b.x, b.y, bottom + 0.2)], 0.06));
            }
        }
    };
    // A screen hangs under a roof edge only with real clearance below it; otherwise it stands on
    // a frame on top of its stand (on top of the roof if there is one), and with no stand on a
    // freestanding frame. Centre-hung screens need a roof high enough to clear the play.
    let can_hang = |line: &[DVec3]| midpoint(line).z - 0.6 - h >= SCREEN_HANG_CLEARANCE;
    let run_of = |name: &str| stands.and_then(|s| s.runs.iter().find(|r| r.0.name == name));
    let roof_of = |name: &str| stands.and_then(|s| s.roofs.iter().find(|r| r.0 == name)).map(|r| &r.1);
    let top_mount = |st: &mut Stadium, tubes: &mut Vec<Polyline>, put: &mut dyn FnMut(&mut Stadium, String, DVec3, f64, f64), name: String, run_name: &str| -> bool {
        let Some((run, tiers)) = run_of(run_name) else { return false };
        let (q, n) = run.frame(0.5 * (run.t0 + run.t1));
        let mut base = tiers.iter().map(|t| t.h_last()).fold(f64::MIN, f64::max) + 1.1;
        let mut foot = base;
        if let Some(r) = roof_of(run_name) {
            base = base.max(r.top + 0.3);
            foot = foot.max(r.tail_top);
        }
        let xy = q + n * (stand_depth(tiers) - 1.0);
        on_frame(st, tubes, put, name, xy, base + 1.0, foot);
        true
    };
    let placement_kind = text(cfg, "placement", "ends").to_string();
    let mut kind = placement_kind.as_str();
    if kind == "center" {
        let roofs = stands.map(|s| &s.roofs[..]).unwrap_or(&[]);
        let z_edge = roofs.iter().map(|r| r.1.edge_line.iter().map(|q| q.z).fold(f64::MAX, f64::min)).fold(f64::MAX, f64::min);
        if roofs.is_empty() {
            st.report.warnings.push("a centre-hung screen needs roofs to hang from; screens placed at the ends instead".into());
            kind = "ends";
        } else if (z_edge - 6.0 - h).min(25.0) < SCREEN_CENTRE_CLEARANCE {
            st.report.warnings.push(format!("the roofs are too low for a {h:.0} m centre-hung screen; screens placed at the ends instead"));
            kind = "ends";
        }
    }
    match kind {
        "ends" | "stand_top" => {
            for (side, x) in [("east", 1.0), ("west", -1.0)] {
                let name = format!("Screen_{side}");
                let line = roof_of(side).map(|r| r.edge_line.clone());
                if kind == "ends" && line.as_ref().is_some_and(|l| can_hang(l)) {
                    hang(st, &mut tubes, &mut put, name, line.as_ref().unwrap());
                } else if !top_mount(st, &mut tubes, &mut put, name.clone(), side) {
                    on_frame(st, &mut tubes, &mut put, name, DVec2::new(x * (l / 2.0 + p.runoff_end + 4.0), 0.0), 4.0, 0.0);
                }
            }
        }
        "corners" => {
            for (sx, sy, arc) in [(1.0, 1.0, "NE"), (-1.0, -1.0, "SW")] {
                let name = format!("Screen_{arc}");
                let line = roof_of(arc).map(|r| r.edge_line.clone());
                if line.as_ref().is_some_and(|l| can_hang(l)) {
                    hang(st, &mut tubes, &mut put, name, line.as_ref().unwrap());
                } else if !top_mount(st, &mut tubes, &mut put, name.clone(), arc) {
                    // freestanding in the open corner, far enough out that its width clears both stands
                    let (a, b) = stands.map(|s| (s.a, s.b)).unwrap_or((l / 2.0 + 10.0, wp / 2.0 + 8.0));
                    let d = (4.0 * std::f64::consts::SQRT_2).max(w / 2.0 + 2.0);
                    let xy = DVec2::new(sx * (a + d / std::f64::consts::SQRT_2), sy * (b + d / std::f64::consts::SQRT_2));
                    on_frame(st, &mut tubes, &mut put, name, xy, 8.0, 0.0);
                }
            }
        }
        "center" => {
            let roofs = stands.map(|s| &s.roofs[..]).unwrap_or(&[]);
            let z_edge = roofs.iter().map(|r| r.1.edge_line.iter().map(|q| q.z).fold(f64::MAX, f64::min)).fold(f64::MAX, f64::min);
            let cz = (z_edge - 6.0 - h / 2.0).min(25.0 + h / 2.0);
            let half = w / 2.0 + 0.3;
            let mut caps = MeshAcc::new();
            for z in [cz - h / 2.0 - 0.4, cz + h / 2.0 + 0.4] {
                let e = half + 0.3;
                add_box(&mut caps, DVec3::new(-e, -e, z - 0.3), DVec3::new(e, e, z + 0.3), 0);
            }
            st.parts.push(Part { name: "ScreenCaps".into(), mesh: caps, mats: vec![Mat::tint("Screen_Body", "#6b6f75")], smooth_deg: 0.0 });
            for (k, (ux, uy)) in [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)].into_iter().enumerate() {
                put(st, format!("Screen_centre_{k}"), DVec3::new(ux * half, uy * half, cz), (-ux).atan2(uy), 0.0);
            }
            for r in roofs {
                let mid = midpoint(&r.1.edge_line);
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

/// The lit segments of a seven-segment digit in the box x0..x1, z0..z1 (segment thickness t).
fn segments(digit: usize, x0: f64, x1: f64, z0: f64, z1: f64, t: f64) -> Vec<(f64, f64, f64, f64)> {
    let zm = 0.5 * (z0 + z1);
    let all = [
        (x0 + t, x1 - t, z1 - t, z1),          // a: top
        (x1 - t, x1, zm + t / 2.0, z1 - t),    // b: top right
        (x1 - t, x1, z0 + t, zm - t / 2.0),    // c: bottom right
        (x0 + t, x1 - t, z0, z0 + t),          // d: bottom
        (x0, x0 + t, z0 + t, zm - t / 2.0),    // e: bottom left
        (x0, x0 + t, zm + t / 2.0, z1 - t),    // f: top left
        (x0 + t, x1 - t, zm - t / 2.0, zm + t / 2.0), // g: middle
    ];
    let lit = ["abcdef", "bc", "abged", "abgcd", "fgbc", "afgcd", "afgedc", "abc", "abcdefg", "abcdfg"][digit % 10];
    lit.bytes().map(|ch| all[(ch - b'a') as usize]).collect()
}

struct DugoutCfg {
    style: String,
    seats: usize,
    seat: String,
    color: String,
}

/// Dugout fronts sit 4.3 m behind the touchline, closer (not under 1.5 m) when the main stand's
/// front is near (see dugout.front_from_touchline).
fn dugout_front(p: &Pitch, stands: Option<&StandsOut>) -> f64 {
    match stands {
        Some(s) if s.runs.iter().any(|r| r.0.name == "south") => (s.b - p.width / 2.0 - DUGOUT_DEPTH - 0.65).min(DUGOUT_FRONT).max(1.5),
        _ => DUGOUT_FRONT,
    }
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

/// Clear polycarbonate panels (back above `z_base`, sides from `z_base`) under a roof sloping from
/// 2.35 m at the back to 2.15 m over the front, on a steel frame of posts, rails and rafters, with a
/// club-colour fascia along the front edge. `walled`: brick walls carry the back below z_base.
fn dugout_canopy(m: &mut MeshAcc, frame: &mut Vec<Polyline>, lx: f64, depth: f64, z_base: f64, walled: bool) {
    let (yb, yf, zb, zf) = (-depth - 0.05, 0.3, 2.35, 2.15);
    let zr = |y: f64| zb + (zf - zb) * (y - yb) / (yf - yb);
    let quad = |m: &mut MeshAcc, q: [DVec3; 4], tag: i32| {
        let base = m.verts.len() as u32;
        m.verts.extend(q);
        let uv = q.iter().map(|p| DVec2::new(p.x + p.y, p.z)).collect();
        m.add_face(vec![base, base + 1, base + 2, base + 3], tag, Some(uv));
    };
    let (x0, x1) = (-lx / 2.0, lx / 2.0);
    quad(m, [DVec3::new(x0, yb, zb), DVec3::new(x1, yb, zb), DVec3::new(x1, yf, zf), DVec3::new(x0, yf, zf)], 1);
    quad(m, [DVec3::new(x0, -depth + 0.02, z_base), DVec3::new(x1, -depth + 0.02, z_base), DVec3::new(x1, -depth + 0.02, zr(-depth)), DVec3::new(x0, -depth + 0.02, zr(-depth))], 1);
    for x in [x0, x1] {
        quad(m, [DVec3::new(x, -depth, z_base), DVec3::new(x, 0.0, z_base), DVec3::new(x, 0.0, zr(0.0)), DVec3::new(x, -depth, zr(-depth))], 1);
    }
    add_box(m, DVec3::new(x0 - 0.05, yf - 0.06, zf - 0.18), DVec3::new(x1 + 0.05, yf + 0.02, zf + 0.04), 4);
    let r = 0.03;
    for x in [x0, x1] {
        frame.push((vec![DVec3::new(x, -depth, if walled { 1.0 } else { 0.12 }), DVec3::new(x, -depth, zr(-depth))], r));
        frame.push((vec![DVec3::new(x, 0.0, if walled { 1.0 } else { 0.12 }), DVec3::new(x, 0.0, zr(0.0))], r));
        frame.push((vec![DVec3::new(x, -depth, z_base), DVec3::new(x, 0.0, z_base)], r));
    }
    frame.push((vec![DVec3::new(x0, yb, zb), DVec3::new(x1, yb, zb)], r));
    frame.push((vec![DVec3::new(x0, 0.0, zr(0.0)), DVec3::new(x1, 0.0, zr(0.0))], r));
    frame.push((vec![DVec3::new(x0, -depth + 0.02, z_base), DVec3::new(x1, -depth + 0.02, z_base)], r));
    let n = 2usize.max(crate::py::round_half_even(lx / 1.2) as usize);
    for k in 0..=n {
        let x = x0 + lx * k as f64 / n as f64;
        frame.push((vec![DVec3::new(x, yb, zb), DVec3::new(x, yf, zf)], 0.022));
        if k > 0 && k < n {
            frame.push((vec![DVec3::new(x, -depth + 0.02, z_base), DVec3::new(x, -depth + 0.02, zr(-depth))], 0.02));
        }
    }
}

fn build_dugouts(st: &mut Stadium, cfg: &Value, cat: &Catalog, stands: Option<&StandsOut>) -> Result<(), String> {
    let d = dugout_cfg(cfg);
    let lx = dugout_length(cfg, cat)?;
    let w_p = st.pitch.width;
    let front = dugout_front(&st.pitch, stands);
    let y = -(w_p / 2.0 + front);
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
            // base and lip rails along the shell, a club-colour back panel and a kick plate
            let (q0, q1) = (prof[0], prof[prof.len() - 1]);
            frame_lines.push((vec![DVec3::new(-lx / 2.0, q0.x, q0.y), DVec3::new(lx / 2.0, q0.x, q0.y)], 0.04));
            frame_lines.push((vec![DVec3::new(-lx / 2.0, q1.x, q1.y), DVec3::new(lx / 2.0, q1.x, q1.y)], 0.04));
            add_box(&mut m, DVec3::new(-lx / 2.0 + 0.05, -depth + 0.14, 0.42), DVec3::new(lx / 2.0 - 0.05, -depth + 0.2, 1.55), 4);
            add_box(&mut m, DVec3::new(-lx / 2.0, -0.08, 0.12), DVec3::new(lx / 2.0, 0.0, 0.32), 4);
        }
        "box" => {
            // a brick base (back wall and low side walls) under clear polycarbonate sides and a
            // sloping polycarbonate roof on a steel frame, with a club-colour fascia
            add_box(&mut m, DVec3::new(-lx / 2.0 - 0.25, -depth - 0.25, 0.0), DVec3::new(lx / 2.0 + 0.25, -depth, 1.1), 3);
            for sx in [-1.0, 1.0] {
                let (xa, xb) = if sx < 0.0 { (-lx / 2.0 - 0.25, -lx / 2.0) } else { (lx / 2.0, lx / 2.0 + 0.25) };
                add_box(&mut m, DVec3::new(xa, -depth - 0.25, 0.0), DVec3::new(xb, 0.0, 1.0), 3);
            }
            add_box(&mut m, DVec3::new(-lx / 2.0 - 0.3, -depth - 0.3, 1.1), DVec3::new(lx / 2.0 + 0.3, -depth + 0.05, 1.18), 0);
            dugout_canopy(&mut m, &mut frame_lines, lx, depth, 1.1, true);
        }
        _ => {
            mats[1] = Mat::tint("Dugout_Polycarbonate", "#dfe8ea");
            dugout_canopy(&mut m, &mut frame_lines, lx, depth, 0.3, false);
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
    // the fourth official's desk beside the tunnel exit, between the dugouts
    let mut desk = MeshAcc::new();
    let (dx, dy) = (DUGOUT_GAP / 2.0 - 0.6, y + 0.9);
    add_box(&mut desk, DVec3::new(dx - 0.7, dy - 0.3, 0.72), DVec3::new(dx + 0.7, dy + 0.3, 0.76), 0);
    add_box(&mut desk, DVec3::new(dx - 0.7, dy + 0.26, 0.05), DVec3::new(dx + 0.7, dy + 0.3, 0.72), 1);
    for sx in [-1.0, 1.0] {
        add_box(&mut desk, DVec3::new(dx + sx * 0.66 - 0.03, dy - 0.3, 0.0), DVec3::new(dx + sx * 0.66 + 0.03, dy + 0.3, 0.72), 0);
    }
    // the electronic substitution board in its holder at one end: a slim housing with a carry
    // handle, and on both faces the two-digit number cells, red (off) over green (on)
    let (bx0, by) = (dx - 0.62, dy - 0.05);
    add_box(&mut desk, DVec3::new(bx0, by - 0.035, 0.76), DVec3::new(bx0 + 0.62, by + 0.035, 1.24), 0);
    for hx in [bx0 + 0.1, bx0 + 0.52] {
        add_box(&mut desk, DVec3::new(hx - 0.015, by - 0.012, 1.24), DVec3::new(hx + 0.015, by + 0.012, 1.33), 0);
    }
    add_box(&mut desk, DVec3::new(bx0 + 0.085, by - 0.015, 1.3), DVec3::new(bx0 + 0.535, by + 0.015, 1.34), 0);
    for sy in [-1.0, 1.0] {
        let (ya, yb) = if sy < 0.0 { (by - 0.039, by - 0.035) } else { (by + 0.035, by + 0.039) };
        for (z0, mi, digits) in [(1.02, 2, [1usize, 4]), (0.8, 3, [2, 3])] {
            for (c, &dg) in [bx0 + 0.07, bx0 + 0.33].iter().zip(digits.iter()) {
                // read the right way round from the side it faces
                let x0 = if sy < 0.0 { *c } else { 2.0 * bx0 + 0.62 - c - 0.22 };
                for (sx0, sx1, sz0, sz1) in segments(dg, x0 + 0.03, x0 + 0.19, z0 + 0.01, z0 + 0.17, 0.024) {
                    // (the glyph itself mirrored too on the face read from +y)
                    let (sx0, sx1) = if sy < 0.0 { (sx0, sx1) } else { (2.0 * x0 + 0.22 - sx1, 2.0 * x0 + 0.22 - sx0) };
                    add_box(&mut desk, DVec3::new(sx0, ya, sz0), DVec3::new(sx1, yb, sz1), mi);
                }
            }
        }
    }
    // water bottles at the other end
    for (k, bxx) in [dx + 0.25, dx + 0.36, dx + 0.47].into_iter().enumerate() {
        let byy = dy - 0.1 + 0.06 * k as f64;
        add_box(&mut desk, DVec3::new(bxx - 0.035, byy - 0.035, 0.76), DVec3::new(bxx + 0.035, byy + 0.035, 0.98), 4);
        add_box(&mut desk, DVec3::new(bxx - 0.015, byy - 0.015, 0.98), DVec3::new(bxx + 0.015, byy + 0.015, 1.02), 0);
    }
    let led = |c: DVec3| Mat { emission: Some((c, 2.5)), ..Mat::tint("Dugout_Frame", "#101112") };
    st.parts.push(Part {
        name: "FourthOfficial".into(),
        mesh: desk,
        mats: vec![Mat::tint("Dugout_Frame", "#2b2d30"), Mat::tint("TINT_dugout", &d.color), led(DVec3::new(1.0, 0.08, 0.05)),
                   led(DVec3::new(0.1, 1.0, 0.25)), Mat::tint("Dugout_Polycarbonate", "#6fa6d6")],
        smooth_deg: 0.0,
    });
    // technical areas: dashed lines 1 m beyond the seats each side, to 1 m from the touchline
    let y_front = -(w_p / 2.0 + 1.0);
    let y_back = -(w_p / 2.0 + front);
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
    st.parts.push(Part { name: "TechnicalAreas".into(), mesh: paint, mats: vec![paint_mat(&st.pitch)], smooth_deg: 0.0 });
    Ok(())
}

/// Depth of the lit porch at the tunnel mouth: a telescopic tunnel keeps most of the run.
fn tunnel_porch_depth(style: &str, length: f64) -> f64 {
    if style == "telescopic" { 2.4f64.min(0.4 * length).max(0.6) } else { 2.4 }
}

fn build_tunnel(st: &mut Stadium, cfg: &Value, stands: Option<&StandsOut>) {
    let style = text(cfg, "style", "telescopic").to_string();
    let color = hexv(cfg, "color", "#c8102e");
    let (w, h) = (num(cfg, "width", 3.6), num(cfg, "height", 2.7));
    let y_end = -(st.pitch.width / 2.0 + dugout_front(&st.pitch, stands));
    let has_stand = stands.is_some_and(|s| s.runs.iter().any(|r| r.0.name == "south"));
    let y_wall = if has_stand { -stands.unwrap().b } else { y_end - 3.0 };
    let mut m = MeshAcc::new();
    // a lit porch in front of the stand wall: side walls with club-colour panels, a roof slab with
    // light strips and a club-colour header, a floor mat, and double doors in the back wall
    let dp = tunnel_porch_depth(&style, y_end - y_wall);
    let (y0, y1) = (y_wall, y_wall + dp);
    add_box(&mut m, DVec3::new(-w / 2.0, y0 - 0.02, 0.0), DVec3::new(w / 2.0, y0 + 0.01, h), 2);
    add_box(&mut m, DVec3::new(-1.05, y0 + 0.01, 0.0), DVec3::new(1.05, y0 + 0.04, 2.35), 1);
    for x in [-0.52, 0.52] {
        add_box(&mut m, DVec3::new(x - 0.45, y0 + 0.04, 0.9), DVec3::new(x + 0.45, y0 + 0.05, 2.2), 6);
    }
    for sx in [-1.0, 1.0] {
        add_box(&mut m, DVec3::new(sx * w / 2.0 - if sx < 0.0 { 0.3 } else { 0.0 }, y0, 0.0), DVec3::new(sx * w / 2.0 + if sx > 0.0 { 0.3 } else { 0.0 }, y1, h + 0.35), 2);
        let (xa, xb) = if sx < 0.0 { (-w / 2.0, -w / 2.0 + 0.02) } else { (w / 2.0 - 0.02, w / 2.0) };
        add_box(&mut m, DVec3::new(xa, y0 + 0.3, 1.0), DVec3::new(xb, y1 - 0.2, 2.2), 3);
    }
    for sx in [-1.0, 1.0] {
        // the club panels framed in trim, a steel handrail along each wall below them
        let xi = |o: f64| sx * (w / 2.0 - o);
        let (xa, xb) = (xi(0.02).min(xi(0.045)), xi(0.02).max(xi(0.045)));
        for (za, zb) in [(0.96, 1.0), (2.2, 2.24)] {
            add_box(&mut m, DVec3::new(xa, y0 + 0.26, za), DVec3::new(xb, y1 - 0.16, zb), 1);
        }
        for (ya, yb) in [(y0 + 0.26, y0 + 0.3), (y1 - 0.2, y1 - 0.16)] {
            add_box(&mut m, DVec3::new(xa, ya, 0.96), DVec3::new(xb, yb, 2.24), 1);
        }
        let (ra, rb) = (xi(0.07).min(xi(0.11)), xi(0.07).max(xi(0.11)));
        add_box(&mut m, DVec3::new(ra, y0 + 0.15, 0.86), DVec3::new(rb, y1 - 0.1, 0.9), 7);
        for y in [y0 + 0.4, y1 - 0.3] {
            add_box(&mut m, DVec3::new(xi(0.0).min(xi(0.09)), y - 0.02, 0.8), DVec3::new(xi(0.0).max(xi(0.09)), y + 0.02, 0.86), 7);
        }
    }
    // the doors: a meeting stile, a mid rail and stainless push bars across both leaves
    add_box(&mut m, DVec3::new(-0.04, y0 + 0.04, 0.0), DVec3::new(0.04, y0 + 0.07, 2.3), 1);
    add_box(&mut m, DVec3::new(-1.0, y0 + 0.04, 1.0), DVec3::new(1.0, y0 + 0.07, 1.1), 1);
    for x in [-0.52, 0.52] {
        add_box(&mut m, DVec3::new(x - 0.36, y0 + 0.08, 1.03), DVec3::new(x + 0.36, y0 + 0.12, 1.07), 7);
    }
    add_box(&mut m, DVec3::new(-w / 2.0 - 0.3, y0, h), DVec3::new(w / 2.0 + 0.3, y1, h + 0.35), 2);
    add_box(&mut m, DVec3::new(-w / 2.0 - 0.3, y1, h - 0.05), DVec3::new(w / 2.0 + 0.3, y1 + 0.06, h + 0.4), 3);
    for x in [-w / 4.0, w / 4.0] {
        add_box(&mut m, DVec3::new(x - 0.08, y0 + 0.2, h - 0.04), DVec3::new(x + 0.08, y1 - 0.2, h), 5);
    }
    add_box(&mut m, DVec3::new(-w / 2.0, y0, 0.0), DVec3::new(w / 2.0, y1, 0.03), 4);
    if !has_stand {
        add_box(&mut m, DVec3::new(-w / 2.0 - 1.2, y_wall - 4.0, 0.0), DVec3::new(w / 2.0 + 1.2, y_wall, h + 0.8), 2);
    }
    let light = Mat { emission: Some((DVec3::new(1.0, 0.97, 0.92), 8.0)), ..Mat::named("Tunnel_Hoop") };
    let glass = Mat { emission: Some((DVec3::new(1.0, 0.95, 0.85), 0.1)), ..Mat::named("Facade_DarkGlass") };
    let mats = vec![Mat::named("Tunnel_Mouth"), Mat::named("Tunnel_Trim"), Mat::named("Tunnel_Building"), Mat::tint("TINT_tunnel", &color), Mat::named("Tunnel_Mat"), light, glass, Mat::named("Tunnel_Hoop")];
    if style == "telescopic" {
        // telescoping segments, each a little larger than the one it slides into, overlapping it;
        // fabric drooping between each segment's end hoops and middle rib, a rubber skirt at the foot
        let (ya, yb) = (y1 + 0.1, y_end);
        let n_seg = (crate::py::round_half_even((yb - ya) / 1.6) as usize).clamp(2, 7);
        let seg = (yb - ya) / n_seg as f64;
        let hw = 0.5f64.max(h - w / 2.0);
        let mut prof = vec![DVec2::new(-w / 2.0, 0.0)];
        prof.extend(linspace(PI, 0.0, 17).iter().map(|a| DVec2::new((w / 2.0) * a.cos(), hw + (w / 2.0) * a.sin())));
        prof.push(DVec2::new(w / 2.0, 0.0));
        let mut hoops: Vec<Polyline> = Vec::new();
        let k = prof.len() as u32;
        for sg in 0..n_seg {
            let sc = 1.0 + 0.035 * sg as f64;
            let p: Vec<DVec2> = prof.iter().map(|q| DVec2::new(q.x * sc, q.y * sc)).collect();
            let s0 = ya + sg as f64 * seg;
            let s1 = ya + (sg as f64 + 1.0) * seg + if sg + 1 < n_seg { 0.12 } else { 0.0 };
            let base = m.verts.len() as u32;
            for i in 0..5 {
                let y = s0 + (s1 - s0) * i as f64 / 4.0;
                let sag = if i % 2 == 1 { 0.035 } else { 0.0 };
                for q in &p {
                    if q.y <= 1e-6 {
                        m.verts.push(DVec3::new(q.x, y, q.y));
                        continue;
                    }
                    let r = DVec2::new(q.x, q.y - hw * sc);
                    let qq = DVec2::new(0.0, hw * sc) + r * (1.0 - sag / r.length().max(1e-3));
                    m.verts.push(DVec3::new(qq.x, y, qq.y));
                }
            }
            for i in 0..4u32 {
                for j in 0..k - 1 {
                    let f = vec![base + i * k + j, base + (i + 1) * k + j, base + (i + 1) * k + j + 1, base + i * k + j + 1];
                    let uv = f.iter().map(|&v| { let q = m.verts[v as usize]; DVec2::new(q.y, q.x + q.z) }).collect();
                    m.add_face(f, 3, Some(uv));
                }
            }
            for (y, r) in [(s0, 0.03), (0.5 * (s0 + s1), 0.025), (s1, 0.05)] {
                hoops.push((p.iter().map(|q| DVec3::new(q.x * 1.01, y, q.y * 1.01)).collect(), r));
            }
            let xw = w / 2.0 * sc;
            for sx in [-1.0, 1.0] {
                add_box(&mut m, DVec3::new(sx * xw - 0.03, s0, 0.0), DVec3::new(sx * xw + 0.03, s1, 0.2), 4);
            }
        }
        if let Some(t) = tubes_part("TunnelHoops", &hoops, 8, Mat::named("Tunnel_Hoop")) {
            st.parts.push(t);
        }
        add_box(&mut m, DVec3::new(-w / 2.0, ya, 0.0), DVec3::new(w / 2.0, yb, 0.03), 4);
    }
    st.parts.push(Part { name: "Tunnel".into(), mesh: m, mats, smooth_deg: 40.0 });
    st.sockets.push(Socket { kind: "tunnel", id: "players".into(), loc: DVec3::new(0.0, y_end, 0.0), n: DVec2::new(0.0, 1.0), props: vec![("width", w), ("height", h)] });
}

// -- floodlights -------------------------------------------------------------------------------

fn build_floodlights(st: &mut Stadium, cfg: &Value, cat: &Catalog, stands: Option<&StandsOut>) -> Result<(), String> {
    let mut spec = resolve_preset(&cat.floodlights, cfg)?.unwrap_or_default();
    // a roof ribbon needs roof edges high enough to light the pitch from; under low roofs the
    // lamps would sit a few metres above the grass, so light it from corner pylons instead
    if text(&spec, "type", "") == "roof" {
        let low = stands.map(|s| s.roofs.iter().flat_map(|r| r.1.edge_line.iter().map(|q| q.z)).fold(f64::MAX, f64::min)).unwrap_or(f64::MAX);
        if low < ROOF_LIGHTS_MIN_HEIGHT {
            st.report.warnings.push(format!("roof edges at {low:.1} m are too low for roof floodlights; using corner pylons"));
            spec["type"] = json!("corner_pylons");
            spec["fixture"] = json!("led");
        }
    }
    let on = spec.get("on").and_then(|v| v.as_bool()).unwrap_or(st.night);
    let info = stands.map(|s| StandInfo {
        a: s.a,
        b: s.b,
        // how far out the structures go: facades, the stands themselves, roof back spans
        reach: s
            .facade_lines
            .iter()
            .map(|f| f.1)
            .chain(s.runs.iter().map(|r| stand_depth(&r.1)))
            .chain(s.roofs.iter().map(|r| r.1.tail_s))
            .fold(10.0, f64::max),
        roof_top: s.roofs.iter().map(|r| r.1.tail_top).fold(0.0, f64::max),
        roof_edges: s.roofs.iter().map(|r| r.1.edge_line.clone()).collect(),
        corner_r: s.corner_r,
        main_half: s.runs.iter().find(|r| r.0.name == "south").map(|(r, _)| r.world(r.t0, 0.0, 0.0).x.abs().max(r.world(r.t1, 0.0, 0.0).x.abs())).unwrap_or(0.0),
    });
    let p = &st.pitch;
    let d = floodlights::design(&spec, p.length, p.width, p.runoff_side, info.as_ref())?;
    let fixture_key = format!("floodlight_fixture.{}", d.fixture);
    let sc = d.lamp_scale.max(1e-3);
    let pivot = floodlights::fixture_pivot(&d.fixture) * sc;
    let items = d
        .lamps
        .iter()
        .map(|(pos, t)| Placement { scale: DVec3::splat(sc), ..Placement::aim(floodlights::mount_point(*pos, *t, pivot), *t - *pos) })
        .collect();
    // lenses glow ("EMIT:" lights a material up, see viewer build.rs): full at night, and stadium
    // floodlights are switched on for televised day matches too, so roof rings and masts glow
    // by day as well (grassroots poles stay off by day)
    let glow = if on { 2.5 } else if d.kind == "poles" { 0.0 } else { 1.6 };
    let tints = if glow > 0.0 { vec![("EMIT:Fixture_Lens".to_string(), floodlights::kelvin_rgb(d.kelvin) * glow)] } else { vec![] };
    st.props.push(PropSet { key: fixture_key, items, tints, net: None });
    if let Some(t) = tubes_part("FloodlightSteel", &d.steel, 6, Mat::named("Galvanised")) {
        st.parts.push(t);
    }
    if d.kind == "roof" && d.fixture == "led" {
        // the lit diffuser panels of the roof ring's floods (dimmer by day, when they are not on)
        let c = floodlights::kelvin_rgb(d.kelvin);
        let glow = Mat { emission: Some((c, if on { 14.0 } else { 7.0 })), ..Mat::named("Tunnel_Hoop") };
        st.parts.push(Part { name: "FloodlightLenses".into(), mesh: floodlights::lens_panels(&d.lamps, sc), mats: vec![glow], smooth_deg: 0.0 });
    }
    if !d.mesh.faces.is_empty() {
        st.parts.push(Part { name: "FloodlightPlates".into(), mesh: d.mesh.clone(), mats: vec![Mat::named("Galvanised"), Mat::tint("Stand_Concrete", "#a29e95"), Mat::named("Fixture_Body")], smooth_deg: 0.0 });
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
