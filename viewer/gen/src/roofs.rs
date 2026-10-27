//! Roofs over a stand run (port of kit/stands/roofs.py): cantilever and goalpost systems, the
//! sightline/headroom solver, deck, fascia and steel tubes.

use crate::catalog::text;
use crate::mesh::{MeshAcc, Section};
use crate::py::{clip, linspace, round_half_even, round_to};
use crate::runs::Run;
use crate::sightlines::{Tier, SEATED_EYE, SEATED_EYE_FWD, WALL};
use crate::stands::{stand_depth, Polyline, Socket};
use glam::{DVec2, DVec3};
use serde_json::Value;

pub const TAG_DECK: i32 = 0;
pub const TAG_TRANSLUCENT: i32 = 1;
pub const TAG_FASCIA: i32 = 2;
pub const TAG_UNDERSIDE: i32 = 3;
pub const TAG_STEEL: i32 = 4;
pub const BALL_HEIGHT: f64 = 15.0;
/// rafter ribs expressed above a cantilever's sheeting
pub const RIB_W: f64 = 0.45;
pub const RIB_H: f64 = 0.4;
pub const HEADROOM: f64 = 2.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoofKind {
    Cantilever,
    Goalpost,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub rise_deg: f64,
    pub slope_deg: f64,
    pub back_clearance: f64,
    pub cover: f64,
    pub back_span: f64,
    pub translucent: f64,
    pub max_cantilever: f64,
    pub truss_every: usize,
    pub post_every: usize,
    pub girder_depth: Option<f64>,
    pub clad_girder: bool,
    /// rooflight bands across a metal deck
    pub rooflights: usize,
}

impl Params {
    pub fn resolve(kind: RoofKind, spec: &Value) -> Params {
        let g = |k: &str, d: f64| match spec.get(k) {
            Some(Value::Number(n)) => n.as_f64().unwrap_or(d),
            _ => d,
        };
        let mut p = match kind {
            RoofKind::Cantilever => Params {
                rise_deg: g("rise_deg", 5.0),
                slope_deg: 0.0,
                back_clearance: g("back_clearance", 3.0),
                cover: g("cover", 0.0),
                back_span: g("back_span", 0.22),
                translucent: g("translucent", 0.25),
                max_cantilever: g("max_cantilever", 60.0),
                truss_every: g("truss_every", 1.0) as usize,
                post_every: 3,
                girder_depth: None,
                clad_girder: true,
                rooflights: 0,
            },
            RoofKind::Goalpost => Params {
                rise_deg: 0.0,
                slope_deg: g("slope_deg", 3.0),
                back_clearance: g("back_clearance", 2.6),
                cover: g("cover", 1.0),
                back_span: 0.0,
                translucent: g("translucent", 0.0),
                max_cantilever: 60.0,
                truss_every: 1,
                post_every: g("post_every", 3.0) as usize,
                girder_depth: spec.get("girder_depth").and_then(|v| v.as_f64()).filter(|v| *v != 0.0),
                clad_girder: spec.get("clad_girder").and_then(|v| v.as_bool()).unwrap_or(true),
                rooflights: 0,
            },
        };
        // rooflight bands across a metal deck (none on a membrane or an all-translucent roof)
        p.rooflights = if text(spec, "deck", "metal") == "membrane" || p.translucent >= 1.0 { 0 } else { g("rooflights", 2.0) as usize };
        p
    }
}

#[derive(Clone, Debug)]
pub struct Roof {
    pub kind: RoofKind,
    pub params: Params,
    pub s_back: f64,
    pub s_front: f64,
    pub z_back: f64,
    pub z_front: f64,
    pub depth_back: f64,
    pub depth_front: f64,
    pub length: f64,
    pub back_span: f64,
    pub s_post: f64,
    pub raised_for_sightline: f64,
    pub raised_for_headroom: f64,
    pub clearance_to_sightline: f64,
}

fn line_z(eye: DVec2, target: DVec2, s: f64) -> f64 {
    eye.y + (target.y - eye.y) * (eye.x - s) / (eye.x - target.x)
}

pub fn kind_of(spec: &Value) -> RoofKind {
    if text(spec, "type", "cantilever") == "goalpost" { RoofKind::Goalpost } else { RoofKind::Cantilever }
}

pub fn solve(tiers: &[Tier], spec: &Value, d_focal: f64, far_width: f64) -> Roof {
    let kind = kind_of(spec);
    let p = Params::resolve(kind, spec);
    let top = &tiers[tiers.len() - 1];
    let s_back = stand_depth(tiers);
    let h_last = tiers.iter().map(|t| t.h_last()).fold(f64::MIN, f64::max);
    let s_front = tiers[0].s0 - WALL;
    let eye = DVec2::new(top.s[top.s.len() - 1] + top.tread - SEATED_EYE_FWD, top.h_last() + SEATED_EYE);
    let target = DVec2::new(-(d_focal + far_width), BALL_HEIGHT);
    let mut r = Roof {
        kind,
        params: p.clone(),
        s_back,
        s_front: 0.0,
        z_back: 0.0,
        z_front: 0.0,
        depth_back: 0.0,
        depth_front: 0.0,
        length: 0.0,
        back_span: 0.0,
        s_post: 0.0,
        raised_for_sightline: 0.0,
        raised_for_headroom: 0.0,
        clearance_to_sightline: 0.0,
    };
    match kind {
        RoofKind::Cantilever => {
            let s_f = (s_front - p.cover).max(s_back - p.max_cantilever);
            let l = s_back - s_f;
            let z_b = h_last + p.back_clearance;
            let mut z_f = z_b + l * p.rise_deg.to_radians().tan();
            let need = line_z(eye, target, s_f) + 0.5;
            let raised = (need - z_f).max(0.0);
            z_f += raised;
            r.s_front = s_f;
            r.z_back = z_b;
            r.z_front = z_f;
            r.depth_back = clip(l / 7.0, 1.2, 9.0);
            r.depth_front = clip(l / 40.0, 0.5, 1.5);
            r.length = l;
            r.back_span = l * p.back_span;
            r.raised_for_sightline = round_to(raised, 3);
        }
        RoofKind::Goalpost => {
            let s_g = s_front + p.cover;
            let l = s_back - s_g;
            let z_b = h_last + p.back_clearance;
            let mut z_g = z_b - l * p.slope_deg.to_radians().tan();
            let need = line_z(eye, target, s_g) + 0.3;
            let raised = (need - z_g).max(0.0);
            z_g += raised;
            r.s_front = s_g;
            r.z_back = z_b;
            r.z_front = z_g;
            r.length = l;
            r.raised_for_sightline = round_to(raised, 3);
            r.s_post = s_front + 0.45;
        }
    }
    let (zb, zf, sb, sf) = (r.z_back, r.z_front, r.s_back, r.s_front);
    let mut need_f = zf;
    for t in tiers {
        for (s_i, h_i) in t.s.iter().zip(&t.h) {
            if sf <= *s_i && *s_i < sb - 1e-6 {
                need_f = need_f.max(zb + (h_i + HEADROOM - zb) * (sb - sf) / (sb - s_i));
            }
        }
    }
    r.raised_for_headroom = round_to(need_f - zf, 3);
    r.z_front = need_f;
    r.clearance_to_sightline = round_to(r.z_front - line_z(eye, target, r.s_front), 3);
    r
}

fn truss_nodes(roof: &Roof) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (s_b, s_f, z_b, z_f, d_b, d_f) = (roof.s_back, roof.s_front, roof.z_back, roof.z_front, roof.depth_back, roof.depth_front);
    let n = 3usize.max(round_half_even(roof.length / 1.5f64.max(0.6 * (d_b + d_f))) as usize);
    let u = linspace(0.0, 1.0, n + 1);
    let s = u.iter().map(|u| s_b + (s_f - s_b) * u).collect();
    let zb: Vec<f64> = u.iter().map(|u| z_b + (z_f - z_b) * u).collect();
    let zt = u.iter().zip(&zb).map(|(u, zb)| zb + d_b + (d_f - d_b) * u).collect();
    (s, zb, zt)
}

pub fn top_z(roof: &Roof, s: f64) -> f64 {
    let zt_b = roof.z_back + roof.depth_back;
    let zt_f = roof.z_front + roof.depth_front;
    zt_b + (zt_f - zt_b) * (roof.s_back - s) / (roof.s_back - roof.s_front)
}

/// Deck plate along the top of the structure: a translucent leading strip (optional), rooflight
/// bands 1.5 m wide further back, and a maintenance walkway near the back edge.
fn deck(acc: &mut MeshAcc, run: &Run, s_a: f64, s_b_end: f64, z_of: &dyn Fn(f64) -> f64, translucent: f64, rooflights: usize) {
    let l = s_b_end - s_a;
    let mut segs: Vec<(f64, f64, i32)> = Vec::new();
    if translucent > 0.0 {
        segs.push((s_a, s_a + translucent * l, TAG_TRANSLUCENT));
    }
    let mut s = segs.last().map(|x| x.1).unwrap_or(s_a);
    for k in 0..rooflights {
        let c = s_a + l * (0.45 + 0.3 * k as f64 / (rooflights.max(2) - 1) as f64);
        if c - 0.75 > s + 0.5 && c + 0.75 < s_b_end - 3.0 {
            segs.push((s, c - 0.75, TAG_DECK));
            segs.push((c - 0.75, c + 0.75, TAG_TRANSLUCENT));
            s = c + 0.75;
        }
    }
    segs.push((s, s_b_end, TAG_DECK));
    for &(s1, s2, tag) in &segs {
        let (z1, z2) = (z_of(s1), z_of(s2));
        let sec = Section::new(
            vec![DVec2::new(s1, z1), DVec2::new(s2, z2), DVec2::new(s2, z2 + 0.12), DVec2::new(s1, z1 + 0.12)],
            vec![TAG_UNDERSIDE, TAG_FASCIA, tag, TAG_FASCIA],
        );
        acc.add_prism(run, run.t0, run.t1, &sec, TAG_FASCIA);
    }
    if l > 12.0 {
        let (w1, w2) = (s_b_end - 2.6, s_b_end - 2.0);
        let sec = Section::uniform(vec![DVec2::new(w1, z_of(w1) + 0.12), DVec2::new(w2, z_of(w2) + 0.12), DVec2::new(w2, z_of(w2) + 0.2), DVec2::new(w1, z_of(w1) + 0.2)], TAG_UNDERSIDE);
        acc.add_prism(run, run.t0, run.t1, &sec, TAG_UNDERSIDE);
    }
}

/// Box gutter along the back edge (s0 .. s0 + 0.45, top at z_top) and downpipes to the ground.
fn gutter(acc: &mut MeshAcc, tubes: &mut Vec<Polyline>, run: &Run, s0: f64, z_top: f64, downpipe_ts: &[f64]) {
    let sec = Section::uniform(vec![DVec2::new(s0, z_top - 0.3), DVec2::new(s0 + 0.45, z_top - 0.3), DVec2::new(s0 + 0.45, z_top), DVec2::new(s0, z_top)], TAG_FASCIA);
    acc.add_prism(run, run.t0, run.t1, &sec, TAG_FASCIA);
    let m = run.metric(s0 + 0.22);
    for &t in downpipe_ts {
        let t = (t + 0.35 / m).max(run.t0).min(run.t1);
        tubes.push((vec![run.world(t, s0 + 0.22, z_top - 0.3), run.world(t, s0 + 0.22, 0.0)], 0.055));
    }
}

fn rect(s0: f64, s1: f64, z0: f64, z1: f64) -> Vec<DVec2> {
    vec![DVec2::new(s0, z0), DVec2::new(s1, z0), DVec2::new(s1, z1), DVec2::new(s0, z1)]
}

#[derive(Clone, Debug, Default)]
pub struct RoofOut {
    pub mesh: MeshAcc,
    pub tubes: Vec<Polyline>,
    pub sockets: Vec<Socket>,
    pub edge_line: Vec<DVec3>,
    pub tail_s: f64,
    pub tail_top: f64,
    /// top of the leading edge (fascia or clad girder)
    pub edge_top: f64,
    pub girder_depth: f64,
    /// highest point of the roof (deck and steel)
    pub top: f64,
}

pub fn build(run: &Run, roof: &Roof, bay_list: &[(f64, f64)], include_ends: bool) -> RoofOut {
    let mut o = RoofOut::default();
    let mut bounds: Vec<f64> = bay_list.iter().map(|b| b.0).collect();
    bounds.push(bay_list[bay_list.len() - 1].1);
    if !include_ends {
        bounds = bounds[1..bounds.len() - 1].to_vec();
    }
    let w = |t: f64, s: f64, z: f64| run.world(t, s, z);
    let (z_edge_lo, z_edge_hi);
    match roof.kind {
        RoofKind::Cantilever => {
            let (s, zb, zt) = truss_nodes(roof);
            let l = roof.length;
            // primary chords visibly heavier than the lacing
            let r_chord = 0.11 + l / 350.0;
            let r_web = 0.045 + l / 1200.0;
            let s_tail = roof.s_back + roof.back_span;
            let every = roof.params.truss_every.max(1);
            let (first, last) = (bounds.first().copied(), bounds.last().copied());
            let mut trusses = Vec::new();
            for (k, &t) in bounds.iter().enumerate() {
                if k % every != 0 && Some(t) != first && Some(t) != last {
                    continue;
                }
                trusses.push(t);
                o.tubes.push((s.iter().zip(&zb).map(|(a, b)| w(t, *a, *b)).collect(), r_chord));
                let mut top = vec![w(t, s_tail, top_z(roof, s_tail))];
                top.extend(s.iter().zip(&zt).map(|(a, b)| w(t, *a, *b)));
                o.tubes.push((top, r_chord));
                for i in 0..s.len() - 1 {
                    let (a, b) = (i, i + 1);
                    let even = i % 2 == 0;
                    o.tubes.push((
                        vec![w(t, s[a], if even { zb[a] } else { zt[a] }), w(t, s[b], if even { zt[b] } else { zb[b] })],
                        r_web,
                    ));
                }
                o.tubes.push((vec![w(t, s[0], zb[0]), w(t, s[0], zt[0])], r_web));
                o.tubes.push((vec![w(t, s[0], 0.0), w(t, s[0], zt[0])], 0.3f64.max(r_chord * 2.2)));
                o.tubes.push((vec![w(t, s_tail, 0.0), w(t, s_tail, top_z(roof, s_tail))], r_chord * 0.8));
            }
            let lift = r_chord + 0.08;
            // a ring beam tying the back columns together just above the stand's rear parapet, so
            // they visibly land on structure rather than ending in the air over the back rows
            let zp = roof.z_back - roof.params.back_clearance + 1.1;
            let sb = roof.s_back;
            o.mesh.add_prism(run, run.t0, run.t1, &Section::uniform(rect(sb - 0.35, sb + 0.35, zp, zp + 0.6), TAG_STEEL), TAG_STEEL);
            // purlins (300 mm deep) along the run at every top-chord node, under the sheeting
            for (a, b) in s[1..].iter().zip(&zt[1..]) {
                o.mesh.add_prism(run, run.t0, run.t1, &Section::uniform(rect(a - 0.06, a + 0.06, b + lift - 0.32, b + lift), TAG_STEEL), TAG_STEEL);
            }
            // secondary rafters halfway between the trusses
            let under = |ss: f64| top_z(roof, ss) + lift;
            let sf0 = roof.s_front - 0.2;
            let rafter = Section::uniform(vec![DVec2::new(sf0, under(sf0) - 0.35), DVec2::new(s_tail, under(s_tail) - 0.35), DVec2::new(s_tail, under(s_tail)), DVec2::new(sf0, under(sf0))], TAG_STEEL);
            for w2 in trusses.windows(2) {
                let (tm, hw) = (0.5 * (w2[0] + w2[1]), 0.08 / run.metric(roof.s_front));
                o.mesh.add_prism(run, tm - hw, tm + hw, &rafter, TAG_STEEL);
            }
            deck(&mut o.mesh, run, roof.s_front - 0.35, s_tail, &|ss| top_z(roof, ss) + lift, roof.params.translucent, roof.params.rooflights);
            // a deep fascia closing the leading edge, and a gutter with downpipes at the back
            let (sf, zt_f) = (roof.s_front, top_z(roof, roof.s_front) + lift + 0.25);
            let sec = Section::new(
                vec![DVec2::new(sf - 0.45, roof.z_front - 0.4), DVec2::new(sf - 0.2, roof.z_front - 0.4), DVec2::new(sf - 0.2, zt_f), DVec2::new(sf - 0.45, zt_f)],
                vec![TAG_UNDERSIDE, TAG_FASCIA, TAG_FASCIA, TAG_FASCIA],
            );
            o.mesh.add_prism(run, run.t0, run.t1, &sec, TAG_FASCIA);
            gutter(&mut o.mesh, &mut o.tubes, run, s_tail, top_z(roof, s_tail) + lift + 0.05, &trusses);
            // the primary rafters expressed above the sheeting on every other truss line
            let top = |ss: f64| top_z(roof, ss) + lift + 0.12;
            let rib = Section::uniform(vec![DVec2::new(sf - 0.2, top(sf - 0.2)), DVec2::new(s_tail, top(s_tail)), DVec2::new(s_tail, top(s_tail) + RIB_H), DVec2::new(sf - 0.2, top(sf - 0.2) + RIB_H)], TAG_UNDERSIDE);
            for &t in trusses.iter().step_by(2) {
                let hw = 0.5 * RIB_W / run.metric(roof.s_front);
                o.mesh.add_prism(run, (t - hw).max(run.t0), (t + hw).min(run.t1), &rib, TAG_UNDERSIDE);
            }
            z_edge_lo = roof.z_front - 0.35;
            z_edge_hi = top_z(roof, roof.s_front) + 0.3;
        }
        RoofKind::Goalpost => {
            let p = &roof.params;
            let (s_g, z_g, s_b) = (roof.s_front, roof.z_front, roof.s_back);
            let bay_w = (bay_list[0].1 - bay_list[0].0) * run.metric(s_g);
            let span = p.post_every as f64 * bay_w;
            let dg = p.girder_depth.unwrap_or_else(|| clip(span / 10.0, 0.6, 3.0));
            o.girder_depth = dg;
            let z_top_back = roof.z_back + 0.5;
            let z_of = |ss: f64| (z_g + dg) + (z_top_back - z_g - dg) * (ss - s_g) / (s_b - s_g);
            let n0 = run.samples(run.t0, run.t1, s_g).len();
            let n1 = ((run.t1 - run.t0) * run.metric(s_g) / dg) as i64 as usize + 1;
            let ts = linspace(run.t0, run.t1, n0.max(n1));
            let r = 0.06 + dg / 30.0;
            o.tubes.push((ts.iter().map(|&t| w(t, s_g, z_g)).collect(), r));
            o.tubes.push((ts.iter().map(|&t| w(t, s_g, z_g + dg)).collect(), r));
            for i in 0..ts.len() - 1 {
                let odd = i % 2 == 1;
                o.tubes.push((
                    vec![w(ts[i], s_g, z_g + if odd { dg } else { 0.0 }), w(ts[i + 1], s_g, z_g + if odd { 0.0 } else { dg })],
                    r * 0.6,
                ));
            }
            // posts count bays from the run's start; a run without its ends (a bowl's corner arc)
            // leaves the ends to its neighbours, which already have a post there
            let last = bounds.last().copied().filter(|_| include_ends);
            let first = usize::from(!include_ends);
            let r_post = 0.16 + z_g / 150.0;
            for (k, &t) in bounds.iter().enumerate() {
                o.tubes.push((vec![w(t, s_g, z_g + dg), w(t, s_b, z_top_back)], 0.12));
                if (k + first) % p.post_every.max(1) == 0 || Some(t) == last {
                    o.tubes.push((vec![w(t, roof.s_post, 0.0), w(t, roof.s_post, z_g)], r_post));
                }
            }
            deck(&mut o.mesh, run, s_g - 0.3, s_b + 0.3, &z_of, p.translucent, p.rooflights);
            // purlins every 1.5 m under the sheeting, and base plates under the posts
            for sp in crate::py::arange(s_g + 0.8, s_b, 1.5) {
                o.mesh.add_prism(run, run.t0, run.t1, &Section::uniform(rect(sp - 0.05, sp + 0.05, z_of(sp) - 0.22, z_of(sp)), TAG_STEEL), TAG_STEEL);
            }
            for (k, &t) in bounds.iter().enumerate() {
                if (k + first) % p.post_every.max(1) == 0 || Some(t) == last {
                    let hw = 0.25 / run.metric(roof.s_post);
                    o.mesh.add_prism(run, (t - hw).max(run.t0), (t + hw).min(run.t1), &Section::uniform(rect(roof.s_post - 0.25, roof.s_post + 0.25, 0.0, 0.04), TAG_STEEL), TAG_STEEL);
                }
            }
            let posts: Vec<f64> = bounds.iter().enumerate().filter(|(k, _)| (k + first) % p.post_every.max(1) == 0).map(|(_, t)| *t).collect();
            gutter(&mut o.mesh, &mut o.tubes, run, s_b + 0.3, z_of(s_b + 0.3) + 0.05, &posts);
            if p.clad_girder {
                let sec = Section::new(
                    vec![
                        DVec2::new(s_g - 0.35, z_g - 0.05),
                        DVec2::new(s_g + 0.25, z_g - 0.05),
                        DVec2::new(s_g + 0.25, z_g + dg + 0.25),
                        DVec2::new(s_g - 0.35, z_g + dg + 0.25),
                    ],
                    vec![TAG_UNDERSIDE, TAG_FASCIA, TAG_DECK, TAG_FASCIA],
                );
                o.mesh.add_prism(run, run.t0, run.t1, &sec, TAG_FASCIA);
            }
            z_edge_lo = z_g - 0.05;
            z_edge_hi = z_g + dg + 0.25;
        }
    }
    for (b, &(ta, tb)) in bay_list.iter().enumerate() {
        let (p, n) = run.frame(0.5 * (ta + tb));
        let loc = p + n * (roof.s_front - 0.4);
        o.sockets.push(Socket {
            kind: "roof_edge",
            id: format!("{}_B{}", run.name, b),
            loc: DVec3::new(loc.x, loc.y, z_edge_lo),
            n,
            props: vec![
                ("width", round_to((tb - ta) * run.metric(roof.s_front), 3)),
                ("edge_height", round_to(z_edge_hi - z_edge_lo, 3)),
            ],
        });
    }
    o.edge_top = z_edge_hi;
    o.edge_line = run.samples(run.t0, run.t1, roof.s_front).iter().map(|&t| w(t, roof.s_front - 0.4, z_edge_lo)).collect();
    o.top = o.mesh.verts.iter().map(|v| v.z).chain(o.tubes.iter().flat_map(|t| t.0.iter().map(|q| q.z))).fold(f64::MIN, f64::max);
    match roof.kind {
        RoofKind::Cantilever => {
            o.tail_s = roof.s_back + roof.back_span;
            o.tail_top = top_z(roof, o.tail_s) + 0.3;
        }
        RoofKind::Goalpost => {
            o.tail_s = roof.s_back + 0.3;
            o.tail_top = roof.z_back + 0.7;
        }
    }
    o
}
