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
pub const BALL_HEIGHT: f64 = 15.0;
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
}

impl Params {
    pub fn resolve(kind: RoofKind, spec: &Value) -> Params {
        let g = |k: &str, d: f64| match spec.get(k) {
            Some(Value::Number(n)) => n.as_f64().unwrap_or(d),
            _ => d,
        };
        match kind {
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
            },
        }
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

fn deck(acc: &mut MeshAcc, run: &Run, s_a: f64, s_b_end: f64, z_of: &dyn Fn(f64) -> f64, translucent: f64) {
    let mut edges = vec![s_a];
    if translucent > 0.0 {
        edges.push(s_a + translucent * (s_b_end - s_a));
    }
    edges.push(s_b_end);
    for k in 0..edges.len() - 1 {
        let (s1, s2) = (edges[k], edges[k + 1]);
        let tag = if translucent > 0.0 && k == 0 { TAG_TRANSLUCENT } else { TAG_DECK };
        let (z1, z2) = (z_of(s1), z_of(s2));
        let sec = Section::new(
            vec![DVec2::new(s1, z1), DVec2::new(s2, z2), DVec2::new(s2, z2 + 0.12), DVec2::new(s1, z1 + 0.12)],
            vec![TAG_UNDERSIDE, TAG_FASCIA, tag, TAG_FASCIA],
        );
        acc.add_prism(run, run.t0, run.t1, &sec, TAG_FASCIA);
    }
}

#[derive(Clone, Debug, Default)]
pub struct RoofOut {
    pub mesh: MeshAcc,
    pub tubes: Vec<Polyline>,
    pub sockets: Vec<Socket>,
    pub edge_line: Vec<DVec3>,
    pub tail_s: f64,
    pub tail_top: f64,
    pub girder_depth: f64,
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
            let r_chord = 0.09 + l / 450.0;
            let r_web = 0.05 + l / 900.0;
            let s_tail = roof.s_back + roof.back_span;
            let every = roof.params.truss_every.max(1);
            let (first, last) = (bounds.first().copied(), bounds.last().copied());
            for (k, &t) in bounds.iter().enumerate() {
                if k % every != 0 && Some(t) != first && Some(t) != last {
                    continue;
                }
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
            let ts = run.samples(run.t0, run.t1, roof.s_back);
            for (a, b) in s[1..].iter().zip(&zt[1..]) {
                o.tubes.push((ts.iter().map(|&t| w(t, *a, b + r_chord + 0.02)).collect(), 0.06));
            }
            let lift = r_chord + 0.08;
            deck(&mut o.mesh, run, roof.s_front - 0.35, s_tail, &|ss| top_z(roof, ss) + lift, roof.params.translucent);
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
            let last = bounds.last().copied();
            for (k, &t) in bounds.iter().enumerate() {
                o.tubes.push((vec![w(t, s_g, z_g + dg), w(t, s_b, z_top_back)], 0.12));
                if k % p.post_every.max(1) == 0 || Some(t) == last {
                    o.tubes.push((vec![w(t, roof.s_post, 0.0), w(t, roof.s_post, z_g)], 0.16));
                }
            }
            deck(&mut o.mesh, run, s_g - 0.3, s_b + 0.3, &z_of, p.translucent);
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
    o.edge_line = run.samples(run.t0, run.t1, roof.s_front).iter().map(|&t| w(t, roof.s_front - 0.4, z_edge_lo)).collect();
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
