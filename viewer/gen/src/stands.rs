//! Stand geometry (port of kit/stands/geometry.py): cross-sections swept along runs, bays, aisles,
//! vomitories, seat anchors, rails, nosings, crush barriers and sockets.

use crate::mesh::{MeshAcc, Section};
use crate::py::{floordiv, linspace, round_to};
use crate::runs::{yaw_facing_pitch, Run};
use crate::sightlines::{Kind, Tier, WALL};
use glam::{DVec2, DVec3};

pub const HALF_AISLE: f64 = 0.5;
pub const AISLE_RISER_MAX: f64 = 0.19;
pub const VOM_WIDTH: f64 = 2.2;
pub const VOM_HEADROOM: f64 = 2.2;

pub const TAG_CONCRETE: i32 = 0;
pub const TAG_FASCIA: i32 = 1;
pub const TAG_SOFFIT: i32 = 2;
pub const TAG_BLOCK: i32 = 3;

fn v2(s: f64, z: f64) -> DVec2 {
    DVec2::new(s, z)
}

/// Top surface polyline of the given rows (front nosing -> back of last tread).
pub fn tread_path(tier: &Tier, aisle: bool, rows: std::ops::Range<usize>) -> Vec<DVec2> {
    let (s, h, t) = (&tier.s, &tier.h, tier.tread);
    let mut pts = Vec::new();
    for i in rows.clone() {
        pts.push(v2(s[i], h[i]));
        if i + 1 < tier.rows && rows.contains(&(i + 1)) {
            let r = h[i + 1] - h[i];
            let n = if aisle { 1usize.max((r / AISLE_RISER_MAX).ceil() as usize) } else { 1 };
            for k in 1..n {
                pts.push(v2(s[i] + k as f64 * t / n as f64, h[i] + (k - 1) as f64 * r / n as f64));
                pts.push(v2(s[i] + k as f64 * t / n as f64, h[i] + k as f64 * r / n as f64));
            }
            pts.push(v2(s[i] + t, h[i] + (n - 1) as f64 * r / n as f64));
        } else {
            pts.push(v2(s[i] + t, h[i]));
        }
    }
    pts
}

fn bottom(tier: &Tier, s: f64) -> f64 {
    if tier.soffit_front.is_none() { 0.0 } else { tier.soffit_z(s) }
}

pub fn clean(pts: Vec<DVec2>, tags: Vec<i32>) -> Section {
    let mut out: Vec<DVec2> = Vec::new();
    let mut ot = Vec::new();
    for (p, t) in pts.into_iter().zip(tags) {
        if let Some(l) = out.last() {
            if (p.x - l.x).abs() < 1e-6 && (p.y - l.y).abs() < 1e-6 {
                continue;
            }
        }
        out.push(p);
        ot.push(t);
    }
    if out.len() > 1 {
        let (f, l) = (out[0], out[out.len() - 1]);
        if (f.x - l.x).abs() < 1e-6 && (f.y - l.y).abs() < 1e-6 {
            out.pop();
            ot.pop();
        }
    }
    Section::new(out, ot)
}

pub fn tier_section(tier: &Tier, aisle: bool) -> Section {
    let back_parapet = 1.1;
    let (s0, h0, sb) = (tier.s0, tier.h[0], tier.s_back);
    let top = h0 + tier.parapet;
    let hb = tier.h_last();
    let mut pts = vec![v2(s0 - WALL, bottom(tier, s0 - WALL)), v2(s0 - WALL, top), v2(s0, top)];
    let mut tags = vec![TAG_FASCIA, TAG_FASCIA, TAG_CONCRETE];
    let path = tread_path(tier, aisle, 0..tier.rows);
    tags.extend(std::iter::repeat(TAG_CONCRETE).take(path.len()));
    pts.extend(path);
    pts.extend([v2(sb, hb + back_parapet), v2(sb + WALL, hb + back_parapet), v2(sb + WALL, bottom(tier, sb + WALL))]);
    tags.extend([TAG_CONCRETE, TAG_CONCRETE, if tier.soffit_front.is_some() { TAG_SOFFIT } else { TAG_CONCRETE }]);
    clean(pts, tags)
}

pub fn vom_sections(tier: &Tier, r0: usize, k: usize) -> [Section; 2] {
    let (s, h) = (&tier.s, &tier.h);
    let (s0, sb, hb) = (tier.s0, tier.s_back, tier.h_last());
    let floor = h[r0];
    let mut lower = vec![
        v2(s0 - WALL, bottom(tier, s0 - WALL)),
        v2(s0 - WALL, h[0] + tier.parapet),
        v2(s0, h[0] + tier.parapet),
    ];
    let mut p = tread_path(tier, false, 0..r0 + 1);
    p.pop();
    lower.extend(p);
    lower.extend([v2(s[r0], floor), v2(sb + WALL, floor), v2(sb + WALL, bottom(tier, sb + WALL))]);
    let mut ltags = vec![TAG_FASCIA, TAG_FASCIA];
    ltags.extend(std::iter::repeat(TAG_CONCRETE).take(lower.len() - 3));
    ltags.push(TAG_SOFFIT);
    let ceil = floor + VOM_HEADROOM;
    let mut upper = vec![v2(s[r0 + k], ceil), v2(s[r0 + k], h[r0 + k])];
    upper.extend(tread_path(tier, false, r0 + k..tier.rows).into_iter().skip(1));
    upper.extend([v2(sb, hb + 1.1), v2(sb + WALL, hb + 1.1), v2(sb + WALL, ceil)]);
    let utags = vec![TAG_CONCRETE; upper.len()];
    [clean(lower, ltags), clean(upper, utags)]
}

/// Row where the vomitory opens and how many rows it removes.
pub fn vom_rows(tier: &Tier) -> Option<(usize, usize)> {
    if !tier.vomitories || tier.rows < 10 {
        return None;
    }
    let r0 = if tier.index == 0 { 2.max(tier.rows / 4) } else { 2 };
    for k in 2..tier.rows.saturating_sub(r0) {
        if tier.h[r0 + k] - tier.h[r0] >= VOM_HEADROOM + 0.3 {
            return Some((r0, k));
        }
    }
    None
}

pub fn block_section(lower: &Tier, upper: &Tier) -> Section {
    let sa = lower.s_back + WALL;
    let sb = (upper.s_back + WALL).max(sa + 3.0);
    let front = upper.s0 - WALL;
    let mut pts = vec![v2(sa, 0.0), v2(sb, 0.0)];
    let top_b = upper.soffit_z(sb.min(upper.s_back + WALL));
    pts.push(v2(sb, top_b));
    let sf = upper.soffit_front.unwrap_or(0.0);
    if front > sa {
        pts.extend([v2(front, upper.soffit_z(front)), v2(front, sf), v2(sa, sf)]);
    } else {
        pts.push(v2(sa, upper.soffit_z(sa)));
    }
    let n = pts.len();
    clean(pts, vec![TAG_BLOCK; n])
}

#[derive(Clone, Debug)]
pub struct Anchor {
    pub pos: DVec3,
    pub yaw: f64,
    pub tier: usize,
    pub row: usize,
    pub t: f64,
    pub pitch: f64,
    pub bay: usize,
    pub seat: usize,
    pub of: usize,
    pub run: usize,
    pub riser: f64,
}

#[derive(Clone, Debug)]
pub struct Socket {
    pub kind: &'static str,
    pub id: String,
    pub loc: DVec3,
    pub n: DVec2,
    pub props: Vec<(&'static str, f64)>,
}

impl Socket {
    pub fn prop(&self, k: &str) -> Option<f64> {
        self.props.iter().find(|p| p.0 == k).map(|p| p.1)
    }
    /// Rotation about Z in degrees, as the Python sockets write it.
    pub fn yaw_deg(&self) -> f64 {
        self.n.x.atan2(-self.n.y).to_degrees()
    }
}

pub type Polyline = (Vec<DVec3>, f64);

#[derive(Clone, Debug, Default)]
pub struct RunOut {
    pub mesh: MeshAcc,
    pub anchors: Vec<Anchor>,
    pub rails: Vec<Polyline>,
    pub nosings: Vec<(DVec3, DVec3)>,
    pub barriers: Vec<Vec<DVec3>>,
    pub sockets: Vec<Socket>,
    pub bay_list: Vec<(f64, f64)>,
}

/// Geometry, seat anchors, rails and sockets for one run. `run_index` is stored on anchors.
pub fn build_run(run: &Run, run_index: usize, tiers: &[Tier]) -> RunOut {
    let vom_every = 3;
    let mut o = RunOut::default();
    let mids: Vec<f64> = tiers.iter().map(|t| t.s0).chain(tiers.iter().map(|t| t.s_back)).collect();
    let s_mid = mids.iter().sum::<f64>() / mids.len() as f64;
    let bay_list = run.bays(s_mid);
    let m_mid = run.metric(s_mid);
    for (ti, tier) in tiers.iter().enumerate() {
        let full = tier_section(tier, false);
        let aisle = tier_section(tier, true);
        let vr = vom_rows(tier);
        let vom_sec = vr.map(|(r0, k)| vom_sections(tier, r0, k));
        let pitch_target = tier.seat_pitch;
        for (b, &(ta, tb)) in bay_list.iter().enumerate() {
            let ha = HALF_AISLE / m_mid;
            o.mesh.add_prism(run, ta, ta + ha, &aisle, TAG_CONCRETE);
            o.mesh.add_prism(run, tb - ha, tb, &aisle, TAG_CONCRETE);
            let width_mid = (tb - ta) * m_mid;
            let has_vom = vom_sec.is_some() && b % vom_every == 1 && width_mid >= VOM_WIDTH + 3.0;
            if has_vom {
                let (r0, k) = vr.unwrap();
                let tm = 0.5 * (ta + tb);
                let hv = 0.5 * VOM_WIDTH / run.metric(tier.s[r0]);
                o.mesh.add_prism(run, ta + ha, tm - hv, &full, TAG_CONCRETE);
                o.mesh.add_prism(run, tm + hv, tb - ha, &full, TAG_CONCRETE);
                for sec in vom_sec.as_ref().unwrap() {
                    o.mesh.add_prism(run, tm - hv, tm + hv, sec, TAG_CONCRETE);
                }
                o.rails.extend(vom_rails(run, tier, (r0, k), tm, hv));
            } else {
                o.mesh.add_prism(run, ta + ha, tb - ha, &full, TAG_CONCRETE);
            }
            if tier.kind == Kind::Seated {
                let vom = if has_vom { Some((vr.unwrap(), 0.5 * (ta + tb), VOM_WIDTH / 2.0)) } else { None };
                anchors(&mut o.anchors, run, run_index, tier, b, ta + ha, tb - ha, pitch_target, vom);
            } else {
                o.barriers.extend(barriers(run, tier, b, ta + ha, tb - ha));
            }
            o.nosings.extend(aisle_nosings(run, tier, ta, ha));
            o.nosings.extend(aisle_nosings(run, tier, tb - ha, ha));
            let (p, n) = run.frame(0.5 * (ta + tb));
            let q = p + n * (tier.s0 - WALL);
            o.sockets.push(Socket {
                kind: "fascia",
                id: format!("{}_T{}_B{}", run.name, tier.index, b),
                loc: DVec3::new(q.x, q.y, tier.h[0] + tier.parapet),
                n,
                props: vec![("width", round_to((tb - ta) * run.metric(tier.s0), 3)), ("tier", tier.index as f64)],
            });
        }
        let max_riser = tier.h.windows(2).map(|w| w[1] - w[0]).fold(f64::MIN, f64::max);
        if tier.h.len() > 1 && max_riser / tier.tread > 0.45 {
            for &(ta, _) in &bay_list[1..] {
                o.rails.push((row_line(run, ta, tier, 0.9, 1..tier.rows), 0.022));
            }
        }
        if ti == tiers.len() - 1 {
            for (b, &(ta, tb)) in bay_list.iter().enumerate() {
                let (p, n) = run.frame(0.5 * (ta + tb));
                let q = p + n * (tier.s_back + WALL);
                o.sockets.push(Socket {
                    kind: "roof_anchor",
                    id: format!("{}_B{}", run.name, b),
                    loc: DVec3::new(q.x, q.y, tier.h_last() + 1.1),
                    n,
                    props: vec![("width", round_to((tb - ta) * run.metric(tier.s_back), 3))],
                });
            }
        }
    }
    for w in tiers.windows(2) {
        let blk = block_section(&w[0], &w[1]);
        for &(ta, tb) in &bay_list {
            o.mesh.add_prism(run, ta, tb, &blk, TAG_CONCRETE);
        }
        rakers(&mut o.mesh, run, &w[1], &bay_list);
    }
    o.bay_list = bay_list;
    o
}

pub fn stand_depth(tiers: &[Tier]) -> f64 {
    let mut s = tiers.iter().map(|t| t.s_back + WALL).fold(f64::MIN, f64::max);
    for w in tiers.windows(2) {
        s = s.max(w[1].s_back + WALL).max(w[0].s_back + WALL + 3.0);
    }
    s
}

/// Convex hull (monotone chain) of points, as the Python does it (sorted unique tuples).
pub fn convex_hull(mut p: Vec<DVec2>) -> Vec<DVec2> {
    p.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: DVec2, a: DVec2, b: DVec2| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    let mut lo: Vec<DVec2> = Vec::new();
    let mut up: Vec<DVec2> = Vec::new();
    for &q in &p {
        while lo.len() >= 2 && cross(lo[lo.len() - 2], lo[lo.len() - 1], q) <= 0.0 {
            lo.pop();
        }
        lo.push(q);
    }
    for &q in p.iter().rev() {
        while up.len() >= 2 && cross(up[up.len() - 2], up[up.len() - 1], q) <= 0.0 {
            up.pop();
        }
        up.push(q);
    }
    lo.pop();
    up.pop();
    lo.extend(up);
    lo
}

pub fn section_hull(tiers: &[Tier]) -> Vec<DVec2> {
    let mut pts = Vec::new();
    for t in tiers {
        pts.extend(tier_section(t, false).pts);
    }
    for w in tiers.windows(2) {
        pts.extend(block_section(&w[0], &w[1]).pts);
    }
    convex_hull(pts)
}

#[allow(clippy::too_many_arguments)]
fn anchors(
    out: &mut Vec<Anchor>,
    run: &Run,
    run_index: usize,
    tier: &Tier,
    bay: usize,
    ta: f64,
    tb: f64,
    pitch_target: f64,
    vom: Option<((usize, usize), f64, f64)>,
) {
    let t_ = tier.tread;
    for i in 0..tier.rows {
        let s = tier.s[i] + t_;
        let m = run.metric(s);
        let mut segments = vec![(ta, tb)];
        if let Some(((r0, k), tm, half)) = vom {
            if r0 <= i && i < r0 + k {
                let hv = (half + 0.1) / m;
                segments = vec![(ta, tm - hv), (tm + hv, tb)];
            }
        }
        for (sa, sb) in segments {
            let avail = (sb - sa) * m;
            let n = floordiv(avail, pitch_target);
            if n <= 0.0 {
                continue;
            }
            let n = n as usize;
            let pitch = avail / n as f64;
            for j in 0..n {
                let t = sa + (j as f64 + 0.5) * (sb - sa) / n as f64;
                let (_, nrm) = run.frame(t);
                out.push(Anchor {
                    pos: run.world(t, s, tier.h[i]),
                    yaw: yaw_facing_pitch(nrm),
                    tier: tier.index,
                    row: i,
                    t,
                    pitch,
                    bay,
                    seat: j,
                    of: n,
                    run: run_index,
                    riser: if i + 1 < tier.rows { tier.h[i + 1] - tier.h[i] } else { 0.0 },
                });
            }
        }
    }
}

fn row_line(run: &Run, t: f64, tier: &Tier, height: f64, rows: std::ops::Range<usize>) -> Vec<DVec3> {
    rows.map(|i| run.world(t, tier.s[i] + 0.05, tier.h[i] + height)).collect()
}

fn vom_rails(run: &Run, tier: &Tier, (r0, k): (usize, usize), tm: f64, hv: f64) -> Vec<Polyline> {
    let mut rails = Vec::new();
    for side in [-1.0, 1.0] {
        rails.push((row_line(run, tm + side * hv, tier, 1.0, r0 + 1..r0 + k + 1), 0.024));
    }
    let s_front = tier.s[r0 + k] + 0.06;
    let z = tier.h[r0 + k] + 1.0;
    let ts = linspace(tm - hv, tm + hv, 8);
    rails.push((ts.iter().map(|&t| run.world(t, s_front, z)).collect(), 0.024));
    rails
}

fn aisle_nosings(run: &Run, tier: &Tier, ta: f64, ha: f64) -> Vec<(DVec3, DVec3)> {
    let pts = tread_path(tier, true, 0..tier.rows);
    let mut segs = Vec::new();
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        if (b.y - a.y).abs() < 1e-6 && b.x > a.x {
            segs.push((run.world(ta, a.x, a.y), run.world(ta + ha, a.x, a.y)));
        }
    }
    segs
}

fn barriers(run: &Run, tier: &Tier, bay: usize, ta: f64, tb: f64) -> Vec<Vec<DVec3>> {
    let mut out = Vec::new();
    let mut i = 2 + (bay % 2) * 2;
    while i < tier.rows {
        let s = tier.s[i] + 0.12;
        let z = tier.h[i] + 1.1;
        let m = run.metric(s);
        let n = 2usize.max(((tb - ta) * m / 1.0) as i64 as usize + 1);
        let ts = linspace(ta + 0.3 / m, tb - 0.3 / m, n);
        out.push(ts.iter().map(|&t| run.world(t, s, z)).collect());
        i += 5;
    }
    out
}

fn rakers(acc: &mut MeshAcc, run: &Run, tier: &Tier, bay_list: &[(f64, f64)]) {
    let (sa, sb) = (tier.s0 - WALL + 0.4, tier.s_back + WALL);
    let (za, zb) = (tier.soffit_z(sa), tier.soffit_z(sb));
    let depth = 0.9;
    let sec = Section::uniform(vec![v2(sa, za), v2(sb, zb), v2(sb, zb - depth), v2(sa + 1.5, za - 0.35)], TAG_SOFFIT);
    let w = 0.2 / run.metric(sa);
    for &(ta, _) in bay_list {
        acc.add_prism(run, ta - w, ta + w, &sec, TAG_CONCRETE);
    }
    let ta = bay_list[bay_list.len() - 1].1;
    acc.add_prism(run, ta - w, ta + w, &sec, TAG_CONCRETE);
}
