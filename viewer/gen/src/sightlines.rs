//! Sightline solver (port of kit/stands/sightlines.py): every tread height from the C-value.
//!
//! R = (N + C)(D + T)/D - N, risers rounded up to 5 mm, capped at a 34 deg rake; upper tiers step
//! back until every row clears C_MIN.

use crate::catalog::Profile;
use crate::py::round_to;

pub const SEATED_EYE: f64 = 1.15;
pub const STANDING_EYE: f64 = 1.60;
pub const SEATED_EYE_FWD: f64 = 0.30;
pub const MAX_RAKE_DEG: f64 = 34.0;
pub const RISER_STEP: f64 = 0.005;
pub const WALL: f64 = 0.25;
pub const C_MIN: f64 = 0.06;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Seated,
    Standing,
}

impl Kind {
    pub fn parse(s: &str) -> Kind {
        if s == "standing" { Kind::Standing } else { Kind::Seated }
    }
    fn defaults(self) -> (f64, f64, f64, f64) {
        // tread, c_value, parapet, slab
        match self {
            Kind::Seated => (0.80, 0.09, 0.20, 0.75),
            Kind::Standing => (0.38, 0.06, 0.20, 0.75),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tier {
    pub index: usize,
    pub rows: usize,
    pub kind: Kind,
    pub tread: f64,
    pub c_target: f64,
    pub s0: f64,
    pub h: Vec<f64>,
    pub s: Vec<f64>,
    pub c_achieved: Vec<f64>,
    pub s_back: f64,
    pub setback: f64,
    pub parapet: f64,
    pub glass: f64,
    pub slab: f64,
    pub soffit_front: Option<f64>,
    pub soffit_slope: f64,
    pub seat_pitch: f64,
    pub vomitories: bool,
    pub hospitality: bool,
}

impl Tier {
    pub fn soffit_z(&self, s: f64) -> f64 {
        self.soffit_front.unwrap_or(0.0) + self.soffit_slope * (s - (self.s0 - WALL))
    }
    pub fn h_last(&self) -> f64 {
        *self.h.last().unwrap()
    }
}

fn eye(kind: Kind, s_back_of_tread: f64, tread: f64) -> (f64, f64) {
    match kind {
        Kind::Standing => (s_back_of_tread - tread / 2.0, STANDING_EYE),
        Kind::Seated => (s_back_of_tread - SEATED_EYE_FWD, SEATED_EYE),
    }
}

/// (h, c_achieved) for one tier.
pub fn solve_tier(rows: usize, s0: f64, h0: f64, d_focal: f64, kind: Kind, tread: f64, c: f64) -> (Vec<f64>, Vec<f64>) {
    let max_r = tread * MAX_RAKE_DEG.to_radians().tan();
    let mut h = vec![h0];
    let mut ca = Vec::new();
    for i in 0..rows.saturating_sub(1) {
        let (ex, eh) = eye(kind, s0 + (i + 1) as f64 * tread, tread);
        let d = d_focal + ex;
        let n = h[h.len() - 1] + eh;
        let mut r = (n + c) * (d + tread) / d - n;
        r = (r / RISER_STEP).ceil() * RISER_STEP;
        r = r.max(0.0).min(max_r);
        let last = h[h.len() - 1];
        h.push(last + r);
        ca.push((n + r) * d / (d + tread) - n);
    }
    (h, ca)
}

pub fn solve_profile(profile: &Profile, d_focal: f64) -> Vec<Tier> {
    let mut tiers: Vec<Tier> = Vec::new();
    for (k, spec) in profile.tiers.iter().enumerate() {
        let kind = Kind::parse(&spec.kind);
        let (dt, dc, dpar, dslab) = kind.defaults();
        let tread = match spec.tread {
            Some(t) if t != 0.0 => t,
            _ => dt,
        };
        let c = spec.c_value.unwrap_or(dc);
        let (mut s0, h0, soffit_front) = if k == 0 {
            (WALL, profile.front_height, None)
        } else {
            let prev = &tiers[tiers.len() - 1];
            let slab = spec.slab.unwrap_or(dslab);
            let sf = prev.h_last() + spec.clearance.unwrap_or(3.0);
            (prev.s_back - spec.overhang.unwrap_or(4.0), sf + slab, Some(sf))
        };
        let requested = s0;
        let c_min = spec.c_min.unwrap_or(C_MIN);
        let (mut h, mut ca) = solve_tier(spec.rows, s0, h0, d_focal, kind, tread, c);
        while k > 0 && !ca.is_empty() && min(&ca) < c_min - 1e-6 && s0 < requested + 80.0 {
            s0 += 0.25;
            let r = solve_tier(spec.rows, s0, h0, d_focal, kind, tread, c);
            h = r.0;
            ca = r.1;
        }
        let rows = spec.rows;
        let soffit_slope = if soffit_front.is_some() {
            (h[h.len() - 1] - h[0]) / ((rows as f64 - 1.0) * tread).max(1e-6)
        } else {
            0.0
        };
        tiers.push(Tier {
            index: k,
            rows,
            kind,
            tread,
            c_target: c,
            s0,
            s: (0..rows).map(|i| s0 + i as f64 * tread).collect(),
            h,
            c_achieved: ca,
            s_back: s0 + rows as f64 * tread,
            setback: round_to(s0 - requested, 2),
            parapet: spec.parapet.unwrap_or(if k == 0 { dpar } else { 0.85 }),
            glass: spec.glass.unwrap_or(if k > 0 { 0.3 } else { 0.0 }),
            slab: spec.slab.unwrap_or(dslab),
            soffit_front,
            soffit_slope,
            seat_pitch: spec.seat_pitch.unwrap_or(0.5),
            vomitories: spec.vomitories.unwrap_or(rows >= 12),
            hospitality: spec.hospitality.unwrap_or(false),
        });
    }
    tiers
}

fn min(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

#[derive(Clone, Debug)]
pub struct TierReport {
    pub tier: usize,
    pub rows: usize,
    pub standing: bool,
    pub front_tread_m: f64,
    pub back_tread_m: f64,
    pub max_rake_deg: f64,
    pub min_c_mm: Option<f64>,
    pub setback_m: f64,
}

pub fn report(tiers: &[Tier]) -> Vec<TierReport> {
    tiers
        .iter()
        .map(|t| {
            let risers: Vec<f64> = t.h.windows(2).map(|w| w[1] - w[0]).collect();
            let rake = risers.iter().cloned().fold(f64::MIN, f64::max);
            TierReport {
                tier: t.index,
                rows: t.rows,
                standing: t.kind == Kind::Standing,
                front_tread_m: t.h[0],
                back_tread_m: t.h_last(),
                max_rake_deg: if risers.is_empty() { 0.0 } else { (rake / t.tread).atan().to_degrees() },
                min_c_mm: if t.c_achieved.is_empty() { None } else { Some(1000.0 * min(&t.c_achieved)) },
                setback_m: t.setback,
            }
        })
        .collect()
}
