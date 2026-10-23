//! Stand runs around the pitch (port of the run classes in kit/stands/geometry.py).
//!
//! A run maps (t, s, z) to the world: t along the run (metres for straight runs, radians for arcs),
//! s outward from the stand front line, z up.

use crate::py::{linspace, round_half_even};
use glam::{DVec2, DVec3};
use std::f64::consts::PI;

pub const BAY_TARGET: f64 = 8.0;

#[derive(Clone, Debug)]
pub enum Shape {
    Straight { p0: DVec2, dir: DVec2, n: DVec2 },
    Arc { c: DVec2, r0: f64 },
}

#[derive(Clone, Debug)]
pub struct Run {
    pub name: String,
    pub t0: f64,
    pub t1: f64,
    pub shape: Shape,
}

impl Run {
    pub fn straight(name: &str, start: DVec2, direction: DVec2, length: f64) -> Run {
        let dir = direction / direction.length();
        Run {
            name: name.into(),
            t0: 0.0,
            t1: length,
            // outward = right-hand normal (runs go anticlockwise)
            shape: Shape::Straight { p0: start, dir, n: DVec2::new(dir.y, -dir.x) },
        }
    }

    pub fn arc(name: &str, centre: DVec2, radius: f64, a0: f64, a1: f64) -> Run {
        Run { name: name.into(), t0: a0, t1: a1, shape: Shape::Arc { c: centre, r0: radius } }
    }

    pub fn curved(&self) -> bool {
        matches!(self.shape, Shape::Arc { .. })
    }

    /// (front-line point, outward normal) at t.
    pub fn frame(&self, t: f64) -> (DVec2, DVec2) {
        match &self.shape {
            Shape::Straight { p0, dir, n } => (*p0 + *dir * t, *n),
            Shape::Arc { c, r0 } => {
                let n = DVec2::new(t.cos(), t.sin());
                (*c + n * *r0, n)
            }
        }
    }

    /// Metres per unit t at section position s.
    pub fn metric(&self, s: f64) -> f64 {
        match &self.shape {
            Shape::Straight { .. } => 1.0,
            Shape::Arc { r0, .. } => r0 + s,
        }
    }

    pub fn samples(&self, ta: f64, tb: f64, s_max: f64) -> Vec<f64> {
        match &self.shape {
            Shape::Straight { .. } => vec![ta, tb],
            Shape::Arc { r0, .. } => {
                let n = 2usize.max(((tb - ta).abs() * (r0 + s_max) / 1.0).ceil() as usize + 1);
                linspace(ta, tb, n)
            }
        }
    }

    pub fn world(&self, t: f64, s: f64, z: f64) -> DVec3 {
        let (p, n) = self.frame(t);
        let q = p + n * s;
        DVec3::new(q.x, q.y, z)
    }

    /// Length along the front line in metres.
    pub fn length(&self) -> f64 {
        (self.t1 - self.t0) * self.metric(0.0)
    }

    pub fn bays(&self, s_mid: f64) -> Vec<(f64, f64)> {
        let width = (self.t1 - self.t0) * self.metric(s_mid);
        let n = 1usize.max(round_half_even(width / BAY_TARGET) as i64 as usize);
        let e = linspace(self.t0, self.t1, n + 1);
        e.windows(2).map(|w| (w[0], w[1])).collect()
    }
}

/// Yaw that turns local +Y towards -n (the pitch).
pub fn yaw_facing_pitch(n: DVec2) -> f64 {
    n.x.atan2(-n.y)
}

pub const SIDES: [&str; 4] = ["south", "east", "north", "west"];

/// Front-line runs. a, b: half-sizes of the front-line rectangle. sides: (name, optional length)
/// for every side that has a stand.
pub fn perimeter_runs(a: f64, b: f64, sides: &[(String, Option<f64>)], curved: bool, corner_radius: f64) -> Vec<Run> {
    let r = if curved { corner_radius } else { 0.0 };
    let mut runs = Vec::new();
    for name in SIDES {
        let Some((_, len)) = sides.iter().find(|(n, _)| n == name) else { continue };
        let (mut start, d, mut length) = match name {
            "south" => (DVec2::new(-a + r, -b), DVec2::new(1.0, 0.0), 2.0 * a - 2.0 * r),
            "east" => (DVec2::new(a, -b + r), DVec2::new(0.0, 1.0), 2.0 * b - 2.0 * r),
            "north" => (DVec2::new(a - r, b), DVec2::new(-1.0, 0.0), 2.0 * a - 2.0 * r),
            _ => (DVec2::new(-a, b - r), DVec2::new(0.0, -1.0), 2.0 * b - 2.0 * r),
        };
        if !curved {
            if let Some(l) = len.filter(|l| *l != 0.0) {
                let full = length;
                length = full.min(l);
                start += d * (full - length) / 2.0;
            }
        }
        runs.push(Run::straight(name, start, d, length));
    }
    if curved {
        for (name, c, a0) in [
            ("SE", DVec2::new(a - r, -b + r), -PI / 2.0),
            ("NE", DVec2::new(a - r, b - r), 0.0),
            ("NW", DVec2::new(-a + r, b - r), PI / 2.0),
            ("SW", DVec2::new(-a + r, -b + r), PI),
        ] {
            runs.push(Run::arc(name, c, r, a0, a0 + PI / 2.0));
        }
    }
    runs
}
