//! Demolition, cleared ground and construction sites (port of kit/site/construction.py).
//!
//! Demolition: the occupant half knocked down (occupants.rs), welded-mesh fencing round the plot,
//! an excavator and skips. Cleared: levelled topsoil and survey pegs. Construction: branded
//! hoardings, site cabins and a toilet, and the works: for a building its frame rising inside
//! scaffolding with a tower crane; for a pitch the earthworks with drainage; for a car park the
//! sub-base going down.

use super::facilities::{footprint, pitch_dims};
use super::kernel::{self, along, bx, quad, strip, Frame};
use super::occupants::{cover, mound};
use super::{mats, Out, Plot, COVER_Z};
use crate::catalog::Catalog;
use crate::mesh::MeshAcc;
use crate::prng::uniform;
use crate::stands::Polyline;
use glam::{DVec2, DVec3};
use std::f64::consts::{FRAC_PI_2, PI};

/// Panels (a prop every `pitch` metres) round the plot, inset, with an opening at the frontage.
fn perimeter(o: &mut Out, p: &Plot, key: &str, pitch: f64, inset: f64, opening: f64, tints: &[(&str, &str)]) {
    let (hw, hd) = (p.w / 2.0 - inset, p.d / 2.0 - inset);
    let corners = [DVec2::new(-hw, -hd), DVec2::new(hw, -hd), DVec2::new(hw, hd), DVec2::new(-hw, hd)];
    let mid = DVec2::new(p.front.x * hw, p.front.y * hd);
    let tang = DVec2::new(-p.front.y, p.front.x);
    let start = corners.iter().position(|c| (*c - mid).dot(tang) > 0.0 && (*c - mid).dot(p.front) > -0.1).unwrap_or(0);
    let mut ring = vec![mid + tang * (opening / 2.0)];
    for k in 0..4 {
        ring.push(corners[(start + k) % 4]);
    }
    ring.push(mid - tang * (opening / 2.0));
    for (q, d) in along(&ring, pitch, 0.5) {
        o.prop_at(&p.frame, key, q.x, q.y, COVER_Z, (-d.x).atan2(d.y) + FRAC_PI_2, tints);
    }
}

pub fn demolition(o: &mut Out, p: &Plot) {
    perimeter(o, p, "prop.heras_panel", 3.5, 1.5, 8.0, &[]);
    let inner = p.front * ((p.d.min(p.w)) / 2.0 - 9.0);
    o.prop_at(&p.frame, "prop.excavator", inner.x, inner.y, COVER_Z, uniform(p.seed + 800, 0) * 6.283, &[]);
    let tang = DVec2::new(-p.front.y, p.front.x);
    for i in 0..2 {
        let q = inner + tang * (8.0 + i as f64 * 4.0);
        o.prop_at(&p.frame, "prop.skip", q.x, q.y, COVER_Z, (-p.front.x).atan2(p.front.y), &[("TINT_skip", "#c9a227")]);
    }
}

pub fn cleared(o: &mut Out, p: &Plot) {
    cover(o, p, "ground", 1.0, mats::named("Site_Dirt"));
    for (i, (x, y)) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].iter().enumerate() {
        o.prop_at(&p.frame, "prop.survey_peg", x * (p.w / 2.0 - 3.0), y * (p.d / 2.0 - 3.0), COVER_Z, i as f64, &[]);
    }
    let q = p.front * ((if p.front.y.abs() > 0.5 { p.d } else { p.w }) / 2.0 - 2.5);
    o.prop_at(&p.frame, "prop.site_board", q.x, q.y, COVER_Z, (-p.front.x).atan2(p.front.y), &[("TINT_club", &o.club.primary.clone())]);
}

/// A concrete frame (slabs and columns) up to `levels` storeys, with scaffolding round it.
fn frame_and_scaffold(o: &mut Out, f: &Frame, name: &str, w: f64, d: f64, levels: usize, sh: f64) {
    let mut m = MeshAcc::new();
    let z0 = 0.45;
    bx(&mut m, -w / 2.0, -d / 2.0, 0.0, w / 2.0, d / 2.0, z0, 0);
    let (nx, ny) = ((w / 6.0).round().max(1.0) as usize, (d / 6.0).round().max(1.0) as usize);
    for l in 0..levels {
        let z = z0 + l as f64 * sh;
        for i in 0..=nx {
            for j in 0..=ny {
                let (x, y) = (-w / 2.0 + 0.25 + (w - 0.5) * i as f64 / nx as f64, -d / 2.0 + 0.25 + (d - 0.5) * j as f64 / ny as f64);
                bx(&mut m, x - 0.2, y - 0.2, z, x + 0.2, y + 0.2, z + sh - 0.3, 0);
            }
        }
        bx(&mut m, -w / 2.0, -d / 2.0, z + sh - 0.3, w / 2.0, d / 2.0, z + sh, 0);
    }
    o.part_at(f, &format!("{name}:frame"), m, vec![mats::named("Facade_Concrete")], 0.0);
    // scaffolding: standards every 2.5 m, 1.2 m off the slab edge, ledgers every 2 m, boards
    let top = z0 + levels as f64 * sh + 1.0;
    let (ax, ay) = (w / 2.0 + 1.2, d / 2.0 + 1.2);
    let ring = [DVec2::new(-ax, -ay), DVec2::new(ax, -ay), DVec2::new(ax, ay), DVec2::new(-ax, ay), DVec2::new(-ax, -ay)];
    let mut t: Vec<Polyline> = Vec::new();
    for (q, _) in along(&ring, 2.5, 0.0) {
        t.push((vec![DVec3::new(q.x, q.y, COVER_Z), DVec3::new(q.x, q.y, top)], 0.024));
    }
    let mut boards = MeshAcc::new();
    let mut z = 2.0;
    while z < top {
        t.push((ring.iter().map(|q| DVec3::new(q.x, q.y, z)).collect(), 0.024));
        t.push((ring.iter().map(|q| DVec3::new(q.x, q.y, z + 1.0)).collect(), 0.02));
        for i in 0..4 {
            let (a, b) = (ring[i], ring[i + 1]);
            let n = (b - a).normalize();
            let inw = DVec2::new(-n.y, n.x);
            let (a2, b2) = (a + inw * 0.95, b + inw * 0.95);
            quad(&mut boards, DVec3::new(a.x, a.y, z), DVec3::new(b.x, b.y, z), DVec3::new(b2.x, b2.y, z), DVec3::new(a2.x, a2.y, z), 0);
        }
        z += 2.0;
    }
    o.tubes_at(f, &format!("{name}:scaffold"), t, mats::named("Galvanised"), 5);
    o.part_at(f, &format!("{name}:boards"), boards, vec![mats::named("Site_Plywood")], 0.0);
}

pub fn site(o: &mut Out, cat: &Catalog, p: &Plot, fac: &str, tier: usize, _style: &str) {
    let club = o.club.clone();
    perimeter(o, p, "prop.hoarding_panel", 2.44, 1.2, 8.0, &[("TINT_hoarding", &club.primary)]);
    // site cabins and welfare by the gate
    let tang = DVec2::new(-p.front.y, p.front.x);
    let along_x = p.front.y.abs() > 0.5;
    let (pw, pd) = if along_x { (p.w, p.d) } else { (p.d, p.w) };
    let corner = p.front * (pd / 2.0 - 5.0) + tang * (pw / 2.0 - 9.0);
    let yaw = (-tang.x).atan2(tang.y);
    o.prop_at(&p.frame, "prop.site_cabin", corner.x, corner.y, COVER_Z, yaw, &[("TINT_cabin", "#d7d9d4")]);
    o.prop_at(&p.frame, "prop.site_cabin", corner.x, corner.y, COVER_Z + 2.62, yaw, &[("TINT_cabin", "#d7d9d4")]);
    let wc = corner - tang * 7.0;
    o.prop_at(&p.frame, "prop.portaloo", wc.x, wc.y, COVER_Z, yaw + PI, &[]);
    let pal = corner - p.front * 8.0;
    o.prop_at(&p.frame, "prop.pallet_stack", pal.x, pal.y, COVER_Z, 0.3, &[]);
    o.prop_at(&p.frame, "prop.pipe_stack", pal.x - tang.x * 4.0, pal.y - tang.y * 4.0, COVER_Z, yaw, &[]);
    match fac {
        "pitch" | "artificial_pitch" | "goalkeeper_area" | "youth_pitches" | "cages" | "showpiece_pitch" => {
            cover(o, p, "earthworks", 1.0, mats::named("Site_Dirt"));
            let (l, w) = if fac == "goalkeeper_area" { (34.0f64.min(p.w - 8.0), 26.0f64.min(p.d - 6.0)) } else { pitch_dims(p, 105.0, 68.0, 4.0, 3.0) };
            // herringbone drainage trenches into a spine drain
            let mut d = MeshAcc::new();
            strip(&mut d, &[DVec2::new(-l / 2.0, 0.0), DVec2::new(l / 2.0, 0.0)], 0.6, COVER_Z + 0.01, 0, None);
            let n = (l / 6.0) as usize;
            for i in 0..=n {
                let x = -l / 2.0 + l * i as f64 / n.max(1) as f64;
                for s in [-1.0, 1.0] {
                    strip(&mut d, &[DVec2::new(x, 0.0), DVec2::new(x + 4.0, s * w / 2.0)], 0.35, COVER_Z + 0.008, 0, None);
                }
            }
            o.part_at(&p.frame, &format!("{}:drains", p.id), d, vec![mats::named("Site_Gravel")], 0.0);
            let mut heaps = MeshAcc::new();
            for i in 0..4 {
                mound(&mut heaps, -l / 2.0 + 8.0 + i as f64 * 9.0, w / 2.0 + 2.0, 4.0, 3.0, 2.0, p.seed + 810 + i, 0);
            }
            o.part_at(&p.frame, &format!("{}:topsoil", p.id), heaps, vec![mats::named("Site_Dirt")], 75.0);
            o.prop_at(&p.frame, "prop.excavator", -l / 4.0, -w / 4.0, COVER_Z, 0.6, &[]);
            o.prop_at(&p.frame, "prop.dumper", l / 4.0, w / 6.0, COVER_Z, 2.2, &[]);
            o.prop_at(&p.frame, "prop.roller", 0.0, -w / 3.0, COVER_Z, FRAC_PI_2, &[]);
        }
        "car_park" => {
            cover(o, p, "subbase", 1.0, mats::named("Site_Gravel"));
            let mut k = MeshAcc::new();
            let r = kernel::rect(p.w - 2.0, p.d - 2.0, 0.0, 0.0);
            strip(&mut k, &[r[0], r[1], r[2]], 0.25, COVER_Z + 0.1, 0, None);
            o.part_at(&p.frame, &format!("{}:kerb", p.id), k, vec![mats::named("Site_Kerb")], 0.0);
            o.prop_at(&p.frame, "prop.roller", 0.0, 0.0, COVER_Z, 0.0, &[]);
            o.prop_at(&p.frame, "prop.dumper", p.w / 4.0, -p.d / 6.0, COVER_Z, 1.0, &[]);
        }
        _ => {
            cover(o, p, "ground", 1.0, mats::named("Site_Dirt"));
            if let Some((pl, storeys, sh)) = footprint(cat, p, fac, tier) {
                if fac == "indoor_hall" && tier == 1 {
                    // an air dome goes up in a day: a slab and the membrane bundled on it
                    let mut m = MeshAcc::new();
                    bx(&mut m, -pl.w / 2.0, -pl.d / 2.0, COVER_Z, pl.w / 2.0, pl.d / 2.0, COVER_Z + 0.4, 0);
                    bx(&mut m, -6.0, -2.0, COVER_Z + 0.4, 6.0, 2.0, COVER_Z + 2.2, 1);
                    o.part_at(&pl.f, &format!("{}:slab", p.id), m, vec![mats::named("Facade_Concrete"), mats::tint("TINT_roof_membrane", "#eeeeea")], 0.0);
                } else if fac == "indoor_hall" {
                    // steel portal frames going up, half erected
                    let mut t: Vec<Polyline> = Vec::new();
                    let n = (pl.w / 6.0).round() as usize;
                    let h = sh + 0.45;
                    for i in 0..=n / 2 {
                        let x = -pl.w / 2.0 + pl.w * i as f64 / n as f64;
                        let rise = 0.14 * pl.d;
                        let mut arc: Vec<DVec3> = vec![DVec3::new(x, -pl.d / 2.0, 0.0)];
                        for k in 0..=10 {
                            let y = -pl.d / 2.0 + pl.d * k as f64 / 10.0;
                            arc.push(DVec3::new(x, y, h + rise * (1.0 - (y / (pl.d / 2.0)).powi(2))));
                        }
                        arc.push(DVec3::new(x, pl.d / 2.0, 0.0));
                        t.push((arc, 0.25));
                    }
                    o.tubes_at(&pl.f, &format!("{}:portals", p.id), t, mats::tint("TINT_trim", "#d6d9d6"), 6);
                    let mut m = MeshAcc::new();
                    bx(&mut m, -pl.w / 2.0, -pl.d / 2.0, COVER_Z, pl.w / 2.0, pl.d / 2.0, COVER_Z + 0.3, 0);
                    o.part_at(&pl.f, &format!("{}:slab", p.id), m, vec![mats::named("Facade_Concrete")], 0.0);
                    let c = pl.f.pt(pl.w / 2.0 + 6.0, pl.d / 2.0 + 3.0);
                    o.prop("prop.tower_crane", crate::stadium::Placement::at(DVec3::new(c.x, c.y, COVER_Z), pl.f.yaw + PI * 1.2), &[]);
                } else {
                    let levels = if storeys > 1 { storeys - 1 } else { 1 };
                    frame_and_scaffold(o, &pl.f, &p.id, pl.w, pl.d, levels, sh);
                    let c = pl.f.pt(pl.w / 2.0 + 5.0, pl.d / 2.0 + 2.0);
                    o.prop("prop.tower_crane", crate::stadium::Placement::at(DVec3::new(c.x, c.y, COVER_Z), pl.f.yaw + PI * 1.25), &[]);
                }
            } else {
                // portacabins and containers just arrive: a gravel pad
                cover(o, p, "pad", 4.0, mats::named("Site_Gravel"));
            }
        }
    }
}
