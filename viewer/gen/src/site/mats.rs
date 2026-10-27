//! Materials the training-ground generators use (port of kit/site/mats.py). Library materials by
//! name (tools/export_library.py), recoloured where a colour is given; grass surfaces carry their
//! turf for the renderer's procedural grass; fences and nets carry a cord pattern.

use crate::patterns::hex_rgb;
use crate::pitch::GrassSpec;
use crate::stadium::{Mat, NetSpec, Tex};
use glam::DVec2;

pub fn named(key: &str) -> Mat {
    Mat::named(key)
}

pub fn tint(key: &str, hex: &str) -> Mat {
    Mat::tint(key, hex)
}

pub fn glow(key: &str, hex: &str, strength: f64) -> Mat {
    Mat { key: key.into(), color: Some(hex_rgb(hex)), emission: Some((hex_rgb(hex), strength)), texture: None }
}

/// A lawn: kind "amenity" (site grass), "field" (farmland round the site), "rough" (unkempt,
/// long), "fine" (a bowling green), "berm" (sloped banks).
pub fn lawn(kind: &str, origin: DVec2, yaw: f64) -> Mat {
    let (pattern, width, strength, color, dry, wear, height) = match kind {
        "field" => ("none", 6.0, 0.0, "#486329", "#857c4c", 0.25, 70.0),
        "rough" => ("none", 6.0, 0.0, "#535f30", "#8f8457", 0.45, 110.0),
        "fine" => ("stripes", 2.4, 0.22, "#3f6a25", "#8a8247", 0.1, 8.0),
        "berm" => ("none", 6.0, 0.0, "#44602a", "#857c4c", 0.2, 45.0),
        "hay" => ("none", 6.0, 0.0, "#66703a", "#a3975e", 0.12, 140.0),
        "stubble" => ("stripes", 2.2, 0.3, "#9a8a4f", "#b8a66c", 0.05, 30.0),
        _ => ("stripes", 4.5, 0.03, "#43642a", "#7f7a48", 0.15, 35.0),
    };
    Mat {
        key: "Pitch_Grass".into(),
        color: None,
        emission: None,
        texture: Some(Tex::Grass(GrassSpec {
            pattern: pattern.into(),
            stripe_width: width,
            angle_deg: 0.0,
            strength,
            natural: true,
            color: color.into(),
            dry: dry.into(),
            wear,
            wetness: 0.0,
            height_mm: height,
            half: DVec2::new(500.0, 500.0),
            origin,
            yaw,
            wear_zones: false,
        })),
    }
}

/// Artificial turf outside the lines (cage aprons): same as an artificial pitch, no wear zones.
pub fn turf_3g(origin: DVec2, yaw: f64) -> Mat {
    Mat {
        key: "Pitch_Grass".into(),
        color: None,
        emission: None,
        texture: Some(Tex::Grass(GrassSpec {
            pattern: "stripes".into(),
            stripe_width: 4.0,
            angle_deg: 0.0,
            strength: 0.18,
            natural: false,
            color: "#46742c".into(),
            dry: "#5d7031".into(),
            wear: 0.1,
            wetness: 0.0,
            height_mm: 40.0,
            half: DVec2::new(500.0, 500.0),
            origin,
            yaw,
            wear_zones: false,
        })),
    }
}

/// A cord mesh card (chain link, ball-stop netting, welded mesh): UVs in metres.
pub fn net(mesh: &str, size_mm: u32, cord_mm: f64, color: &str) -> Mat {
    let c = hex_rgb(color) * 255.0;
    Mat {
        key: "TINT_net_card".into(),
        color: None,
        emission: None,
        texture: Some(Tex::Net(NetSpec {
            mesh: mesh.into(),
            mesh_size_mm: size_mm,
            cord_tenths_mm: (cord_mm * 10.0).round() as u32,
            pattern: "solid".into(),
            band_cells: 1,
            colors: vec![[c.x.round() as u8, c.y.round() as u8, c.z.round() as u8]],
            metric: true,
        })),
    }
}

pub fn chainlink() -> Mat {
    net("diamond", 55, 3.5, "#2f4a35")
}

pub fn ball_stop() -> Mat {
    net("square", 100, 3.0, "#1c1f1d")
}

pub fn welded_mesh() -> Mat {
    net("square", 75, 4.0, "#c9ccc8")
}
