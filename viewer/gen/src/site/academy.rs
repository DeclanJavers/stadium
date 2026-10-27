//! Academy grading (port of kit/site/academy.py): which academy category a site's built facilities
//! reach. The requirements are modelled on the facility criteria of the Premier League's Elite
//! Player Performance Plan (Category 4, the entry level, up to Category 1), simplified to what the
//! kit builds; each category also needs everything below it.

use super::{option, Plot, SitePitch};
use crate::catalog::{num, text, Catalog};
use serde_json::{json, Value};
use std::collections::BTreeSet;

pub const LEVELS: [usize; 4] = [4, 3, 2, 1];

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Grade {
    /// the best category met (None: not yet Category 4)
    pub category: Option<usize>,
    /// the one above (None at Category 1)
    pub next: Option<usize>,
    /// what the next one still needs
    pub missing: Vec<String>,
}

/// The requirements per category, in LEVELS order: (category, [(label, met)]).
pub fn requirements(ps: &[Plot], cat: &Catalog, pitches: &[SitePitch]) -> Vec<(usize, Vec<(&'static str, bool)>)> {
    let built: Vec<(&str, String, usize, Value)> = ps
        .iter()
        .filter(|p| p.state() == "built")
        .map(|p| (p.id.as_str(), text(&p.cfg, "facility", "").to_string(), num(&p.cfg, "tier", 1.0) as usize, p.cfg.get("options").filter(|v| v.is_object()).cloned().unwrap_or(json!({}))))
        .collect();
    let has = |fac: &str, tier: usize| built.iter().any(|(_, f, t, _)| f == fac && *t >= tier);
    let modules = |fac: &str, tier: usize| -> Vec<Vec<String>> {
        built
            .iter()
            .filter(|(_, f, t, _)| f == fac && *t >= tier)
            .filter_map(|(_, f, t, o)| option(cat, f, *t, o, "modules").as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()))
            .collect()
    };
    let formats: BTreeSet<String> = built.iter().filter(|(_, f, _, _)| f == "youth_pitches").map(|(_, f, t, o)| option(cat, f, *t, o, "format").as_str().unwrap_or("9v9").to_string()).collect();
    let full_size = built.iter().filter(|(_, f, t, _)| f == "pitch" || f == "showpiece_pitch" || (f == "artificial_pitch" && *t >= 2)).count();
    let building = built.iter().filter(|(_, f, _, _)| f == "academy_building" || f == "training_building").map(|(_, _, t, _)| *t).max().unwrap_or(0);
    let art_ids: BTreeSet<&str> = built.iter().filter(|(_, f, _, _)| f == "artificial_pitch").map(|(id, _, _, _)| *id).collect();
    let lit = pitches.iter().filter(|sp| sp.lit).count();
    let lit_art = pitches.iter().any(|sp| sp.lit && art_ids.contains(sp.plot.as_str()));
    let fmt = |f: &str| formats.contains(f);
    vec![
        (4, vec![("Two full-size pitches", full_size >= 2), ("Changing rooms (an academy or training building)", building >= 1)]),
        (
            3,
            vec![
                ("Foundation-phase pitches (5v5 or 7v7)", fmt("5v5") || fmt("7v7")),
                ("9v9 pitches", fmt("9v9")),
                ("U13-U14 11v11 pitches", fmt("11v11_u13")),
                ("A floodlit pitch", lit >= 1),
                ("An academy block or better", building >= 2),
            ],
        ),
        (
            2,
            vec![
                ("Indoor training (an air dome or hall)", has("indoor_hall", 1)),
                ("A floodlit artificial pitch", lit_art),
                ("A goalkeeper area", has("goalkeeper_area", 1)),
                ("A showpiece match pitch", has("showpiece_pitch", 1)),
                ("An academy centre with classrooms and sports science", modules("academy_building", 3).iter().any(|m| m.iter().any(|x| x == "classrooms") && m.iter().any(|x| x == "sports_science"))),
            ],
        ),
        (
            1,
            vec![
                ("A full-size indoor hall", has("indoor_hall", 3)),
                ("A showpiece pitch with a covered stand", has("showpiece_pitch", 2)),
                ("A Category One centre with a scholars' lodge", modules("academy_building", 4).iter().any(|m| m.iter().any(|x| x == "boarding"))),
                ("Secure arrival (drop-off with controlled entry)", has("drop_off", 3)),
                ("Four floodlit pitches", lit >= 4),
                ("Three full-size pitches", full_size >= 3),
            ],
        ),
    ]
}

pub fn grade(ps: &[Plot], cat: &Catalog, pitches: &[SitePitch]) -> Grade {
    let mut category = None;
    for (level, req) in requirements(ps, cat, pitches) {
        if req.iter().all(|r| r.1) {
            category = Some(level);
        } else {
            return Grade { category, next: Some(level), missing: req.iter().filter(|r| !r.1).map(|r| r.0.to_string()).collect() };
        }
    }
    Grade { category, next: None, missing: vec![] }
}
