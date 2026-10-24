//! Seat colour patterns and mosaics (port of kit/stands/patterns.py).

use crate::prng::uniform;
use crate::py::{ifloordiv, round_half_even};
use crate::stands::Anchor;
use glam::{DVec3, DVec4};
use serde_json::Value;

/// 5x7 pixel font; each row's 5 bits, most significant = leftmost.
const FONT: &[(char, [u8; 7])] = &[
    ('A', [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
    ('B', [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110]),
    ('C', [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111]),
    ('D', [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110]),
    ('E', [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111]),
    ('F', [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000]),
    ('G', [0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111]),
    ('H', [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
    ('I', [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111]),
    ('J', [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100]),
    ('K', [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001]),
    ('L', [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111]),
    ('M', [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001]),
    ('N', [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001]),
    ('O', [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
    ('P', [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000]),
    ('Q', [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101]),
    ('R', [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001]),
    ('S', [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110]),
    ('T', [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
    ('U', [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
    ('V', [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100]),
    ('W', [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010]),
    ('X', [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001]),
    ('Y', [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100]),
    ('Z', [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111]),
    ('0', [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110]),
    ('1', [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
    ('2', [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111]),
    ('3', [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110]),
    ('4', [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010]),
    ('5', [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110]),
    ('6', [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110]),
    ('7', [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000]),
    ('8', [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110]),
    ('9', [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100]),
    (' ', [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000]),
    ('-', [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000]),
    ('.', [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100]),
    ('&', [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101]),
    ('\'', [0b00100, 0b00100, 0b01000, 0b00000, 0b00000, 0b00000, 0b00000]),
    ('!', [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100]),
    ('?', [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100]),
    (':', [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000]),
    ('/', [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000]),
    ('+', [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000]),
];

fn glyph(ch: char) -> [u8; 7] {
    FONT.iter().find(|g| g.0 == ch).or_else(|| FONT.iter().find(|g| g.0 == ' ')).unwrap().1
}

pub fn hex_rgb(h: &str) -> DVec3 {
    let h = h.trim_start_matches('#');
    let c = |i: usize| u8::from_str_radix(h.get(i..i + 2).unwrap_or("00"), 16).unwrap_or(0) as f64 / 255.0;
    DVec3::new(c(0), c(2), c(4))
}

/// (7 rows of W columns) for the text, one blank column between glyphs.
pub fn text_bitmap(text: &str) -> Vec<Vec<bool>> {
    let mut rows = vec![Vec::new(); 7];
    for (k, ch) in text.to_uppercase().chars().enumerate() {
        let g = glyph(ch);
        for (r, row) in rows.iter_mut().enumerate() {
            if k > 0 {
                row.push(false);
            }
            for c in 0..5 {
                row.push(g[r] & (1 << (4 - c)) != 0);
            }
        }
    }
    rows
}

/// Accented capitals drawn with the plain letter.
const ACCENTS: &[(&str, char)] = &[("ÀÁÂÃÄÅ", 'A'), ("Ç", 'C'), ("ÈÉÊË", 'E'), ("ÌÍÎÏ", 'I'), ("Ñ", 'N'), ("ÒÓÔÕÖØ", 'O'), ("ÙÚÛÜ", 'U'), ("Ý", 'Y')];

/// Upper case in the seat font: accents dropped, characters it has no glyph for left blank.
fn plain_text(text: &str, warnings: &mut Vec<String>) -> String {
    let mut out = String::new();
    let mut missing: Vec<char> = Vec::new();
    for ch in text.to_uppercase().chars() {
        let c = ACCENTS.iter().find(|a| a.0.contains(ch)).map(|a| a.1).unwrap_or(ch);
        if FONT.iter().any(|g| g.0 == c) {
            out.push(c);
        } else {
            if !missing.contains(&ch) {
                missing.push(ch);
            }
            out.push(' ');
        }
    }
    if !missing.is_empty() {
        let list: Vec<String> = missing.iter().map(|c| c.to_string()).collect();
        warnings.push(format!("mosaic text {text:?}: the seat font has no {}; left blank", list.join(" ")));
    }
    out
}

fn text_width(s: &str) -> usize {
    (6 * s.chars().count()).saturating_sub(1)
}

/// The text as a bitmap at most `max_w` seats wide: on one line if it fits, else wrapped onto as
/// many lines as `max_h` rows hold (a blank row between lines, each centred), else shortened.
pub fn text_block(text: &str, max_w: usize, max_h: usize, warnings: &mut Vec<String>) -> Vec<Vec<bool>> {
    let t = plain_text(text, warnings);
    let max_lines = 1usize.max((max_h + 1) / 8);
    let mut lines = vec![t.clone()];
    if text_width(&t) > max_w && max_lines > 1 {
        let mut wrapped: Vec<String> = Vec::new();
        for word in t.split_whitespace() {
            match wrapped.last_mut() {
                Some(l) if text_width(l) + 6 + text_width(word) <= max_w => {
                    l.push(' ');
                    l.push_str(word);
                }
                _ => wrapped.push(word.to_string()),
            }
        }
        if wrapped.len() > 1 && wrapped.len() <= max_lines {
            lines = wrapped;
        }
    }
    for l in lines.iter_mut() {
        if text_width(l) > max_w {
            let fit: String = l.chars().take((max_w + 1) / 6).collect();
            // end on a whole word when there is one to end on
            let whole = l.chars().nth(fit.chars().count()) == Some(' ') || !fit.contains(' ');
            let cut = if whole { fit.trim_end().to_string() } else { fit[..fit.rfind(' ').unwrap()].trim_end().to_string() };
            warnings.push(format!("mosaic text {text:?} is {} seats wide but the tier has room for {max_w}; shortened to {cut:?}", text_width(l)));
            *l = cut;
        }
    }
    let w = lines.iter().map(|l| text_width(l)).max().unwrap_or(0);
    let mut rows = Vec::new();
    for (k, l) in lines.iter().enumerate() {
        if k > 0 {
            rows.push(vec![false; w]);
        }
        let pad = (w - text_width(l)) / 2;
        for r in text_bitmap(l) {
            let mut row = vec![false; pad];
            row.extend(r);
            row.resize(w, false);
            rows.push(row);
        }
    }
    rows
}

fn colors_of(spec: &Value) -> Vec<DVec3> {
    match spec.get("colors").and_then(|v| v.as_array()) {
        Some(a) => a.iter().filter_map(|c| c.as_str()).map(hex_rgb).collect(),
        None => vec![hex_rgb(spec.get("color").and_then(|v| v.as_str()).unwrap_or("#c8102e"))],
    }
}

/// Base pattern colours (sRGB) for a set of anchors.
pub fn base_colors(anchors: &[&Anchor], spec: &Value, tier_rows: &[usize], seed: u64) -> Result<Vec<DVec3>, String> {
    let pat = spec.get("pattern").and_then(|v| v.as_str()).unwrap_or("solid");
    let cols = colors_of(spec);
    if cols.is_empty() {
        return Err("seat pattern needs at least one colour".into());
    }
    let grow: Vec<usize> = anchors.iter().map(|a| a.row + tier_rows[..a.tier.min(tier_rows.len())].iter().sum::<usize>()).collect();
    let n = cols.len();
    Ok(match pat {
        "solid" => vec![cols[0]; anchors.len()],
        "tiers" => anchors.iter().map(|a| cols[a.tier % n]).collect(),
        "rows" => {
            let every = spec.get("every").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;
            grow.iter().map(|r| cols[(r / every) % n]).collect()
        }
        "bays" => anchors.iter().map(|a| cols[a.bay % n]).collect(),
        "gradient" => {
            let mx = (*grow.iter().max().unwrap_or(&0)).max(1) as f64;
            grow.iter().map(|g| {
                let f = *g as f64 / mx;
                cols[0] * (1.0 - f) + cols[n - 1] * f
            }).collect()
        }
        "random" => {
            let w: Vec<f64> = match spec.get("weights").and_then(|v| v.as_array()) {
                Some(a) => a.iter().map(|v| v.as_f64().unwrap_or(1.0)).collect(),
                None => vec![1.0; n],
            };
            let total: f64 = w.iter().sum();
            let mut cum = Vec::with_capacity(w.len());
            let mut acc = 0.0;
            for x in &w {
                acc += x / total;
                cum.push(acc);
            }
            (0..anchors.len())
                .map(|i| {
                    let u = uniform(seed, i as u64);
                    let k = cum.partition_point(|c| *c <= u).min(n - 1);
                    cols[k]
                })
                .collect()
        }
        p => return Err(format!("unknown seat pattern {p:?}")),
    })
}

/// An image for a mosaic: row 0 = top, sRGB, None = transparent.
pub struct Picture {
    pub w: usize,
    pub h: usize,
    pub px: Vec<Option<DVec3>>,
}

/// Overlay a text/image mosaic onto one run's tier, in place. Returns the number of seats painted.
pub fn apply_mosaic(
    colors: &mut [DVec3],
    anchors: &[Anchor],
    mosaic: &Value,
    run_index: usize,
    run_length: f64,
    tier_rows: &[usize],
    image: Option<&Picture>,
    warnings: &mut Vec<String>,
) -> usize {
    let tier = mosaic.get("tier").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let idx: Vec<usize> = (0..anchors.len()).filter(|&i| anchors[i].run == run_index && anchors[i].tier == tier).collect();
    if idx.is_empty() || tier >= tier_rows.len() {
        return 0;
    }
    let rows = tier_rows[tier] as i64;
    let (r_lo, r_hi) = match mosaic.get("rows").and_then(|v| v.as_array()) {
        Some(a) if a.len() == 2 => (a[0].as_i64().unwrap_or(1), a[1].as_i64().unwrap_or(rows - 1)),
        _ => (1, rows - 1),
    };
    let avail_rows = r_hi - r_lo + 1;
    let pitch = idx.iter().map(|&i| anchors[i].pitch).sum::<f64>() / idx.len() as f64;
    let span = mosaic.get("span").and_then(|v| v.as_array()).map(|a| (a[0].as_f64().unwrap_or(0.04), a[1].as_f64().unwrap_or(0.96))).unwrap_or((0.04, 0.96));
    let (t0, t1) = (span.0 * run_length, span.1 * run_length);
    let avail_seats = (t1 - t0) / pitch;
    let text_pic;
    let pic = match image {
        Some(p) => p,
        None => {
            let text = mosaic.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let bmp = text_block(text, avail_seats.max(0.0).floor() as usize, avail_rows.max(0) as usize, warnings);
            let fg = hex_rgb(mosaic.get("color").and_then(|v| v.as_str()).unwrap_or("#ffffff"));
            let bg = mosaic.get("background").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(hex_rgb);
            let w = bmp.first().map_or(0, |r| r.len());
            text_pic = Picture { w, h: bmp.len(), px: bmp.iter().flat_map(|r| r.iter().map(|&on| if on { Some(fg) } else { bg })).collect() };
            &text_pic
        }
    };
    let (hh, wd) = (pic.h as i64, pic.w as i64);
    if hh == 0 || wd == 0 {
        return 0;
    }
    let q = 1i64.max(ifloordiv(avail_rows, hh));
    let mut p = 1i64.max(round_half_even(q as f64 * 1.6) as i64);
    while (wd * p) as f64 > avail_seats && p > 1 {
        p -= 1;
    }
    let width_m = (wd * p) as f64 * pitch;
    let centre = 0.5 * (t0 + t1);
    let x_left = centre + width_m / 2.0;
    let top_row = r_lo + ifloordiv(avail_rows + hh * q, 2) - 1;
    let mut painted = 0;
    for i in idx {
        let a = &anchors[i];
        let c = ((x_left - a.t) / (p as f64 * pitch)).floor() as i64;
        let r = ifloordiv(top_row - a.row as i64, q);
        if (0..wd).contains(&c) && (0..hh).contains(&r) {
            if let Some(col) = pic.px[(r * wd + c) as usize] {
                colors[i] = col;
                painted += 1;
            }
        }
    }
    painted
}

/// Per-seat variation and UV fade; returns linear RGBA.
pub fn finish(colors: &[DVec3], fade: f64, seed: u64) -> Vec<DVec4> {
    let variation = 0.025;
    colors
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let k = (uniform(seed + 1, i as u64) - 0.5) * 2.0 * 1.7320508 * variation;
            let mut c = *c * (1.0 + k);
            if fade != 0.0 {
                let grey = (c.x + c.y + c.z) / 3.0;
                c = c * (1.0 - 0.45 * fade) + DVec3::splat((0.25 + 0.55 * grey) * 0.45 * fade + 0.08 * fade);
            }
            let c = c.clamp(DVec3::ZERO, DVec3::ONE);
            let lin = |v: f64| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
            DVec4::new(lin(c.x), lin(c.y), lin(c.z), 1.0)
        })
        .collect()
}

