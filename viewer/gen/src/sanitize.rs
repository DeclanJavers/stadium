//! Layout sanitising (port of kit/sanitize.py): unknown catalog ids, option names and colours fall
//! back to a default with a warning instead of failing the whole stadium. Runs before the build.

use crate::catalog::Catalog;
use serde_json::{json, Map, Value};

pub const MOW: [&str; 7] = ["none", "stripes", "stripes_long", "checkerboard", "diagonal", "diamond", "circles"];
pub const NET_PATTERNS: [&str; 3] = ["solid", "bands", "checker"];
pub const NET_MESHES: [&str; 2] = ["hex", "square"];
pub const SEAT_PATTERNS: [&str; 6] = ["solid", "tiers", "rows", "bays", "gradient", "random"];
pub const SCREEN_PLACEMENTS: [&str; 4] = ["ends", "stand_top", "corners", "center"];
pub const FLOODLIGHT_TYPES: [&str; 3] = ["corner_pylons", "roof", "poles"];

/// "#rrggbb"
pub fn is_hex(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() == 7 && s.starts_with('#') && s[1..].chars().all(|c| c.is_ascii_hexdigit()))
}

struct Ctx {
    warnings: Vec<String>,
}

impl Ctx {
    /// Replace `obj[key]` by `default` unless it is one of `ok` (missing keys are left alone).
    fn choice(&mut self, obj: &mut Map<String, Value>, what: &str, key: &str, ok: &dyn Fn(&str) -> bool, default: &str) {
        if let Some(v) = obj.get(key) {
            if !v.as_str().is_some_and(ok) {
                self.warnings.push(format!("unknown {what} {}; using {default}", show(v)));
                obj.insert(key.into(), json!(default));
            }
        }
    }

    fn color(&mut self, obj: &mut Map<String, Value>, what: &str, key: &str, default: &str) {
        if let Some(v) = obj.get(key) {
            if !is_hex(v) {
                self.warnings.push(format!("{what} {} is not a #rrggbb colour; using {default}", show(v)));
                obj.insert(key.into(), json!(default));
            }
        }
    }

    /// Drop invalid entries from a colour list (an emptied list is removed: the default applies).
    fn colors(&mut self, obj: &mut Map<String, Value>, what: &str, key: &str) {
        if let Some(Value::Array(a)) = obj.get(key) {
            let good: Vec<Value> = a.iter().filter(|c| is_hex(c)).cloned().collect();
            if good.len() != a.len() {
                self.warnings.push(format!("{what}: ignoring colours that are not #rrggbb"));
                if good.is_empty() {
                    obj.remove(key);
                } else {
                    obj.insert(key.into(), Value::Array(good));
                }
            }
        }
    }

    /// A `{"preset": ...}` (or plain string) reference to a catalog entry.
    fn preset(&mut self, holder: &mut Map<String, Value>, key: &str, what: &str, names: &[&String]) {
        let Some(v) = holder.get_mut(key) else { return };
        let name = match v {
            Value::String(s) => s.clone(),
            Value::Object(o) => match o.get("preset") {
                Some(Value::String(s)) => s.clone(),
                Some(other) => show(other),
                None => return,
            },
            _ => return,
        };
        if names.iter().any(|n| **n == name) {
            return;
        }
        let Some(first) = names.first() else { return };
        self.warnings.push(format!("unknown {what} preset {name:?}; using {first}"));
        match v {
            Value::Object(o) => {
                o.insert("preset".into(), json!(first));
            }
            _ => *v = json!(first),
        }
    }
}

fn show(v: &Value) -> String {
    match v {
        Value::String(s) => format!("{s:?}"),
        other => other.to_string(),
    }
}

fn obj_mut<'a>(v: &'a mut Value, key: &str) -> Option<&'a mut Map<String, Value>> {
    v.get_mut(key).and_then(|x| x.as_object_mut())
}

fn seat_colors(c: &mut Ctx, holder: &mut Map<String, Value>, what: &str) {
    if let Some(Value::Object(col)) = holder.get_mut("colors") {
        c.choice(col, &format!("{what} seat pattern"), "pattern", &|s| SEAT_PATTERNS.contains(&s), "solid");
        c.color(col, &format!("{what} seat colour"), "color", "#c8102e");
        c.colors(col, &format!("{what} seat colours"), "colors");
    }
}

/// A sanitised copy of the layout and the warnings for what was changed.
pub fn sanitize(layout: &Value, cat: &Catalog) -> (Value, Vec<String>) {
    let mut l = layout.clone();
    let mut c = Ctx { warnings: Vec::new() };
    if !l.is_object() {
        return (l, c.warnings);
    }
    if let Some(p) = obj_mut(&mut l, "pitch") {
        c.choice(p, "grass", "grass", &|s| cat.grass.contains_key(s), "perennial_ryegrass");
        c.color(p, "line colour", "line_color", "#f5f5f0");
        if let Some(Value::Object(m)) = p.get_mut("mow") {
            c.choice(m, "mowing pattern", "pattern", &|s| MOW.contains(&s), "stripes");
        }
    }
    if let Some(g) = obj_mut(&mut l, "goals") {
        c.choice(g, "goal style", "style", &|s| cat.goals.contains_key(s), "classic");
        c.choice(g, "net pattern", "net_pattern", &|s| NET_PATTERNS.contains(&s), "solid");
        c.choice(g, "net mesh", "mesh", &|s| NET_MESHES.contains(&s), "hex");
        c.color(g, "goal frame colour", "frame_color", "#f4f4f0");
        c.colors(g, "net colours", "net_colors");
    }
    if let Some(f) = obj_mut(&mut l, "corner_flags") {
        c.choice(f, "corner flag preset", "preset", &|s| cat.corner_flags.contains_key(s), "pro_spring");
    }
    let roof_names: Vec<&String> = cat.roofs.keys().collect();
    let facade_names: Vec<&String> = cat.facades.keys().collect();
    if let Some(s) = obj_mut(&mut l, "stands") {
        let seat_ok = |x: &str| cat.seats.contains_key(x);
        c.choice(s, "hospitality seat", "hospitality_seat", &seat_ok, "padded_vip");
        c.color(s, "concrete colour", "concrete", "#9d998f");
        c.color(s, "fascia colour", "fascia_color", "#1b1d20");
        c.preset(s, "roof", "roof", &roof_names);
        c.preset(s, "facade", "facade", &facade_names);
        seat_colors(&mut c, s, "stand");
        let mut profiles: Vec<String> = Vec::new();
        if let Some(Value::Object(sides)) = s.get_mut("sides") {
            let valid: Vec<String> = sides.values().filter_map(|v| v.get("profile").and_then(|p| p.as_str())).filter(|p| cat.profiles.contains_key(*p)).map(String::from).collect();
            let fallback = valid.first().cloned().or_else(|| cat.profile_order.first().cloned()).unwrap_or_default();
            for (name, sc) in sides.iter_mut() {
                let Some(sc) = sc.as_object_mut() else { continue };
                c.choice(sc, &format!("{name} stand profile"), "profile", &|p| cat.profiles.contains_key(p), &fallback);
                c.choice(sc, &format!("{name} seat"), "seat", &seat_ok, "tipup_classic");
                c.preset(sc, "roof", &format!("{name} roof"), &roof_names);
                c.preset(sc, "facade", &format!("{name} facade"), &facade_names);
                seat_colors(&mut c, sc, name);
                if let Some(p) = sc.get("profile").and_then(|p| p.as_str()) {
                    profiles.push(p.to_string());
                }
            }
        }
        if s.get("corners").and_then(|v| v.as_str()) == Some("curved") {
            let mut uniq = profiles.clone();
            uniq.sort();
            uniq.dedup();
            if profiles.len() < 4 || uniq.len() != 1 {
                c.warnings.push("curved corners need all four sides with one profile (a bowl); using open corners".into());
                s.insert("corners".into(), json!("open"));
            }
        }
    }
    if let Some(f) = obj_mut(&mut l, "floodlights") {
        if let Some(name) = f.get("preset").and_then(|v| v.as_str()) {
            if !cat.floodlights.contains_key(name) {
                c.warnings.push(format!("unknown floodlight preset {name:?}; using corner_pylons"));
                f.insert("preset".into(), json!("corner_pylons"));
            }
        }
        c.choice(f, "floodlight type", "type", &|s| FLOODLIGHT_TYPES.contains(&s), "corner_pylons");
        if f.get("preset").is_none() && f.get("type").is_none() {
            f.insert("type".into(), json!("corner_pylons"));
        }
    }
    if let Some(a) = obj_mut(&mut l, "ads") {
        if let Some(Value::Array(b)) = a.get("brands") {
            let good: Vec<Value> = b.iter().filter(|k| k.as_str().is_some_and(|k| cat.brands.contains_key(k))).cloned().collect();
            if good.len() != b.len() {
                let bad: Vec<String> = b.iter().filter(|k| !good.contains(k)).map(show).collect();
                c.warnings.push(format!("unknown brands {}; left out", bad.join(", ")));
                a.insert("brands".into(), Value::Array(good));
            }
        }
    }
    if let Some(sc) = obj_mut(&mut l, "screens") {
        c.choice(sc, "screen placement", "placement", &|s| SCREEN_PLACEMENTS.contains(&s), "ends");
    }
    (l, c.warnings)
}
