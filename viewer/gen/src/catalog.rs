//! The kit's JSON catalogs (shared with the Python pipeline): profiles, roofs, facades, floodlights,
//! seats, brands, grass, goals, corner flags.

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Deserialize, Clone, Debug)]
pub struct TierSpec {
    pub rows: usize,
    #[serde(default = "seated")]
    pub kind: String,
    pub tread: Option<f64>,
    pub c_value: Option<f64>,
    pub overhang: Option<f64>,
    pub clearance: Option<f64>,
    pub slab: Option<f64>,
    pub c_min: Option<f64>,
    pub parapet: Option<f64>,
    pub glass: Option<f64>,
    pub seat_pitch: Option<f64>,
    pub vomitories: Option<bool>,
    pub hospitality: Option<bool>,
}

fn seated() -> String {
    "seated".into()
}
fn front_height() -> f64 {
    0.9
}

#[derive(Deserialize, Clone, Debug)]
pub struct Profile {
    #[serde(default)]
    pub label: String,
    #[serde(default = "front_height")]
    pub front_height: f64,
    pub tiers: Vec<TierSpec>,
}

/// Catalog entries in file order (the order the Python sees, e.g. the default brand rotation).
pub type Entries = serde_json::Map<String, Value>;

#[derive(Clone, Debug, Default)]
pub struct Catalog {
    pub profiles: BTreeMap<String, Profile>,
    pub profile_order: Vec<String>,
    pub roofs: Entries,
    pub facades: Entries,
    pub floodlights: Entries,
    pub seats: Entries,
    pub brands: Entries,
    pub grass: Entries,
    pub goals: Entries,
    pub corner_flags: Entries,
}

fn load_map(dir: &Path, name: &str) -> Result<Entries, String> {
    let p = dir.join(format!("{name}.json"));
    let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut v: Entries = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?;
    v.shift_remove("_doc");
    Ok(v)
}

impl Catalog {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let raw = load_map(dir, "stand_profiles")?;
        let profile_order = raw.keys().cloned().collect();
        let profiles = raw
            .into_iter()
            .map(|(k, v)| serde_json::from_value::<Profile>(v).map(|p| (k.clone(), p)).map_err(|e| format!("profile {k}: {e}")))
            .collect::<Result<_, _>>()?;
        Ok(Catalog {
            profiles,
            profile_order,
            roofs: load_map(dir, "roofs")?,
            facades: load_map(dir, "facades")?,
            floodlights: load_map(dir, "floodlights")?,
            seats: load_map(dir, "seats")?,
            brands: load_map(dir, "brands")?,
            grass: load_map(dir, "grass")?,
            goals: load_map(dir, "goals")?,
            corner_flags: load_map(dir, "corner_flags")?,
        })
    }

    /// Look for catalog/ next to (or above) a path, e.g. the executable or a layout file.
    pub fn find_dir(start: &Path) -> Option<std::path::PathBuf> {
        let mut p = Some(start);
        while let Some(d) = p {
            let c = d.join("catalog");
            if c.join("stand_profiles.json").exists() {
                return Some(c);
            }
            p = d.parent();
        }
        None
    }
}

/// Merge `over` into a copy of `base` (objects deep-merged, everything else replaced).
pub fn merge(base: &Value, over: &Value) -> Value {
    match (base, over) {
        (_, Value::Null) => base.clone(),
        (Value::Object(b), Value::Object(o)) => {
            let mut out = b.clone();
            for (k, v) in o {
                let merged = match out.get(k) {
                    Some(bv) if bv.is_object() && v.is_object() => merge(bv, v),
                    _ => v.clone(),
                };
                out.insert(k.clone(), merged);
            }
            Value::Object(out)
        }
        (_, o) => o.clone(),
    }
}

pub fn num(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
}

pub fn text<'a>(v: &'a Value, key: &str, default: &'a str) -> &'a str {
    v.get(key).and_then(|x| x.as_str()).unwrap_or(default)
}

/// A spec that may name a catalog preset: {"preset": "x", ...overrides}. None/false -> None.
pub fn resolve_preset(entries: &Entries, spec: &Value) -> Result<Option<Value>, String> {
    match spec {
        Value::Null | Value::Bool(false) => Ok(None),
        Value::Object(o) => match o.get("preset").and_then(|p| p.as_str()) {
            Some(name) => {
                let base = entries.get(name).ok_or_else(|| format!("unknown preset {name:?}; options: {:?}", entries.keys().collect::<Vec<_>>()))?;
                let mut over = o.clone();
                over.shift_remove("preset");
                Ok(Some(merge(base, &Value::Object(over))))
            }
            None => Ok(Some(spec.clone())),
        },
        _ => Ok(Some(spec.clone())),
    }
}
