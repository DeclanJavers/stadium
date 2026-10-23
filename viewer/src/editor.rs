//! The stadium editor's UI: every layout option as egui controls editing the stadium-layout/1
//! document directly, plus readouts (capacity, ad inventory, floodlight design, sightlines,
//! warnings). Any change bumps `revision`; the app rebuilds the stadium in the background.

use egui::{Color32, RichText, Ui};
use serde_json::{json, Value};
use stadium_gen::catalog::{resolve_preset, Catalog};
use stadium_gen::stadium::Report;
use std::path::PathBuf;
use std::sync::Arc;

pub const SIDES: [&str; 4] = ["south", "north", "east", "west"];
const MOW: [&str; 7] = ["none", "stripes", "stripes_long", "checkerboard", "diagonal", "diamond", "circles"];
const SEAT_PATTERNS: [&str; 6] = ["solid", "tiers", "rows", "bays", "gradient", "random"];

pub struct Editor {
    pub layout: Value,
    pub cat: Arc<Catalog>,
    pub path: Option<PathBuf>,
    pub layouts_dir: PathBuf,
    pub revision: u64,
    pub saved_revision: u64,
    pub report: Option<Report>,
    pub status: String,
    pub error: Option<String>,
    pub build_ms: (f32, f32, f32),
    /// sections switched off, kept so switching back on restores them
    stash: serde_json::Map<String, Value>,
    pub camera_request: Option<&'static str>,
    pub exposure: f32,
    pub fps: f32,
    /// open every section (headless UI screenshots)
    pub expand_all: bool,
}

fn hex_to_rgb(h: &str) -> [u8; 3] {
    let h = h.trim_start_matches('#');
    let c = |i: usize| u8::from_str_radix(h.get(i..i + 2).unwrap_or("80"), 16).unwrap_or(128);
    [c(0), c(2), c(4)]
}

fn rgb_to_hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// The object at `key` (created empty if missing or not an object).
fn obj<'a>(v: &'a mut Value, key: &str) -> &'a mut Value {
    if !v.get(key).is_some_and(|x| x.is_object()) {
        v[key] = json!({});
    }
    &mut v[key]
}

fn get_f(v: &Value, key: &str, d: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(d)
}

fn get_s<'a>(v: &'a Value, key: &str, d: &'a str) -> &'a str {
    v.get(key).and_then(|x| x.as_str()).unwrap_or(d)
}

fn label_row(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui) -> bool) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.add_sized([118.0, 18.0], egui::Label::new(label).truncate());
        changed = add(ui);
    });
    changed
}

fn num(ui: &mut Ui, label: &str, v: &mut Value, key: &str, d: f64, range: std::ops::RangeInclusive<f64>, step: f64, suffix: &str) -> bool {
    let orig = get_f(v, key, d);
    let mut x = orig;
    let changed = label_row(ui, label, |ui| {
        let mut s = egui::Slider::new(&mut x, range).step_by(step).suffix(suffix);
        if step >= 1.0 {
            s = s.integer();
        }
        ui.add(s).changed()
    }) && (x - orig).abs() > 1e-9;
    if changed {
        v[key] = if step >= 1.0 { json!(x.round() as i64) } else { json!((x / step).round() * step) };
    }
    changed
}

fn color(ui: &mut Ui, label: &str, v: &mut Value, key: &str, d: &str) -> bool {
    let mut c = hex_to_rgb(get_s(v, key, d));
    let changed = label_row(ui, label, |ui| {
        let r = ui.color_edit_button_srgb(&mut c).changed();
        ui.weak(rgb_to_hex(c));
        r
    });
    if changed {
        v[key] = json!(rgb_to_hex(c));
    }
    changed
}

fn check(ui: &mut Ui, label: &str, v: &mut Value, key: &str, d: bool) -> bool {
    let mut b = v.get(key).and_then(|x| x.as_bool()).unwrap_or(d);
    let changed = ui.checkbox(&mut b, label).changed();
    if changed {
        v[key] = json!(b);
    }
    changed
}

/// Combo over (value, label) pairs writing a string.
fn combo(ui: &mut Ui, id: &str, label: &str, v: &mut Value, key: &str, d: &str, options: &[(String, String)]) -> bool {
    let cur = get_s(v, key, d).to_string();
    let mut sel = cur.clone();
    let shown = options.iter().find(|o| o.0 == cur).map(|o| o.1.clone()).unwrap_or(cur.clone());
    label_row(ui, label, |ui| {
        egui::ComboBox::from_id_salt(id).selected_text(shown).width(190.0).show_ui(ui, |ui| {
            for (val, lab) in options {
                ui.selectable_value(&mut sel, val.clone(), lab);
            }
        });
        false
    });
    if sel != cur {
        v[key] = json!(sel);
        return true;
    }
    false
}

fn opts(list: &[&str]) -> Vec<(String, String)> {
    list.iter().map(|s| (s.to_string(), s.replace('_', " "))).collect()
}

fn cat_opts(entries: &serde_json::Map<String, Value>) -> Vec<(String, String)> {
    entries.iter().map(|(k, v)| (k.clone(), v.get("label").and_then(|l| l.as_str()).unwrap_or(k).to_string())).collect()
}

impl Editor {
    pub fn new(layout: Value, cat: Arc<Catalog>, path: Option<PathBuf>, layouts_dir: PathBuf) -> Self {
        Editor {
            layout,
            cat,
            path,
            layouts_dir,
            revision: 1,
            saved_revision: 1,
            report: None,
            status: String::new(),
            error: None,
            build_ms: (0.0, 0.0, 0.0),
            stash: Default::default(),
            camera_request: None,
            exposure: 1.0,
            fps: 0.0,
            expand_all: false,
        }
    }

    pub fn night(&self) -> bool {
        self.layout.get("lighting").and_then(|v| v.as_str()) == Some("night")
    }

    fn touch(&mut self, changed: bool) {
        if changed {
            self.revision += 1;
        }
    }

    /// Switch a whole section on/off, keeping its contents.
    fn section_toggle(&mut self, ui: &mut Ui, key: &str, label: &str, default: Value) -> bool {
        let on = self.layout.get(key).is_some_and(|v| v.is_object());
        let mut want = on;
        ui.checkbox(&mut want, label);
        if want != on {
            if want {
                let v = self.stash.shift_remove(key).unwrap_or(default);
                self.layout[key] = v;
            } else if let Some(v) = self.layout.as_object_mut().and_then(|o| o.shift_remove(key)) {
                self.stash.insert(key.into(), v);
                if key == "goals" || key == "corner_flags" {
                    self.layout[key] = Value::Null; // these default to "on" when the key is absent
                }
            }
            self.revision += 1;
        }
        want
    }

    // -- file ---------------------------------------------------------------------------------

    pub fn open(&mut self, path: PathBuf) {
        match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| e.to_string())) {
            Ok(v) if v.is_object() => {
                self.layout = v;
                self.path = Some(path.clone());
                self.stash.clear();
                self.revision += 1;
                self.saved_revision = self.revision;
                self.status = format!("opened {}", path.display());
            }
            Ok(_) => self.status = "not a layout object".into(),
            Err(e) => self.status = format!("open failed: {e}"),
        }
    }

    pub fn save(&mut self, path: PathBuf) {
        let mut v = self.layout.clone();
        v["schema"] = json!("stadium-layout/1");
        match serde_json::to_string_pretty(&v).map_err(|e| e.to_string()).and_then(|t| std::fs::write(&path, t + "\n").map_err(|e| e.to_string())) {
            Ok(()) => {
                self.path = Some(path.clone());
                self.saved_revision = self.revision;
                self.status = format!("saved {}", path.display());
            }
            Err(e) => self.status = format!("save failed: {e}"),
        }
    }

    fn templates(&self) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&self.layouts_dir).map(|d| d.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect()).unwrap_or_default();
        v.sort();
        v
    }

    pub fn top_bar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.menu_button("File", |ui| {
                ui.menu_button("New from template", |ui| {
                    for p in self.templates() {
                        if ui.button(p.file_stem().unwrap().to_string_lossy()).clicked() {
                            self.open(p);
                            self.path = None;
                            ui.close();
                        }
                    }
                });
                if ui.button("Open…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("stadium layout", &["json"]).set_directory(&self.layouts_dir).pick_file() {
                        self.open(p);
                    }
                    ui.close();
                }
                let can_save = self.path.is_some();
                if ui.add_enabled(can_save, egui::Button::new("Save")).clicked() {
                    if let Some(p) = self.path.clone() {
                        self.save(p);
                    }
                    ui.close();
                }
                if ui.button("Save as…").clicked() {
                    let name = format!("{}.json", get_s(&self.layout, "name", "stadium").to_lowercase().replace(|c: char| !c.is_alphanumeric(), "_"));
                    if let Some(p) = rfd::FileDialog::new().add_filter("stadium layout", &["json"]).set_directory(&self.layouts_dir).set_file_name(name).save_file() {
                        self.save(p);
                    }
                    ui.close();
                }
            });
            ui.separator();
            let mut name = get_s(&self.layout, "name", "Untitled stadium").to_string();
            if ui.add(egui::TextEdit::singleline(&mut name).desired_width(240.0)).changed() {
                self.layout["name"] = json!(name);
                self.saved_revision = 0;
            }
            if self.revision != self.saved_revision {
                ui.weak("● unsaved");
            }
            ui.separator();
            let mut night = self.night();
            let label = if night { "Night" } else { "Day" };
            if ui.toggle_value(&mut night, label).changed() {
                self.layout["lighting"] = json!(if night { "night" } else { "day" });
                self.revision += 1;
            }
            ui.separator();
            for (k, l) in [("broadcast", "Broadcast"), ("aerial", "Aerial"), ("pitch", "Pitch"), ("goal", "Goal")] {
                if ui.button(l).clicked() {
                    self.camera_request = Some(k);
                }
            }
            ui.separator();
            ui.add(egui::Slider::new(&mut self.exposure, 0.2..=1.5).text("exposure"));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(format!("{:.0} fps", self.fps));
                if !self.status.is_empty() {
                    ui.weak(&self.status);
                }
            });
        });
    }

    // -- left panel: the layout -----------------------------------------------------------------

    pub fn controls(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.spacing_mut().slider_width = 150.0;
            let all = self.expand_all;
            egui::CollapsingHeader::new(RichText::new("Pitch").strong()).default_open(true).show(ui, |ui| self.pitch_ui(ui));
            egui::CollapsingHeader::new(RichText::new("Goals & corner flags").strong()).default_open(all).show(ui, |ui| self.goals_ui(ui));
            egui::CollapsingHeader::new(RichText::new("Stands").strong()).default_open(true).show(ui, |ui| self.stands_ui(ui));
            egui::CollapsingHeader::new(RichText::new("Seat colours & mosaics").strong()).default_open(all).show(ui, |ui| self.colours_ui(ui));
            egui::CollapsingHeader::new(RichText::new("Roof & facade").strong()).default_open(all).show(ui, |ui| self.roof_facade_ui(ui));
            egui::CollapsingHeader::new(RichText::new("Floodlights").strong()).default_open(all).show(ui, |ui| self.lights_ui(ui));
            egui::CollapsingHeader::new(RichText::new("Advertising").strong()).default_open(all).show(ui, |ui| self.ads_ui(ui));
            egui::CollapsingHeader::new(RichText::new("Screens, dugouts, tunnel").strong()).default_open(all).show(ui, |ui| self.furniture_ui(ui));
        });
    }

    fn pitch_ui(&mut self, ui: &mut Ui) {
        let cat = self.cat.clone();
        let p = obj(&mut self.layout, "pitch");
        let mut c = false;
        c |= num(ui, "Length", p, "length", 105.0, 90.0..=120.0, 0.5, " m");
        c |= num(ui, "Width", p, "width", 68.0, 45.0..=90.0, 0.5, " m");
        c |= combo(ui, "grass", "Grass", p, "grass", "perennial_ryegrass", &cat_opts(&cat.grass));
        let g = cat.grass.get(get_s(p, "grass", "perennial_ryegrass")).cloned().unwrap_or(Value::Null);
        let (lo, hi, d) = (g["height_mm"]["min"].as_f64().unwrap_or(15.0), g["height_mm"]["max"].as_f64().unwrap_or(40.0), g["height_mm"]["default"].as_f64().unwrap_or(25.0));
        let cur = get_f(p, "height_mm", d);
        c |= num(ui, "Cut height", p, "height_mm", d, lo.min(cur)..=hi.max(cur), 1.0, " mm");
        let mow = obj(p, "mow");
        c |= combo(ui, "mow", "Mowing", mow, "pattern", "stripes", &opts(&MOW));
        c |= num(ui, "Stripe width", mow, "width", 5.25, 1.5..=12.0, 0.05, " m");
        c |= num(ui, "Stripe angle", mow, "angle", 0.0, -90.0..=90.0, 1.0, "°");
        c |= num(ui, "Wear", p, "wear", 0.25, 0.0..=1.0, 0.01, "");
        c |= num(ui, "Wetness", p, "wetness", 0.0, 0.0..=1.0, 0.01, "");
        c |= color(ui, "Line paint", p, "line_color", "#f5f5f0");
        c |= num(ui, "Hand-painted", p, "hand_painted", 0.0, 0.0..=1.0, 0.05, "");
        c |= num(ui, "Crown (camber)", p, "crown", 0.06, 0.0..=0.3, 0.005, " m");
        c |= check(ui, "Optional marks (10 yd)", p, "optional_marks", true);
        if let Some(gp) = g.get("gameplay") {
            ui.weak(format!(
                "{} · {} · roll {:.2} · bounce {:.2} · traction {:.2} · wear rate {:.2} · {}–{} °C",
                g["label"].as_str().unwrap_or(""),
                g["climate"].as_str().unwrap_or(""),
                gp["ball_roll"].as_f64().unwrap_or(1.0),
                gp["bounce"].as_f64().unwrap_or(1.0),
                gp["traction"].as_f64().unwrap_or(1.0),
                gp["wear_rate"].as_f64().unwrap_or(1.0),
                gp["temp_c"][0].as_f64().unwrap_or(0.0),
                gp["temp_c"][1].as_f64().unwrap_or(0.0)
            ));
        }
        self.touch(c);
    }

    fn goals_ui(&mut self, ui: &mut Ui) {
        let cat = self.cat.clone();
        let mut c = false;
        if self.section_toggle(ui, "goals", "Goals", json!({"style": "classic"})) {
            let g = obj(&mut self.layout, "goals");
            c |= combo(ui, "goal", "Style", g, "style", "classic", &cat_opts(&cat.goals));
            c |= color(ui, "Frame", g, "frame_color", "#f4f4f0");
            let mut net = g.get("net_colors").and_then(|v| v.as_array()).and_then(|a| a.first()).and_then(|v| v.as_str()).unwrap_or("#f7f7f5").to_string();
            let mut rgb = hex_to_rgb(&net);
            if label_row(ui, "Net", |ui| ui.color_edit_button_srgb(&mut rgb).changed()) {
                net = rgb_to_hex(rgb);
                let mut list: Vec<Value> = g.get("net_colors").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                if list.is_empty() {
                    list.push(json!(net));
                } else {
                    list[0] = json!(net);
                }
                g["net_colors"] = Value::Array(list);
                c = true;
            }
            ui.weak("Net cord pattern and mesh come from the library's net cards (Blender exports).");
        }
        if self.section_toggle(ui, "corner_flags", "Corner flags", json!({"preset": "pro_spring"})) {
            let f = obj(&mut self.layout, "corner_flags");
            c |= combo(ui, "flags", "Preset", f, "preset", "pro_spring", &cat_opts(&cat.corner_flags));
        }
        self.touch(c);
    }

    fn stands_ui(&mut self, ui: &mut Ui) {
        let cat = self.cat.clone();
        let mut c = false;
        let default_stands = json!({"front_gap": 3.0, "corners": "open", "sides": {"south": {"profile": "single_tier_20", "seat": "tipup_classic"}}});
        if !self.section_toggle(ui, "stands", "Stands", default_stands) {
            return;
        }
        let profiles: Vec<(String, String)> = cat.profile_order.iter().map(|k| (k.clone(), if cat.profiles[k].label.is_empty() { k.clone() } else { cat.profiles[k].label.clone() })).collect();
        let seats = cat_opts(&cat.seats);
        let s = obj(&mut self.layout, "stands");
        let was_curved = get_s(s, "corners", "open") == "curved";
        c |= combo(ui, "corners", "Corners", s, "corners", "open", &opts(&["open", "curved"]));
        let curved = get_s(s, "corners", "open") == "curved";
        if curved && !was_curved {
            // a bowl: every side gets the south (or first) stand's profile
            let sides = obj(s, "sides");
            let first = sides.as_object().and_then(|o| o.get("south").or_else(|| o.values().next())).and_then(|v| v.get("profile")).and_then(|v| v.as_str()).unwrap_or("three_tier_bowl").to_string();
            for side in SIDES {
                let sd = obj(sides, side);
                sd["profile"] = json!(first);
            }
        }
        if curved {
            c |= num(ui, "Corner radius", s, "corner_radius", 14.0, 6.0..=40.0, 0.5, " m");
        }
        c |= num(ui, "Front gap", s, "front_gap", 3.0, 1.0..=12.0, 0.1, " m");
        c |= combo(ui, "apron", "Apron", s, "apron", "turf", &opts(&["turf", "concrete", "track"]));
        c |= color(ui, "Concrete", s, "concrete", "#9d998f");
        c |= color(ui, "Fascia", s, "fascia_color", "#1b1d20");
        c |= combo(ui, "hosp", "Hospitality seat", s, "hospitality_seat", "padded_vip", &seats);
        c |= num(ui, "Seat UV fade", s, "seat_fade", 0.0, 0.0..=1.0, 0.05, "");
        ui.separator();
        let sides = obj(s, "sides");
        if curved {
            let cur = sides.get("south").and_then(|v| v.get("profile")).and_then(|v| v.as_str()).unwrap_or("three_tier_bowl").to_string();
            let mut prof = json!({"profile": cur.clone()});
            if combo(ui, "bowlprof", "Bowl profile", &mut prof, "profile", &cur, &profiles) {
                for side in SIDES {
                    obj(sides, side)["profile"] = prof["profile"].clone();
                }
                c = true;
            }
        }
        for side in SIDES {
            let has = sides.get(side).is_some_and(|v| v.get("profile").is_some());
            ui.horizontal(|ui| {
                let mut on = has;
                ui.add_enabled_ui(!curved, |ui| {
                    if ui.checkbox(&mut on, RichText::new(side).strong()).changed() {
                        if on {
                            sides[side] = json!({"profile": "single_tier_20", "seat": "tipup_classic"});
                        } else if let Some(o) = sides.as_object_mut() {
                            o.shift_remove(side);
                        }
                        c = true;
                    }
                });
            });
            if !sides.get(side).is_some_and(|v| v.get("profile").is_some()) {
                continue;
            }
            let sd = obj(sides, side);
            ui.indent(side, |ui| {
                if !curved {
                    c |= combo(ui, &format!("prof_{side}"), "Profile", sd, "profile", "single_tier_20", &profiles);
                    let full = 0.0;
                    let mut has_len = sd.get("length").is_some();
                    if ui.checkbox(&mut has_len, "Shorter than the side").changed() {
                        if has_len {
                            sd["length"] = json!(60.0);
                        } else if let Some(o) = sd.as_object_mut() {
                            o.shift_remove("length");
                        }
                        c = true;
                    }
                    if has_len {
                        c |= num(ui, "Length", sd, "length", 60.0, 10.0 + full..=140.0, 1.0, " m");
                    }
                }
                c |= combo(ui, &format!("seat_{side}"), "Seat", sd, "seat", "tipup_classic", &seats);
            });
        }
        self.touch(c);
    }

    fn colour_spec_ui(ui: &mut Ui, id: &str, spec: &mut Value) -> bool {
        let mut c = combo(ui, &format!("{id}_pat"), "Pattern", spec, "pattern", "solid", &opts(&SEAT_PATTERNS));
        let pat = get_s(spec, "pattern", "solid").to_string();
        if pat == "solid" {
            if spec.get("color").is_none() {
                let first = spec.get("colors").and_then(|v| v.as_array()).and_then(|a| a.first()).cloned().unwrap_or(json!("#c8102e"));
                spec["color"] = first;
            }
            c |= color(ui, "Colour", spec, "color", "#c8102e");
            return c;
        }
        if spec.get("colors").and_then(|v| v.as_array()).is_none_or(|a| a.is_empty()) {
            let first = spec.get("color").cloned().unwrap_or(json!("#c8102e"));
            spec["colors"] = json!([first, "#f2f2ee"]);
        }
        let mut list: Vec<String> = spec["colors"].as_array().unwrap().iter().filter_map(|v| v.as_str().map(String::from)).collect();
        let mut weights: Vec<f64> = spec.get("weights").and_then(|v| v.as_array()).map(|a| a.iter().map(|v| v.as_f64().unwrap_or(1.0)).collect()).unwrap_or_default();
        weights.resize(list.len(), 1.0);
        let mut remove = None;
        for (k, col) in list.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let mut rgb = hex_to_rgb(col);
                ui.add_sized([118.0, 18.0], egui::Label::new(format!("Colour {}", k + 1)));
                if ui.color_edit_button_srgb(&mut rgb).changed() {
                    *col = rgb_to_hex(rgb);
                    c = true;
                }
                if pat == "random" {
                    c |= ui.add(egui::DragValue::new(&mut weights[k]).range(0.0..=20.0).speed(0.1).prefix("weight ")).changed();
                }
                if k > 0 && ui.small_button("✕").clicked() {
                    remove = Some(k);
                }
            });
        }
        if let Some(k) = remove {
            list.remove(k);
            weights.remove(k);
            c = true;
        }
        if list.len() < 6 && ui.small_button("+ colour").clicked() {
            list.push("#f2f2ee".into());
            weights.push(1.0);
            c = true;
        }
        if c {
            spec["colors"] = json!(list);
            if pat == "random" {
                spec["weights"] = json!(weights);
            }
        }
        if pat == "rows" {
            c |= num(ui, "Rows per band", spec, "every", 1.0, 1.0..=10.0, 1.0, "");
        }
        c
    }

    fn colours_ui(&mut self, ui: &mut Ui) {
        if !self.layout.get("stands").is_some_and(|v| v.is_object()) {
            ui.weak("No stands.");
            return;
        }
        let mut c = false;
        let s = obj(&mut self.layout, "stands");
        ui.label(RichText::new("All stands").italics());
        c |= Self::colour_spec_ui(ui, "all", obj(s, "colors"));
        let side_names: Vec<String> = s.get("sides").and_then(|v| v.as_object()).map(|o| o.keys().cloned().collect()).unwrap_or_default();
        for side in &side_names {
            let sd = &mut s["sides"][side.as_str()];
            let mut own = sd.get("colors").is_some();
            if ui.checkbox(&mut own, format!("{side}: own colours")).changed() {
                if own {
                    sd["colors"] = json!({"pattern": "solid", "color": "#1d4ea8"});
                } else if let Some(o) = sd.as_object_mut() {
                    o.shift_remove("colors");
                }
                c = true;
            }
            if own {
                ui.indent(format!("col_{side}"), |ui| c |= Self::colour_spec_ui(ui, side, obj(sd, "colors")));
            }
        }
        ui.separator();
        ui.label(RichText::new("Mosaics (text spelled in seats)").italics());
        let mut list: Vec<Value> = s.get("mosaics").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let mut remove = None;
        for (k, mz) in list.iter_mut().enumerate() {
            ui.push_id(format!("mz{k}"), |ui| {
                ui.horizontal(|ui| {
                    let mut text = get_s(mz, "text", "").to_string();
                    if ui.add(egui::TextEdit::singleline(&mut text).desired_width(120.0)).changed() {
                        mz["text"] = json!(text.to_uppercase());
                        c = true;
                    }
                    if ui.small_button("✕").clicked() {
                        remove = Some(k);
                    }
                });
                c |= combo(ui, &format!("mzrun{k}"), "Stand", mz, "run", "north", &opts(&side_names.iter().map(|s| s.as_str()).collect::<Vec<_>>()));
                c |= num(ui, "Tier", mz, "tier", 0.0, 0.0..=3.0, 1.0, "");
                c |= color(ui, "Letters", mz, "color", "#ffffff");
                let mut bg = mz.get("background").is_some();
                if ui.checkbox(&mut bg, "Background colour").changed() {
                    if bg {
                        mz["background"] = json!("#111111");
                    } else if let Some(o) = mz.as_object_mut() {
                        o.shift_remove("background");
                    }
                    c = true;
                }
                if bg {
                    c |= color(ui, "Background", mz, "background", "#111111");
                }
            });
            ui.separator();
        }
        if let Some(k) = remove {
            list.remove(k);
            c = true;
        }
        if ui.button("+ mosaic").clicked() {
            list.push(json!({"run": side_names.first().cloned().unwrap_or("north".into()), "tier": 0, "text": "KIT FC", "color": "#f4f4f0"}));
            c = true;
        }
        if c {
            s["mosaics"] = Value::Array(list);
        }
        self.touch(c);
    }

    fn roof_or_facade(ui: &mut Ui, id: &str, label: &str, slot: &mut Value, key: &str, entries: &serde_json::Map<String, Value>, allow_inherit: bool, colours: &[(&str, &str, &str)]) -> bool {
        let mut c = false;
        let cur = match slot.get(key) {
            None => "inherit".to_string(),
            Some(Value::Object(o)) => o.get("preset").and_then(|p| p.as_str()).unwrap_or("custom").to_string(),
            _ => "none".to_string(),
        };
        let mut options: Vec<(String, String)> = Vec::new();
        if allow_inherit {
            options.push(("inherit".into(), "same as all stands".into()));
        }
        options.push(("none".into(), "none".into()));
        options.extend(cat_opts(entries));
        let mut sel = json!({"v": cur.clone()});
        if combo(ui, id, label, &mut sel, "v", &cur, &options) {
            let v = sel["v"].as_str().unwrap().to_string();
            match v.as_str() {
                "inherit" => {
                    if let Some(o) = slot.as_object_mut() {
                        o.shift_remove(key);
                    }
                }
                "none" => slot[key] = json!(false),
                p => slot[key] = json!({"preset": p}),
            }
            c = true;
        }
        if let Some(spec) = slot.get_mut(key).filter(|v| v.is_object()) {
            let resolved = resolve_preset(entries, spec).ok().flatten().unwrap_or(Value::Null);
            for (k, lab, d) in colours {
                if resolved.get("type").and_then(|t| t.as_str()) == Some("etfe") || *k != "glow" {
                    let dflt = get_s(&resolved, k, d).to_string();
                    c |= color(ui, lab, spec, k, &dflt);
                }
            }
            if get_s(&resolved, "type", "") == "goalpost" {
                c |= num(ui, "Post every", spec, "post_every", get_f(&resolved, "post_every", 3.0), 1.0..=6.0, 1.0, " bays");
            }
            if get_s(&resolved, "type", "") == "cantilever" {
                c |= num(ui, "Rise", spec, "rise_deg", get_f(&resolved, "rise_deg", 5.0), 0.0..=15.0, 0.5, "°");
                c |= num(ui, "Translucent edge", spec, "translucent", get_f(&resolved, "translucent", 0.25), 0.0..=1.0, 0.05, "");
            }
        }
        c
    }

    fn roof_facade_ui(&mut self, ui: &mut Ui) {
        if !self.layout.get("stands").is_some_and(|v| v.is_object()) {
            ui.weak("No stands.");
            return;
        }
        let cat = self.cat.clone();
        let roof_cols = [("color", "Sheeting", "#e3e4e2"), ("fascia_color", "Roof fascia", "#1b1d20"), ("steel_color", "Steelwork", "#eceeee")];
        let fac_cols = [("color", "Main colour", "#b8b3a8"), ("accent", "Accent", "#b8b3a8"), ("glow", "Night glow", "#d0102b")];
        let mut c = false;
        let s = obj(&mut self.layout, "stands");
        ui.label(RichText::new("All stands").italics());
        c |= Self::roof_or_facade(ui, "roof_all", "Roof", s, "roof", &cat.roofs, false, &roof_cols);
        c |= Self::roof_or_facade(ui, "fac_all", "Facade", s, "facade", &cat.facades, false, &fac_cols);
        let side_names: Vec<String> = s.get("sides").and_then(|v| v.as_object()).map(|o| o.keys().cloned().collect()).unwrap_or_default();
        let curved = get_s(s, "corners", "open") == "curved";
        if curved {
            ui.weak("Bowls share one roof and facade all round.");
        } else {
            for side in side_names {
                ui.separator();
                ui.label(RichText::new(&side).italics());
                let sd = &mut s["sides"][side.as_str()];
                c |= Self::roof_or_facade(ui, &format!("roof_{side}"), "Roof", sd, "roof", &cat.roofs, true, &roof_cols);
                c |= Self::roof_or_facade(ui, &format!("fac_{side}"), "Facade", sd, "facade", &cat.facades, true, &fac_cols);
            }
        }
        self.touch(c);
    }

    fn lights_ui(&mut self, ui: &mut Ui) {
        let cat = self.cat.clone();
        let mut c = false;
        if self.section_toggle(ui, "floodlights", "Floodlights", json!({"preset": "corner_pylons"})) {
            let f = obj(&mut self.layout, "floodlights");
            c |= combo(ui, "fl", "Installation", f, "preset", "corner_pylons", &cat_opts(&cat.floodlights));
            let resolved = resolve_preset(&cat.floodlights, f).ok().flatten().unwrap_or(Value::Null);
            c |= num(ui, "Target", f, "lux", get_f(&resolved, "lux", 1200.0), 50.0..=3000.0, 10.0, " lux");
            c |= num(ui, "Colour temp.", f, "kelvin", get_f(&resolved, "kelvin", 5000.0), 3000.0..=6500.0, 100.0, " K");
            if get_s(&resolved, "type", "") == "poles" {
                c |= num(ui, "Poles per side", f, "per_side", get_f(&resolved, "per_side", 3.0), 1.0..=6.0, 1.0, "");
                c |= num(ui, "Pole height", f, "height", get_f(&resolved, "height", 16.0), 8.0..=30.0, 0.5, " m");
                c |= num(ui, "Lamps per pole", f, "lamps", get_f(&resolved, "lamps", 3.0), 1.0..=8.0, 1.0, "");
            }
            if get_s(&resolved, "type", "") == "roof" {
                c |= num(ui, "Spacing", f, "spacing", get_f(&resolved, "spacing", 2.4), 1.0..=8.0, 0.1, " m");
            }
            ui.weak("Lights switch on with Night (top bar).");
        }
        self.touch(c);
    }

    fn ads_ui(&mut self, ui: &mut Ui) {
        let cat = self.cat.clone();
        let mut c = false;
        if !self.section_toggle(ui, "ads", "Advertising", json!({"perimeter": {"type": "led"}})) {
            return;
        }
        let a = obj(&mut self.layout, "ads");
        let per = obj(a, "perimeter");
        c |= combo(ui, "per", "Perimeter boards", per, "type", "led", &opts(&["led", "static", "none"]));
        let mut carpets = a.get("carpets").is_some_and(|v| v.as_bool() != Some(false) && !v.is_null());
        if ui.checkbox(&mut carpets, "3D carpets behind the goals").changed() {
            a["carpets"] = json!(carpets);
            c = true;
        }
        let mut rib = a.get("ribbons").is_some_and(|v| v.is_object());
        if ui.checkbox(&mut rib, "LED ribbons on upper-tier fascias").changed() {
            if rib {
                a["ribbons"] = json!({"tiers": [1, 2, 3]});
            } else if let Some(o) = a.as_object_mut() {
                o.shift_remove("ribbons");
            }
            c = true;
        }
        ui.label("Brands (rotation order):");
        let mut chosen: Vec<String> = a.get("brands").and_then(|v| v.as_array()).map(|x| x.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_else(|| cat.brands.iter().filter(|(_, b)| b.get("tier").and_then(|t| t.as_i64()).unwrap_or(1) > 0).map(|(k, _)| k.clone()).collect());
        let mut bc = false;
        egui::Grid::new("brands").num_columns(2).show(ui, |ui| {
            for (k, (key, b)) in cat.brands.iter().enumerate() {
                let mut on = chosen.contains(key);
                let tier = b.get("tier").and_then(|t| t.as_i64()).unwrap_or(1);
                let name = b.get("name").and_then(|v| v.as_str()).unwrap_or(key);
                if ui.checkbox(&mut on, format!("{name}{}", if tier == 0 { " (local)" } else { "" })).changed() {
                    if on {
                        chosen.push(key.clone());
                    } else {
                        chosen.retain(|x| x != key);
                    }
                    bc = true;
                }
                if k % 2 == 1 {
                    ui.end_row();
                }
            }
        });
        if bc && !chosen.is_empty() {
            a["brands"] = json!(chosen);
            c = true;
        }
        self.touch(c);
    }

    fn furniture_ui(&mut self, ui: &mut Ui) {
        let cat = self.cat.clone();
        let mut c = false;
        if self.section_toggle(ui, "screens", "Big screens", json!({"placement": "ends", "width": 18, "height": 8})) {
            let s = obj(&mut self.layout, "screens");
            c |= combo(ui, "scr", "Placement", s, "placement", "ends", &opts(&["ends", "corners", "center", "stand_top"]));
            c |= num(ui, "Width", s, "width", 18.0, 6.0..=40.0, 0.5, " m");
            c |= num(ui, "Height", s, "height", 8.0, 3.0..=16.0, 0.5, " m");
        }
        ui.separator();
        if self.section_toggle(ui, "dugouts", "Dugouts", json!({"style": "shell", "seats": 14})) {
            let d = obj(&mut self.layout, "dugouts");
            c |= combo(ui, "dug", "Style", d, "style", "shell", &opts(&["shell", "box", "shelter"]));
            c |= num(ui, "Seats each", d, "seats", 14.0, 4.0..=30.0, 1.0, "");
            let dflt = if get_s(d, "style", "shell") == "shelter" { "bench" } else { "padded_vip" };
            c |= combo(ui, "dugseat", "Seat", d, "seat", dflt, &cat_opts(&cat.seats));
            c |= color(ui, "Colour", d, "color", "#c8102e");
        }
        ui.separator();
        if self.section_toggle(ui, "tunnel", "Players' tunnel", json!({"style": "telescopic"})) {
            let t = obj(&mut self.layout, "tunnel");
            c |= combo(ui, "tun", "Style", t, "style", "telescopic", &opts(&["telescopic", "portal"]));
            c |= color(ui, "Colour", t, "color", "#c8102e");
        }
        self.touch(c);
    }

    // -- right panel: readouts ---------------------------------------------------------------------

    pub fn readouts(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            if let Some(e) = &self.error {
                ui.colored_label(Color32::from_rgb(255, 120, 100), format!("⚠ {e}"));
                ui.separator();
            }
            let Some(r) = &self.report else {
                ui.weak("building…");
                return;
            };
            ui.heading(format!("{} spectators", fmt_int(r.capacity.total)));
            ui.label(format!("{} seated · {} standing", fmt_int(r.capacity.seated), fmt_int(r.capacity.standing)));
            for (s, n) in &r.capacity.by_style {
                let label = self.cat.seats.get(s).and_then(|v| v.get("label")).and_then(|v| v.as_str()).unwrap_or(s);
                ui.weak(format!("  {label}: {}", fmt_int(*n)));
            }
            if r.footprint.0 > 0.0 {
                ui.weak(format!("Stand front line {:.0} × {:.0} m", 2.0 * r.footprint.0, 2.0 * r.footprint.1));
            }
            if !r.warnings.is_empty() {
                ui.separator();
                for w in &r.warnings {
                    ui.colored_label(Color32::from_rgb(240, 190, 80), format!("⚠ {w}"));
                }
            }
            ui.separator();
            egui::CollapsingHeader::new("Sightlines").default_open(self.expand_all).show(ui, |ui| {
                egui::Grid::new("tiers").striped(true).show(ui, |ui| {
                    ui.strong("stand");
                    ui.strong("tier");
                    ui.strong("rows");
                    ui.strong("rake");
                    ui.strong("min C");
                    ui.end_row();
                    for (run, tiers) in &r.tiers {
                        if run.len() <= 2 && run != "NE" {
                            continue; // corners repeat the sides
                        }
                        for t in tiers {
                            ui.label(run);
                            ui.label(format!("{}{}", t.tier, if t.standing { " (standing)" } else { "" }));
                            ui.label(format!("{}", t.rows));
                            ui.label(format!("{:.1}°", t.max_rake_deg));
                            ui.label(t.min_c_mm.map(|c| format!("{c:.0} mm")).unwrap_or("-".into()));
                            ui.end_row();
                        }
                    }
                });
                ui.weak("Green Guide: C ≥ 90 mm good, ≥ 60 mm acceptable; rake ≤ 34°.");
            });
            if !r.roofs.is_empty() {
                egui::CollapsingHeader::new("Roofs").show(ui, |ui| {
                    for (run, s, h, cl) in &r.roofs {
                        ui.label(format!("{run}: leading edge raised {s:.2} m for sightlines, {h:.2} m for headroom; clears the view by {cl:.1} m"));
                    }
                });
            }
            ui.separator();
            ui.strong("Floodlights");
            match &r.lights {
                Some(d) if !d.fixtures.is_empty() => {
                    ui.label(format!("{} fixtures ({}), beam {:.0}°", d.fixtures.len(), d.kind.replace('_', " "), d.beam));
                    ui.label(format!("{:.0} lux average (target {:.0})", d.lux_avg, d.lux_target));
                    let u1_ok = d.u1 >= 0.7;
                    ui.colored_label(if u1_ok { Color32::from_rgb(120, 220, 120) } else { Color32::from_rgb(240, 190, 80) }, format!("uniformity U1 {:.2} · U2 {:.2} (UEFA elite: U1 ≥ 0.7, U2 ≥ 0.8)", d.u1, d.u2));
                }
                _ => {
                    ui.weak("none");
                }
            }
            ui.separator();
            ui.strong("Ad inventory");
            if r.ad_totals.is_empty() {
                ui.weak("none");
            }
            let mut total_w = 0.0;
            for (k, t) in &r.ad_totals {
                ui.label(format!("{}: {} slots, {:.0} m², TV-weighted {:.0} m²", k.replace('_', " "), t.slots, t.area_m2, t.tv_weighted_m2));
                total_w += t.tv_weighted_m2;
            }
            if total_w > 0.0 {
                ui.weak(format!("TV-weighted total {total_w:.0} m² (input for the revenue model)"));
            }
            if r.screens > 0 {
                ui.label(format!("{} screen faces", r.screens));
            }
            ui.separator();
            ui.weak(format!("generate {:.0} ms · scene {:.0} ms · upload {:.0} ms", self.build_ms.0, self.build_ms.1, self.build_ms.2));
        });
    }
}

fn fmt_int(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}
