//! The training-ground editor: site settings (template, club, style, boundary, gate) and every
//! plot's state, with the player's moves as buttons (knock down, clear, build, finish, upgrade) and
//! the facility, tier, style and options of what stands there. Readouts: pitches, parking, area,
//! plot states and the facilities list.

use super::{cat_opts, check, color, combo, fmt_int, get_s, label_row, num, opts, Editor, Section};
use egui::{Color32, RichText, Ui};
use serde_json::{json, Value};
use stadium_gen::site;

const STATE_LABELS: [(&str, &str); 5] = [("occupied", "Occupied"), ("demolition", "Being demolished"), ("cleared", "Cleared"), ("construction", "Under construction"), ("built", "Built")];

impl Editor {
    /// Bring a site layout to its full form (every plot listed, merges resolved) for editing.
    pub fn normalize_site(&mut self) {
        let (l, _) = site::sanitize(&self.layout, &self.cat);
        if l != self.layout {
            self.layout = l;
        }
    }

    fn template(&self) -> Value {
        let t = self.layout.get("template").and_then(|v| v.as_str()).unwrap_or("edge_of_town");
        self.cat.sites.get(t).cloned().unwrap_or(Value::Null)
    }

    fn plot_kind(&self, id: &str) -> String {
        let tpl = self.template();
        let find = |list: &Value| list.as_array().and_then(|a| a.iter().find(|p| p["id"].as_str() == Some(id))).and_then(|p| p["kind"].as_str()).map(String::from);
        find(&tpl["plots"]).or_else(|| find(&tpl["merges"])).unwrap_or_else(|| "medium".into())
    }

    pub fn site_controls(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.spacing_mut().slider_width = 150.0;
            let all = self.expand_all;
            let selected = self.focus.as_ref().filter(|f| f.section == Section::Plot).and_then(|f| f.side.clone());
            let reveal = if self.reveal { selected.clone() } else { None };
            self.reveal = false;
            egui::CollapsingHeader::new(RichText::new("Site").strong()).default_open(true).show(ui, |ui| {
                let mut c = false;
                let mut tpls: Vec<(String, String)> = cat_opts(&self.cat.sites);
                tpls.retain(|t| t.0 != "plot_kinds");
                let before = self.layout["template"].clone();
                c |= combo(ui, "tpl", "Land", &mut self.layout, "template", "edge_of_town", &tpls);
                if self.layout["template"] != before {
                    // a different site: its own plots
                    self.layout["plots"] = json!({});
                    self.normalize_site();
                }
                let mut name = get_s(&self.layout, "name", "").to_string();
                c |= label_row(ui, "Name", |ui| ui.text_edit_singleline(&mut name).changed());
                self.layout["name"] = json!(name);
                let club = super::obj(&mut self.layout, "club");
                let mut cn = get_s(club, "name", "KIT FC").to_string();
                c |= label_row(ui, "Club", |ui| ui.text_edit_singleline(&mut cn).changed());
                club["name"] = json!(cn);
                c |= color(ui, "Club colour", club, "primary", "#c8102e");
                c |= color(ui, "Second colour", club, "secondary", "#f2f2f2");
                let styles = cat_opts(&self.cat.styles);
                c |= combo(ui, "style", "Style", &mut self.layout, "style", "classic", &styles);
                let b = super::obj(&mut self.layout, "boundary");
                c |= combo(ui, "btype", "Boundary", b, "type", "chainlink", &opts(&site::BOUNDARIES));
                c |= combo(ui, "bscreen", "Screening", b, "screening", "none", &opts(&site::SCREENING));
                let g = super::obj(&mut self.layout, "gate");
                c |= num(ui, "Gate", g, "tier", 1.0, 1.0..=3.0, 1.0, "");
                ui.weak(match g.get("tier").and_then(|v| v.as_u64()).unwrap_or(1) {
                    1 => "    farm gate",
                    2 => "    barrier and security hut",
                    _ => "    gatehouse",
                });
                let ls = super::obj(&mut self.layout, "landscape");
                c |= num(ui, "Trees", ls, "trees", 0.5, 0.0..=1.0, 0.05, "");
                self.touch(c);
            });
            let ids: Vec<String> = self.layout["plots"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
            ui.add_space(4.0);
            ui.label(RichText::new("Plots").strong());
            for id in ids {
                let kind = self.plot_kind(&id);
                let cfg = self.layout["plots"][&id].clone();
                let state = get_s(&cfg, "state", "occupied").to_string();
                let what = match state.as_str() {
                    "occupied" | "demolition" => self.cat.occupants.get(get_s(&cfg, "occupant", "")).and_then(|o| o["label"].as_str()).unwrap_or("").to_string(),
                    "construction" | "built" => {
                        let f = get_s(&cfg, "facility", "");
                        let tier = cfg.get("tier").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                        let tl = site::tier_spec(&self.cat, f, tier).and_then(|t| t["label"].as_str()).unwrap_or("");
                        format!("{} ({tl})", self.cat.facilities.get(f).and_then(|v| v["label"].as_str()).unwrap_or(f))
                    }
                    _ => String::new(),
                };
                let st_label = STATE_LABELS.iter().find(|s| s.0 == state).map_or("", |s| s.1);
                let mut title = RichText::new(format!("{id} · {kind} · {st_label} {what}"));
                if selected.as_deref() == Some(id.as_str()) {
                    title = title.color(Color32::from_rgb(255, 196, 90)).strong();
                }
                let h = egui::CollapsingHeader::new(title).id_salt(format!("plot-{id}")).default_open(all);
                let h = if reveal.as_deref() == Some(id.as_str()) { h.open(Some(true)) } else { h };
                h.show(ui, |ui| {
                    if reveal.as_deref() == Some(id.as_str()) {
                        ui.scroll_to_cursor(Some(egui::Align::TOP));
                    }
                    self.plot_ui(ui, &id, &kind);
                });
            }
        });
    }

    fn plot_ui(&mut self, ui: &mut Ui, id: &str, kind: &str) {
        let cat = self.cat.clone();
        let mut c = false;
        let mut cfg = self.layout["plots"][id].clone();
        let state = get_s(&cfg, "state", "occupied").to_string();
        // the player's moves
        ui.horizontal_wrapped(|ui| match state.as_str() {
            "occupied" => {
                if ui.button("Knock down").clicked() {
                    cfg["state"] = json!("demolition");
                    c = true;
                }
            }
            "demolition" => {
                if ui.button("Clear the site").clicked() {
                    cfg = json!({"state": "cleared"});
                    c = true;
                }
            }
            "cleared" => {
                for (fac, spec) in cat.facilities.iter() {
                    if site::fits(&cat, fac, 1, kind) && ui.button(format!("Build {}", spec["label"].as_str().unwrap_or(fac).to_lowercase())).clicked() {
                        cfg = json!({"state": "construction", "facility": fac, "tier": 1, "options": {}});
                        c = true;
                    }
                }
            }
            "construction" => {
                if ui.button("Finish").clicked() {
                    cfg["state"] = json!("built");
                    c = true;
                }
            }
            _ => {
                let fac = get_s(&cfg, "facility", "").to_string();
                let tier = cfg.get("tier").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                if tier < site::tier_count(&cat, &fac) && site::fits(&cat, &fac, tier + 1, kind) && ui.button("Upgrade").on_hover_text("builds the next tier").clicked() {
                    cfg["tier"] = json!(tier + 1);
                    cfg["state"] = json!("construction");
                    c = true;
                }
                if ui.button("Demolish").clicked() {
                    cfg = json!({"state": "cleared"});
                    c = true;
                }
            }
        });
        let states: Vec<(String, String)> = STATE_LABELS.iter().map(|(k, l)| (k.to_string(), l.to_string())).collect();
        c |= combo(ui, &format!("{id}-state"), "State", &mut cfg, "state", "occupied", &states);
        let state = get_s(&cfg, "state", "occupied").to_string();
        match state.as_str() {
            "occupied" | "demolition" => {
                let occ: Vec<(String, String)> = cat
                    .occupants
                    .iter()
                    .filter(|(_, v)| v["fits"].as_array().is_some_and(|a| a.iter().any(|k| k.as_str() == Some(kind))))
                    .map(|(k, v)| (k.clone(), v["label"].as_str().unwrap_or(k).to_string()))
                    .collect();
                let d = occ.first().map(|o| o.0.clone()).unwrap_or_default();
                c |= combo(ui, &format!("{id}-occ"), "On the plot", &mut cfg, "occupant", &d, &occ);
            }
            "construction" | "built" => {
                let facs: Vec<(String, String)> = cat.facilities.iter().filter(|(f, _)| site::fits(&cat, f, 1, kind)).map(|(k, v)| (k.clone(), v["label"].as_str().unwrap_or(k).to_string())).collect();
                let d = facs.first().map(|o| o.0.clone()).unwrap_or_default();
                let before = cfg.get("facility").cloned();
                c |= combo(ui, &format!("{id}-fac"), "Facility", &mut cfg, "facility", &d, &facs);
                if cfg.get("facility") != before.as_ref() {
                    cfg["tier"] = json!(1);
                    cfg["options"] = json!({});
                }
                let fac = get_s(&cfg, "facility", &d).to_string();
                let n = site::tier_count(&cat, &fac).max(1);
                let top = (1..=n).rev().find(|&t| site::fits(&cat, &fac, t, kind)).unwrap_or(1);
                c |= num(ui, "Tier", &mut cfg, "tier", 1.0, 1.0..=top as f64, 1.0, &format!(" of {n}"));
                let tier = cfg.get("tier").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                if let Some(t) = site::tier_spec(&cat, &fac, tier) {
                    ui.weak(format!("    {}", t["label"].as_str().unwrap_or("")));
                }
                if top < n {
                    ui.weak(format!("    tier {} and up need a bigger plot", top + 1));
                }
                let building = matches!(fac.as_str(), "training_building" | "indoor_hall" | "recovery" | "maintenance");
                if building {
                    let mut styles = vec![(String::new(), "Site style".to_string())];
                    styles.extend(cat_opts(&cat.styles));
                    c |= combo(ui, &format!("{id}-style"), "Style", &mut cfg, "style", "", &styles);
                    if cfg.get("style").and_then(|v| v.as_str()) == Some("") {
                        cfg.as_object_mut().unwrap().remove("style");
                    }
                }
                let o = super::obj(&mut cfg, "options");
                if building {
                    let style = self.layout["style"].as_str().unwrap_or("classic").to_string();
                    let st = cat.styles.get(&style).cloned().unwrap_or(Value::Null);
                    c |= color(ui, "Walls", o, "wall_color", st["wall_color"].as_str().unwrap_or("#c8c4bb"));
                    c |= color(ui, "Trim", o, "trim_color", st["trim_color"].as_str().unwrap_or("#2a2c30"));
                }
                if let Some(options) = cat.facilities.get(&fac).and_then(|f| f["options"].as_object()) {
                    for (key, spec) in options {
                        let min_tier = spec["min_tier"].as_u64().unwrap_or(1) as usize;
                        if tier < min_tier {
                            continue;
                        }
                        let label = key.replace('_', " ");
                        let label = format!("{}{}", label[..1].to_uppercase(), &label[1..]);
                        if let Some(r) = spec["range"].as_array() {
                            let (lo, hi) = (r[0].as_f64().unwrap_or(0.0), r[1].as_f64().unwrap_or(1.0));
                            let d = spec["default"].as_f64().unwrap_or(lo);
                            let step = if hi - lo > 5.0 { 1.0 } else { 0.05 };
                            c |= num(ui, &label, o, key, d, lo..=hi, step, if step >= 1.0 { " m" } else { "" });
                        } else if let Some(ch) = spec["choices"].as_array() {
                            let names: Vec<&str> = ch.iter().filter_map(|v| v.as_str()).collect();
                            if spec["default"].is_array() {
                                // a set (modules)
                                let mut cur: Vec<String> = o.get(key).and_then(|v| v.as_array()).or(spec["default"].as_array()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
                                ui.label(format!("{label}:"));
                                ui.horizontal_wrapped(|ui| {
                                    for nm in &names {
                                        let mut on = cur.iter().any(|x| x == nm);
                                        if ui.checkbox(&mut on, *nm).changed() {
                                            if on {
                                                cur.push(nm.to_string());
                                            } else {
                                                cur.retain(|x| x != nm);
                                            }
                                            c = true;
                                        }
                                    }
                                });
                                o[key.as_str()] = json!(cur);
                            } else {
                                let tier_default = site::tier_spec(&cat, &fac, tier).and_then(|t| t[key.as_str()].as_str()).unwrap_or(names[0]).to_string();
                                c |= combo(ui, &format!("{id}-{key}"), &label, o, key, &tier_default, &opts(&names));
                            }
                        } else {
                            let tier_default = site::option(&cat, &fac, tier, &Value::Null, key).as_bool().unwrap_or(false);
                            c |= check(ui, &label, o, key, tier_default);
                        }
                    }
                }
            }
            _ => {}
        }
        // merges: build two neighbouring plots as one
        let tpl = self.template();
        if let Some(merges) = tpl["merges"].as_array() {
            for m in merges {
                let mid = m["id"].as_str().unwrap_or("");
                let parts: Vec<&str> = m["plots"].as_array().map(|a| a.iter().filter_map(|v| v.as_str()).collect()).unwrap_or_default();
                if id != mid && !parts.contains(&id) {
                    continue;
                }
                let merged = id == mid;
                let mut want = merged;
                let can = merged || parts.iter().all(|p| self.layout["plots"].get(*p).is_some_and(|v| get_s(v, "state", "") == "cleared"));
                ui.add_enabled_ui(can, |ui| {
                    if ui.checkbox(&mut want, format!("Build {} as one plot", parts.join(" + "))).on_disabled_hover_text("clear both plots first").changed() {
                        let plots = self.layout["plots"].as_object_mut().unwrap();
                        if want {
                            for p in &parts {
                                plots.shift_remove(*p);
                            }
                            plots.insert(mid.into(), json!({"state": "cleared"}));
                        } else {
                            plots.shift_remove(mid);
                            for p in &parts {
                                plots.insert(p.to_string(), json!({"state": "cleared"}));
                            }
                        }
                        self.revision += 1;
                    }
                });
                if want != merged {
                    return;
                }
            }
        }
        if c {
            self.layout["plots"][id] = cfg;
            self.revision += 1;
        }
    }

    pub fn site_readouts(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            if let Some(e) = &self.error {
                ui.colored_label(Color32::from_rgb(255, 120, 100), format!("⚠ {e}"));
                ui.separator();
            }
            let Some(r) = &self.site_report else {
                ui.weak("building…");
                return;
            };
            ui.heading(format!("{} pitches", r.pitches));
            ui.label(format!("{} grass · {} hybrid · {} artificial · {} floodlit", r.grass, r.hybrid, r.artificial, r.lit));
            if r.replica {
                ui.label("Match-replica pitch");
            }
            ui.label(format!("{} parking spaces", fmt_int(r.parking)));
            ui.label(format!("Site {:.1} ha, developed {:.1} ha", r.area_m2 / 1e4, r.developed_m2 / 1e4));
            if let Some(wk) = r.walk_m {
                ui.label(format!("Training building to the furthest pitch: {wk:.0} m"));
            }
            // academy grade (only worth showing once a site has youth pitches or an academy building)
            let academy = r.plots.iter().any(|p| matches!(p.facility.as_deref(), Some("youth_pitches" | "academy_building")));
            if academy {
                ui.separator();
                let a = &r.academy;
                ui.strong(match a.category {
                    Some(c) => format!("Academy: Category {c}"),
                    None => "Academy: not yet Category 4".into(),
                });
                if let Some(n) = a.next {
                    ui.label(format!("For Category {n}:"));
                    for m in &a.missing {
                        ui.label(format!("  • {m}"));
                    }
                }
            }
            if !r.warnings.is_empty() {
                ui.separator();
                for w in &r.warnings {
                    ui.colored_label(Color32::from_rgb(240, 190, 80), format!("⚠ {w}"));
                }
            }
            ui.separator();
            let count = |s: &str| r.plots.iter().filter(|p| p.state == s).count();
            ui.label(format!("{} plots: {} built, {} under construction, {} cleared, {} being demolished, {} still occupied", r.plots.len(), count("built"), count("construction"), count("cleared"), count("demolition"), count("occupied")));
            ui.separator();
            egui::Grid::new("plots").striped(true).show(ui, |ui| {
                ui.strong("plot");
                ui.strong("what stands there");
                ui.strong("style");
                ui.end_row();
                for p in &r.plots {
                    ui.label(&p.id);
                    ui.label(&p.label);
                    ui.weak(p.style.as_deref().filter(|_| matches!(p.facility.as_deref(), Some("training_building" | "indoor_hall" | "recovery" | "maintenance"))).unwrap_or(""));
                    ui.end_row();
                }
            });
            ui.separator();
            ui.weak(format!("generate {:.0} ms · scene {:.0} ms · {:.0} fps", self.build_ms.0, self.build_ms.1, self.fps));
        });
    }
}
