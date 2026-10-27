//! Parity of the training-ground generator with its Python twin (kit/site): every layout in
//! tests/sites/ built here must match tests/reference/site_<name>.json, written by
//! tools/dump_site_reference.py. Counts, names, tags, keys and warnings exactly; coordinates and
//! sums to ~1e-9 relative.

use glam::{DVec2, DVec3};
use serde_json::Value;
use stadium_gen::catalog::Catalog;
use stadium_gen::mesh::MeshAcc;
use stadium_gen::stadium::{Mat, Tex};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[track_caller]
fn close(a: f64, b: &Value, tol: f64, what: &str) {
    let b = b.as_f64().unwrap_or_else(|| panic!("{what}: expected number, got {b}"));
    let scale = 1.0f64.max(a.abs()).max(b.abs());
    assert!((a - b).abs() <= tol * scale, "{what}: rust {a} vs python {b}");
}

#[track_caller]
fn close_v(a: &[f64], b: &Value, tol: f64, what: &str) {
    let b = b.as_array().unwrap_or_else(|| panic!("{what}: expected array, got {b}"));
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        close(*x, y, tol, &format!("{what}[{i}]"));
    }
}

fn v3(v: DVec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

fn sum3(v: impl Iterator<Item = DVec3>) -> [f64; 3] {
    v3(v.fold(DVec3::ZERO, |a, b| a + b))
}

#[track_caller]
fn check_mesh(m: &MeshAcc, r: &Value, what: &str) {
    assert_eq!(m.verts.len() as u64, r["n_verts"].as_u64().unwrap(), "{what}: n_verts");
    assert_eq!(m.faces.len() as u64, r["n_faces"].as_u64().unwrap(), "{what}: n_faces");
    let sizes: Vec<u64> = m.faces.iter().map(|f| f.len() as u64).collect();
    let rs: Vec<u64> = r["face_sizes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    assert_eq!(sizes, rs, "{what}: face sizes");
    let tags: Vec<i64> = m.tags.iter().map(|t| *t as i64).collect();
    let rt: Vec<i64> = r["tags"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    assert_eq!(tags, rt, "{what}: tags");
    for (k, f) in m.faces.iter().take(6).enumerate() {
        let rf: Vec<u64> = r["first_faces"][k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(f.iter().map(|i| *i as u64).collect::<Vec<_>>(), rf, "{what}: face {k}");
    }
    if m.verts.is_empty() {
        return;
    }
    let lo = m.verts.iter().fold(DVec3::splat(f64::MAX), |a, b| a.min(*b));
    let hi = m.verts.iter().fold(DVec3::splat(f64::MIN), |a, b| a.max(*b));
    close_v(&v3(lo), &r["bounds"][0], 1e-9, &format!("{what}: bounds lo"));
    close_v(&v3(hi), &r["bounds"][1], 1e-9, &format!("{what}: bounds hi"));
    close_v(&sum3(m.verts.iter().copied()), &r["vsum"], 1e-8, &format!("{what}: vsum"));
    let us = m.uvs.iter().fold(DVec2::ZERO, |a, b| a + *b);
    close_v(&[us.x, us.y], &r["uvsum"], 1e-8, &format!("{what}: uvsum"));
}

#[track_caller]
fn check_mat(m: &Mat, r: &Value, what: &str) {
    assert_eq!(m.key, r["key"].as_str().unwrap(), "{what}: key");
    match (&m.color, &r["color"]) {
        (Some(c), v) if v.is_array() => close_v(&v3(*c), v, 1e-12, &format!("{what}: color")),
        (None, Value::Null) => {}
        (a, b) => panic!("{what}: color {a:?} vs {b}"),
    }
    match (&m.emission, &r["emission"]) {
        (Some((c, s)), v) if v.is_array() => {
            close_v(&v3(*c), &v[0], 1e-12, &format!("{what}: emission"));
            close(*s, &v[1], 1e-12, &format!("{what}: emission strength"));
        }
        (None, Value::Null) => {}
        (a, b) => panic!("{what}: emission {a:?} vs {b}"),
    }
    let t = &r["texture"];
    match &m.texture {
        Some(Tex::Grass(g)) => {
            assert_eq!(t[0], "grass", "{what}: texture");
            let s = &t[1];
            assert_eq!(g.pattern, s["pattern"].as_str().unwrap(), "{what}: pattern");
            assert_eq!(g.color, s["color"].as_str().unwrap(), "{what}: grass colour");
            assert_eq!(g.dry, s["dry"].as_str().unwrap(), "{what}: dry colour");
            assert_eq!(g.natural, s["natural"].as_bool().unwrap(), "{what}: natural");
            close_v(&[g.stripe_width, g.strength, g.wear, g.height_mm, g.yaw], &serde_json::json!([s["stripe_width"], s["strength"], s["wear"], s["height_mm"], s["yaw"]]), 1e-12, &format!("{what}: grass"));
            close_v(&[g.origin.x, g.origin.y], &s["origin"], 1e-9, &format!("{what}: grass origin"));
        }
        Some(Tex::Net(n)) => {
            assert_eq!(t[0], "net", "{what}: texture");
            let s = &t[1];
            assert_eq!(n.mesh, s["mesh"].as_str().unwrap(), "{what}: net mesh");
            assert_eq!(n.mesh_size_mm as u64, s["mesh_size_mm"].as_u64().unwrap(), "{what}: net size");
            assert_eq!(n.cord_tenths_mm as u64, s["cord_tenths_mm"].as_u64().unwrap(), "{what}: cord");
            let rc: Vec<u64> = s["colors"][0].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            assert_eq!(n.colors[0].iter().map(|c| *c as u64).collect::<Vec<_>>(), rc, "{what}: net colour");
        }
        None => assert!(t.is_null(), "{what}: texture {t}"),
        Some(_) => panic!("{what}: unexpected texture"),
    }
}

fn check_layout(name: &str) {
    let cat = Catalog::load(&root().join("catalog")).unwrap();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let layout: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("sites").join(format!("{name}.json"))).unwrap()).unwrap();
    let r: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("reference").join(format!("site_{name}.json"))).unwrap()).unwrap();
    let s = stadium_gen::site::build(&layout, &cat).unwrap();
    // report first: it tells which plots went which way
    let rep = &r["report"];
    let warnings: Vec<&str> = rep["warnings"].as_array().unwrap().iter().map(|w| w.as_str().unwrap()).collect();
    assert_eq!(s.report.warnings, warnings, "{name}: warnings");
    assert_eq!(s.report.plots.len(), rep["plots"].as_array().unwrap().len(), "{name}: plots");
    for (p, q) in s.report.plots.iter().zip(rep["plots"].as_array().unwrap()) {
        let w = format!("{name}: plot {}", p.id);
        assert_eq!(p.id, q["id"].as_str().unwrap(), "{w}");
        assert_eq!(p.state, q["state"].as_str().unwrap(), "{w} state");
        assert_eq!(p.label, q["label"].as_str().unwrap(), "{w} label");
        assert_eq!(p.facility.as_deref(), q["facility"].as_str(), "{w} facility");
        assert_eq!(p.tier as u64, q["tier"].as_u64().unwrap(), "{w} tier");
        assert_eq!(p.style.as_deref(), q["style"].as_str(), "{w} style");
    }
    for (k, v) in [("pitches", s.report.pitches), ("grass", s.report.grass), ("hybrid", s.report.hybrid), ("artificial", s.report.artificial), ("lit", s.report.lit), ("parking", s.report.parking)] {
        assert_eq!(v as u64, rep[k].as_u64().unwrap(), "{name}: report {k}");
    }
    assert_eq!(s.report.replica, rep["replica"].as_bool().unwrap(), "{name}: replica");
    let ac = &rep["academy"];
    assert_eq!(s.report.academy.category, ac["category"].as_u64().map(|v| v as usize), "{name}: academy category");
    assert_eq!(s.report.academy.next, ac["next"].as_u64().map(|v| v as usize), "{name}: academy next");
    let missing: Vec<String> = ac["missing"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(s.report.academy.missing, missing, "{name}: academy missing");
    close(s.report.area_m2, &rep["area_m2"], 1e-9, &format!("{name}: area"));
    close(s.report.developed_m2, &rep["developed_m2"], 1e-9, &format!("{name}: developed"));
    match (s.report.walk_m, &rep["walk_m"]) {
        (Some(a), b) if b.is_number() => close(a, b, 1e-9, &format!("{name}: walk")),
        (None, Value::Null) => {}
        (a, b) => panic!("{name}: walk {a:?} vs {b}"),
    }
    // parts
    let rp = r["parts"].as_array().unwrap();
    let names: Vec<&str> = s.parts.iter().map(|p| p.name.as_str()).collect();
    let rnames: Vec<&str> = rp.iter().map(|p| p["name"].as_str().unwrap()).collect();
    assert_eq!(names, rnames, "{name}: part names");
    for (p, q) in s.parts.iter().zip(rp) {
        let w = format!("{name}: {}", p.name);
        assert_eq!(p.mats.len(), q["mats"].as_array().unwrap().len(), "{w}: mats");
        for (i, (m, rm)) in p.mats.iter().zip(q["mats"].as_array().unwrap()).enumerate() {
            check_mat(m, rm, &format!("{w} mat {i}"));
        }
        close(p.smooth_deg, &q["smooth_deg"], 1e-12, &format!("{w}: smooth"));
        if p.tubes.is_empty() {
            check_mesh(&p.mesh, &q["mesh"], &w);
        } else {
            let t = &q["tubes"];
            assert_eq!(p.sides as u64, q["sides"].as_u64().unwrap(), "{w}: sides");
            assert_eq!(p.tubes.len() as u64, t["lines"].as_u64().unwrap(), "{w}: lines");
            let pts: Vec<DVec3> = p.tubes.iter().flat_map(|(l, _)| l.iter().copied()).collect();
            assert_eq!(pts.len() as u64, t["points"].as_u64().unwrap(), "{w}: points");
            close_v(&sum3(pts.into_iter()), &t["psum"], 1e-8, &format!("{w}: points sum"));
            close(p.tubes.iter().map(|(_, r)| r).sum(), &t["rsum"], 1e-9, &format!("{w}: radii"));
        }
    }
    // props
    let rq = r["props"].as_array().unwrap();
    let keys: Vec<&str> = s.props.iter().map(|p| p.key.as_str()).collect();
    let rkeys: Vec<&str> = rq.iter().map(|p| p["key"].as_str().unwrap()).collect();
    assert_eq!(keys, rkeys, "{name}: prop sets");
    for (ps, q) in s.props.iter().zip(rq) {
        let w = format!("{name}: {}", ps.key);
        assert_eq!(ps.items.len() as u64, q["count"].as_u64().unwrap(), "{w}: count");
        let tn: Vec<&str> = ps.tints.iter().map(|t| t.0.as_str()).collect();
        let rt: Vec<&str> = q["tints"].as_array().unwrap().iter().map(|t| t[0].as_str().unwrap()).collect();
        assert_eq!(tn, rt, "{w}: tints");
        for (t, rt) in ps.tints.iter().zip(q["tints"].as_array().unwrap()) {
            close_v(&v3(t.1), &rt[1], 1e-12, &format!("{w}: tint colour"));
        }
        close_v(&sum3(ps.items.iter().map(|i| i.pos)), &q["pos"], 1e-8, &format!("{w}: positions"));
        close(ps.items.iter().map(|i| i.yaw).sum(), &q["yaw"], 1e-8, &format!("{w}: yaw"));
        close(ps.items.iter().map(|i| i.tilt).sum(), &q["tilt"], 1e-8, &format!("{w}: tilt"));
        close_v(&sum3(ps.items.iter().map(|i| i.scale)), &q["scale"], 1e-8, &format!("{w}: scale"));
    }
    // pitches, lights, sockets, camera
    let rpi = r["pitches"].as_array().unwrap();
    assert_eq!(s.pitches.len(), rpi.len(), "{name}: pitches");
    for (p, q) in s.pitches.iter().zip(rpi) {
        let w = format!("{name}: pitch {}", p.plot);
        assert_eq!(p.plot, q["plot"].as_str().unwrap(), "{w}");
        assert_eq!(p.params, q["params"], "{w}: params");
        close_v(&[p.origin.x, p.origin.y], &q["origin"], 1e-9, &format!("{w}: origin"));
        close(p.yaw, &q["yaw"], 1e-12, &format!("{w}: yaw"));
        assert_eq!(p.goals.as_deref(), q["goals"].as_str(), "{w}: goals");
        assert_eq!(p.flags.as_deref(), q["flags"].as_str(), "{w}: flags");
        assert_eq!(p.both_goals, q["both_goals"].as_bool().unwrap(), "{w}: both goals");
        assert_eq!(p.lit, q["lit"].as_bool().unwrap(), "{w}: lit");
        assert_eq!(p.surface, q["surface"].as_str().unwrap(), "{w}: surface");
    }
    let sp = &r["spots"];
    assert_eq!(s.spots.len() as u64, sp["count"].as_u64().unwrap(), "{name}: spots");
    close_v(&sum3(s.spots.iter().map(|x| x.pos)), &sp["pos"], 1e-8, &format!("{name}: spot positions"));
    close_v(&sum3(s.spots.iter().map(|x| x.dir)), &sp["dir"], 1e-8, &format!("{name}: spot directions"));
    close(s.spots.iter().map(|x| x.power).sum(), &sp["power"], 1e-8, &format!("{name}: spot power"));
    let rs = r["sockets"].as_array().unwrap();
    assert_eq!(s.sockets.len(), rs.len(), "{name}: sockets");
    for (k, q) in s.sockets.iter().zip(rs) {
        assert_eq!(k.kind, q["kind"].as_str().unwrap(), "{name}: socket kind");
        assert_eq!(k.id, q["id"].as_str().unwrap(), "{name}: socket id");
        close_v(&v3(k.loc), &q["loc"], 1e-9, &format!("{name}: socket {}", k.id));
        close_v(&[k.n.x, k.n.y], &q["n"], 1e-12, &format!("{name}: socket {} normal", k.id));
    }
    close_v(&v3(s.camera.0), &r["camera"][0], 1e-9, &format!("{name}: camera"));
    close_v(&v3(s.camera.1), &r["camera"][1], 1e-9, &format!("{name}: camera target"));
}

#[test]
fn site_mix() {
    check_layout("mix");
}

#[test]
fn site_river_night() {
    check_layout("river_night");
}

#[test]
fn site_all_occupants() {
    check_layout("all_occupants");
}

#[test]
fn site_works() {
    check_layout("works");
}

#[test]
fn site_chain() {
    check_layout("chain");
}

#[test]
fn site_academy() {
    check_layout("academy");
}

#[test]
fn site_academy_elite() {
    check_layout("academy_elite");
}

#[test]
fn site_medical() {
    check_layout("medical");
}

#[test]
fn site_medical_cut() {
    check_layout("medical_cut");
}

#[test]
fn site_medical_small() {
    check_layout("medical_small");
}

#[test]
fn site_scouting() {
    check_layout("scouting");
}

#[test]
fn site_scouting_cut() {
    check_layout("scouting_cut");
}

#[test]
fn site_scouting_small() {
    check_layout("scouting_small");
}

#[test]
fn site_commercial() {
    check_layout("commercial");
}

#[test]
fn site_commercial_cut() {
    check_layout("commercial_cut");
}

#[test]
fn site_commercial_small() {
    check_layout("commercial_small");
}

#[test]
fn site_media() {
    check_layout("media");
}

#[test]
fn site_media_cut() {
    check_layout("media_cut");
}

#[test]
fn site_media_small() {
    check_layout("media_small");
}
