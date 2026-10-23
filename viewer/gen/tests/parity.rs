//! Parity with the Python reference generators (tests/reference/*.json, written by
//! tools/dump_reference.py). Geometry must match to floating-point noise: counts and indices
//! exactly, coordinates and sums to ~1e-9 relative.

use glam::{DVec2, DVec3};
use serde_json::{json, Value};
use stadium_gen::catalog::Catalog;
use stadium_gen::mesh::MeshAcc;
use stadium_gen::*;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn reference(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference").join(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn cat() -> Catalog {
    Catalog::load(&root().join("catalog")).unwrap()
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
    let vs = m.verts.iter().fold(DVec3::ZERO, |a, b| a + *b);
    close_v(&v3(vs), &r["vsum"], 1e-8, &format!("{what}: vsum"));
    let us = m.uvs.iter().fold(DVec2::ZERO, |a, b| a + *b);
    close_v(&[us.x, us.y], &r["uvsum"], 1e-8, &format!("{what}: uvsum"));
}

fn pts3(v: &[DVec3]) -> Vec<f64> {
    v.iter().flat_map(|p| v3(*p)).collect()
}

fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        Value::Bool(b) => vec![if *b { 1.0 } else { 0.0 }],
        _ => vec![],
    }
}

#[test]
fn sightlines() {
    let c = cat();
    let r = reference("sightlines");
    for (name, prof) in &c.profiles {
        for d in [9.0, 13.5] {
            let tiers = sightlines::solve_profile(prof, d);
            let rt = r[name][format!("{d:?}")].as_array().unwrap();
            assert_eq!(tiers.len(), rt.len(), "{name}");
            for (t, x) in tiers.iter().zip(rt) {
                let w = format!("{name}@{d} tier {}", t.index);
                assert_eq!(t.rows as u64, x["rows"].as_u64().unwrap(), "{w}");
                assert_eq!(if t.kind == sightlines::Kind::Standing { "standing" } else { "seated" }, x["kind"].as_str().unwrap(), "{w}");
                close(t.tread, &x["tread"], 1e-12, &w);
                close(t.c_target, &x["c_target"], 1e-12, &w);
                close(t.s0, &x["s0"], 1e-12, &format!("{w} s0"));
                close_v(&t.h, &x["h"], 1e-12, &format!("{w} h"));
                close_v(&t.s, &x["s"], 1e-12, &format!("{w} s"));
                close_v(&t.c_achieved, &x["c_achieved"], 1e-12, &format!("{w} c"));
                close(t.s_back, &x["s_back"], 1e-12, &w);
                close(t.setback, &x["setback"], 1e-12, &w);
                close(t.parapet, &x["parapet"], 1e-12, &w);
                close(t.glass, &x["glass"], 1e-12, &w);
                close(t.slab, &x["slab"], 1e-12, &w);
                close(t.seat_pitch, &x["seat_pitch"], 1e-12, &w);
                assert_eq!(t.vomitories, x["vomitories"].as_bool().unwrap(), "{w}");
                assert_eq!(t.hospitality, x["hospitality"].as_bool().unwrap(), "{w}");
                match t.soffit_front {
                    Some(sf) => {
                        close(sf, &x["soffit_front"], 1e-12, &w);
                        close(t.soffit_slope, &x["soffit_slope"], 1e-12, &w);
                    }
                    None => assert!(x["soffit_front"].is_null(), "{w}"),
                }
            }
        }
    }
}

fn scenario(c: &Catalog, sides: &[(&str, &str, Option<f64>)], curved: bool, r_corner: f64, gap: f64, r: &Value) {
    let p = pitch::Pitch::resolve(&json!({}), Some(c));
    let a = p.length / 2.0 + p.runoff_end + gap;
    let b = p.width / 2.0 + p.runoff_side + gap;
    close(a, &r["A"], 1e-12, "A");
    close(b, &r["B"], 1e-12, "B");
    let side_list: Vec<(String, Option<f64>)> = sides.iter().map(|s| (s.0.to_string(), s.2)).collect();
    let runs = runs::perimeter_runs(a, b, &side_list, curved, r_corner);
    let rr = r["runs"].as_array().unwrap();
    assert_eq!(runs.len(), rr.len());
    let focal = |n: &str| match n {
        "south" | "north" => p.runoff_side + gap,
        _ => p.runoff_end + gap,
    };
    for (i, (run, x)) in runs.iter().zip(rr).enumerate() {
        let w = run.name.clone();
        assert_eq!(run.name, x["run"]["name"].as_str().unwrap());
        assert_eq!(run.curved(), x["run"]["curved"].as_bool().unwrap());
        close(run.t0, &x["run"]["t0"], 1e-12, &w);
        close(run.t1, &x["run"]["t1"], 1e-12, &w);
        let prof = sides.iter().find(|s| s.0 == run.name).unwrap_or(&sides[0]).1;
        let df = if curved { focal("south").min(focal("east")) } else { focal(&run.name) };
        close(df, &x["d_focal"], 1e-12, &w);
        let mut tiers = sightlines::solve_profile(&c.profiles[prof], df);
        for t in tiers.iter_mut() {
            t.seat_pitch = t.seat_pitch.max(0.5);
        }
        let o = stands::build_run(run, i, &tiers);
        close_v(&o.bay_list.iter().flat_map(|b| [b.0, b.1]).collect::<Vec<_>>(), &Value::Array(x["bay_list"].as_array().unwrap().iter().flat_map(|b| b.as_array().unwrap().clone()).collect()), 1e-12, &format!("{w} bays"));
        check_mesh(&o.mesh, &x["mesh"], &w);
        let an = &x["anchors"];
        assert_eq!(o.anchors.len() as u64, an["count"].as_u64().unwrap(), "{w} anchors");
        let n = o.anchors.len();
        let pick: Vec<usize> = if n == 0 { vec![] } else { (0..4).chain(n / 2..n / 2 + 2).chain(n - 4..n).collect() };
        for (k, &ai) in pick.iter().enumerate() {
            let (a_, s) = (&o.anchors[ai], &an["sample"][k]);
            let ww = format!("{w} anchor {ai}");
            close_v(&v3(a_.pos), &s["pos"], 1e-10, &ww);
            close(a_.yaw, &s["yaw"], 1e-12, &ww);
            close(a_.t, &s["t"], 1e-12, &ww);
            close(a_.pitch, &s["pitch"], 1e-12, &ww);
            close(a_.riser, &s["riser"], 1e-12, &ww);
            assert_eq!([a_.tier, a_.row, a_.bay, a_.seat, a_.of].map(|v| v as u64), ["tier", "row", "bay", "seat", "of"].map(|k| s[k].as_u64().unwrap()), "{ww}");
        }
        let ps = o.anchors.iter().fold(DVec3::ZERO, |acc, a| acc + a.pos);
        close_v(&v3(ps), &an["pos_sum"], 1e-9, &format!("{w} anchor sum"));
        assert_eq!(o.rails.len() as u64, x["rails"]["count"].as_u64().unwrap(), "{w} rails");
        assert_eq!(o.rails.iter().map(|r| r.0.len()).sum::<usize>() as u64, x["rails"]["points"].as_u64().unwrap(), "{w} rail pts");
        let rs = o.rails.iter().flat_map(|r| r.0.iter()).fold(DVec3::ZERO, |a, b| a + *b);
        close_v(&v3(rs), &x["rails"]["psum"], 1e-9, &format!("{w} rail sum"));
        assert_eq!(o.nosings.len() as u64, x["nosings"]["count"].as_u64().unwrap(), "{w} nosings");
        let ns = o.nosings.iter().fold(DVec3::ZERO, |a, b| a + b.0 + b.1);
        close_v(&v3(ns), &x["nosings"]["psum"], 1e-9, &format!("{w} nosing sum"));
        assert_eq!(o.barriers.len() as u64, x["barriers"]["count"].as_u64().unwrap(), "{w} barriers");
        assert_eq!(o.barriers.iter().map(|b| b.len()).sum::<usize>() as u64, x["barriers"]["points"].as_u64().unwrap());
        let rsock = x["sockets"].as_array().unwrap();
        assert_eq!(o.sockets.len(), rsock.len(), "{w} sockets");
        for (s, q) in o.sockets.iter().zip(rsock) {
            assert_eq!(s.kind, q[0].as_str().unwrap());
            assert_eq!(s.id, q[1].as_str().unwrap());
            close_v(&v3(s.loc), &q[2], 1e-10, &s.id);
            close_v(&[s.n.x, s.n.y], &q[3], 1e-12, &s.id);
            for (k, v) in &s.props {
                close(*v, &q[4][*k], 1e-12, &format!("{} {k}", s.id));
            }
        }
        close(stands::stand_depth(&tiers), &x["stand_depth"], 1e-12, &w);
        let hull = stands::section_hull(&tiers);
        close_v(&hull.iter().flat_map(|p| [p.x, p.y]).collect::<Vec<_>>(), &Value::Array(flat(&x["hull"]).into_iter().map(|v| json!(v)).collect()), 1e-12, &format!("{w} hull"));
    }
}

#[test]
fn stands_bowl() {
    let c = cat();
    let r = reference("stands");
    let sides: Vec<(&str, &str, Option<f64>)> = ["south", "north", "east", "west"].iter().map(|s| (*s, "three_tier_bowl", None)).collect();
    scenario(&c, &sides, true, 14.0, 3.0, &r["bowl"]);
}

#[test]
fn stands_open() {
    let c = cat();
    let r = reference("stands");
    let sides = [
        ("south", "two_tier_22_18", None),
        ("north", "single_tier_20", None),
        ("east", "kop_terrace_40", None),
        ("west", "single_tier_20", Some(60.0)),
    ];
    scenario(&c, &sides, false, 0.0, 2.5, &r["open"]);
}

#[test]
fn roofs() {
    let c = cat();
    let r = reference("roofs");
    let run = runs::Run::straight("south", DVec2::new(-60.0, -45.0), DVec2::new(1.0, 0.0), 120.0);
    for pname in ["small_seated_8", "single_tier_20", "two_tier_22_18", "three_tier_bowl", "mega_bowl"] {
        let tiers = sightlines::solve_profile(&c.profiles[pname], 9.0);
        let bay_list = run.bays(20.0);
        for (rname, spec) in &c.roofs {
            let w = format!("{pname}/{rname}");
            let x = &r[&w];
            let sol = roofs::solve(&tiers, spec, 9.0, 68.0);
            let s = &x["solve"];
            assert_eq!(s["type"].as_str().unwrap(), if sol.kind == roofs::RoofKind::Goalpost { "goalpost" } else { "cantilever" });
            for (k, v) in [
                ("s_back", sol.s_back),
                ("s_front", sol.s_front),
                ("z_back", sol.z_back),
                ("z_front", sol.z_front),
                ("length", sol.length),
                ("raised_for_sightline", sol.raised_for_sightline),
                ("raised_for_headroom", sol.raised_for_headroom),
                ("clearance_to_sightline", sol.clearance_to_sightline),
            ] {
                close(v, &s[k], 1e-11, &format!("{w} {k}"));
            }
            if sol.kind == roofs::RoofKind::Cantilever {
                close(sol.depth_back, &s["depth_back"], 1e-12, &w);
                close(sol.depth_front, &s["depth_front"], 1e-12, &w);
                close(sol.back_span, &s["back_span"], 1e-12, &w);
            } else {
                close(sol.s_post, &s["s_post"], 1e-12, &w);
            }
            let b = roofs::build(&run, &sol, &bay_list, true);
            check_mesh(&b.mesh, &x["mesh"], &w);
            assert_eq!(b.tubes.len() as u64, x["tubes"]["count"].as_u64().unwrap(), "{w} tubes");
            assert_eq!(b.tubes.iter().map(|t| t.0.len()).sum::<usize>() as u64, x["tubes"]["points"].as_u64().unwrap(), "{w} tube pts");
            let ps = b.tubes.iter().flat_map(|t| t.0.iter()).fold(DVec3::ZERO, |a, q| a + *q);
            close_v(&v3(ps), &x["tubes"]["psum"], 1e-9, &format!("{w} tube sum"));
            close(b.tubes.iter().map(|t| t.1).sum(), &x["tubes"]["rsum"], 1e-10, &format!("{w} radii"));
            close_v(&pts3(&b.edge_line), &Value::Array(flat(&x["edge_line"]).into_iter().map(|v| json!(v)).collect()), 1e-10, &format!("{w} edge"));
            close(b.tail_s, &x["tail_s"], 1e-11, &w);
            close(b.tail_top, &x["tail_top"], 1e-11, &w);
            assert_eq!(b.sockets.len() as u64, x["sockets"].as_u64().unwrap());
        }
    }
}

#[test]
fn facades() {
    let c = cat();
    let r = reference("facades");
    let run = runs::Run::straight("south", DVec2::new(-60.0, -45.0), DVec2::new(1.0, 0.0), 120.0);
    let arc = runs::Run::arc("NE", DVec2::new(40.0, 20.0), 14.0, 0.0, std::f64::consts::PI / 2.0);
    let tiers = sightlines::solve_profile(&c.profiles["three_tier_bowl"], 9.0);
    let sol = roofs::solve(&tiers, &json!({"type": "cantilever"}), 9.0, 68.0);
    let rb = roofs::build(&run, &sol, &run.bays(20.0), true);
    let info = facades::RoofInfo { tail_s: rb.tail_s, tail_top: rb.tail_top };
    for (fname, spec) in &c.facades {
        for (rr, ends) in [(&run, (true, true)), (&arc, (false, false))] {
            let w = format!("{fname}/{}", rr.name);
            let x = &r[&w];
            let o = facades::build(rr, &tiers, &rr.bays(20.0), spec, Some(info), ends);
            check_mesh(&o.mesh, &x["mesh"], &w);
            close(o.line_s, &x["line_s"], 1e-12, &w);
            close(o.height, &x["height"], 1e-12, &w);
            assert_eq!(o.tubes.len() as u64, x["tubes"].as_u64().unwrap(), "{w} tubes");
            assert_eq!(o.tubes.iter().map(|t| t.0.len()).sum::<usize>() as u64, x["tube_points"].as_u64().unwrap(), "{w} tube pts");
            assert_eq!(o.sockets.len() as u64, x["sockets"].as_u64().unwrap());
        }
    }
}

#[test]
fn floodlight_layouts() {
    let r = reference("floodlights");
    let (l, w) = (105.0, 68.0);
    let masts = floodlights::pylons(l, w, 65.5, 43.0, 10.0, 25.0);
    for (m, x) in masts.iter().zip(r["pylons"].as_array().unwrap()) {
        close_v(&[m.base.x, m.base.y], &x["base"], 1e-12, "pylon base");
        close(m.height, &x["height"], 1e-12, "pylon height");
    }
    for (m, x) in masts.iter().zip(r["heads"].as_array().unwrap()) {
        let (pts, _, _) = floodlights::head_layout(m, 5, 7);
        close_v(&pts3(&pts), &Value::Array(flat(x).into_iter().map(|v| json!(v)).collect()), 1e-12, "head");
    }
    for (m, x) in floodlights::poles(l, w, 3, 20.0, 7.2).iter().zip(r["poles"].as_array().unwrap()) {
        close_v(&[m.base.x, m.base.y], &x["base"], 1e-12, "pole");
    }
    let pts: Vec<DVec3> = [-40.0, 0.0, 30.0].iter().flat_map(|&x| [-20.0, 10.0].map(|y| DVec3::new(x, y, 0.0))).collect();
    let heads: Vec<DVec3> = masts.iter().map(|m| floodlights::head_layout(m, 5, 7).0[0]).collect();
    let tg = [DVec3::ZERO, DVec3::new(20.0, 5.0, 0.0), DVec3::new(-30.0, -10.0, 0.0), DVec3::new(0.0, 20.0, 0.0)];
    let e = floodlights::illuminance(&heads, &tg, 1000.0, 40.0, 0.9, &pts);
    close_v(&e, &r["illuminance"], 1e-10, "illuminance");
    let rib = floodlights::roof_ribbon(&[vec![DVec3::new(0.0, 0.0, 30.0), DVec3::new(10.0, 0.0, 30.0), DVec3::new(10.0, 25.0, 31.0)]], 2.4);
    close_v(&pts3(&rib), &Value::Array(flat(&r["ribbon"]).into_iter().map(|v| json!(v)).collect()), 1e-12, "ribbon");
    close(floodlights::spot_angle(160, l * w, 60.0), &r["spot_angle"], 1e-12, "spot angle");
    close(floodlights::lamp_power(160, 40.0, 2000.0, l * w), &r["lamp_power"], 1e-12, "lamp power");
}

#[test]
fn floodlight_optimiser() {
    let r = reference("floodlights");
    let (l, w) = (105.0, 68.0);
    let grid: Vec<DVec3> = py::linspace(-w / 2.0, w / 2.0, 15).iter().flat_map(|&y| py::linspace(-l / 2.0, l / 2.0, 22).into_iter().map(move |x| DVec3::new(x, y, 0.0))).collect();
    let cands: Vec<DVec3> = py::linspace(-w / 2.0, w / 2.0, 9).iter().flat_map(|&y| py::linspace(-l / 2.0, l / 2.0, 15).into_iter().map(move |x| DVec3::new(x, y, 0.0))).collect();
    let small: Vec<DVec3> = [1.0, -1.0].iter().flat_map(|&s| [-30.0, 0.0, 30.0].map(|x| DVec3::new(x, s * (w / 2.0 + 7.2), 20.3))).collect();
    let (t, e) = floodlights::optimise_aims(&small, &cands, &grid, 90.0);
    let o = &r["optimise"];
    close_v(&pts3(&t), &Value::Array(flat(&o["targets"]).into_iter().map(|v| json!(v)).collect()), 1e-12, "targets");
    close_v(&e, &o["E"], 1e-9, "E");
    let (u1, u2) = floodlights::uniformity(&e);
    close_v(&[u1, u2], &o["u"], 1e-9, "uniformity");
}

#[test]
fn placement() {
    let c = cat();
    let r = reference("placement");
    let p = pitch::Pitch::resolve(&json!({}), Some(&c));
    let runs = placement::board_runs(&p, 22.0, placement::BOARD_SIDE_OFFSET, placement::BOARD_END_OFFSET);
    for (run, x) in runs.iter().zip(r["board_runs"].as_array().unwrap()) {
        assert_eq!(run.name, x["name"].as_str().unwrap());
        let (p0, _) = run.frame(0.0);
        close_v(&[p0.x, p0.y], &x["p0"], 1e-12, &run.name);
        close(run.t1, &x["length"], 1e-12, &run.name);
    }
    for (s, x) in placement::carpet_spots(&p, 5.0, 0.8).iter().zip(r["carpet_spots"].as_array().unwrap()) {
        close(s.0, &x[0], 0.0, "spot");
        close(s.1, &x[1], 0.0, "spot");
        close_v(&v3(s.2), &x[2], 1e-12, "spot");
    }
    let cam = DVec3::new(0.0, -70.0, 22.0);
    let (v, f, uv) = placement::carpet(cam, DVec3::new(53.3, 8.0, 0.0), 5.0, 1.0, 8, 4, &|_, _| 0.0);
    close_v(&pts3(&v), &Value::Array(flat(&r["carpet"]["V"]).into_iter().map(|v| json!(v)).collect()), 1e-12, "carpet V");
    let ff: Vec<f64> = f.iter().flat_map(|q| q.map(|i| i as f64)).collect();
    close_v(&ff, &Value::Array(flat(&r["carpet"]["F"]).into_iter().map(|v| json!(v)).collect()), 0.0, "carpet F");
    let uu: Vec<f64> = uv.iter().flat_map(|q| [q.x, q.y]).collect();
    close_v(&uu, &Value::Array(flat(&r["carpet"]["UV"]).into_iter().map(|v| json!(v)).collect()), 1e-12, "carpet UV");
    close(placement::fit_carpet_height(cam, DVec3::new(53.3, 8.0, 0.0), 5.0, 1.1, p.length / 2.0 + 5.1), &r["fit"], 1e-12, "fit");
    for (k, (c, m)) in [(cam, 9.7), (DVec3::new(0.0, -60.85, 14.93), 9.7), (DVec3::new(0.0, -76.05, 23.05), 9.7), (DVec3::new(0.0, -60.85, 14.93), 7.0)].into_iter().enumerate() {
        close(placement::end_board_offset(c, &p, 5.0, m, placement::CARPET_HEIGHT), &r["end_offset"][k], 1e-12, "end board offset");
    }
    // carpets end up entirely behind the goal line, in front of the end boards
    for (k, (_, _, b)) in placement::carpet_spots(&p, 5.0, 0.8).iter().enumerate() {
        let (base, h) = placement::place_carpet(cam, *b, 5.0, 1.1, &p, p.length / 2.0 + 5.1);
        let x = &r["place"][k];
        close_v(&v3(base), &x[0], 1e-9, "carpet base");
        close(h, &x[1], 1e-9, "carpet height");
        let (v, _, _) = placement::carpet(cam, base, 5.0, h, 8, 4, &|_, _| 0.0);
        assert!(v.iter().all(|q| q.x.abs() >= p.length / 2.0 + 0.3 - 1e-6 && q.x.abs() <= p.length / 2.0 + 5.1 + 1e-6), "carpet {k} overlaps the pitch or the boards");
    }
    let tv: Vec<f64> = [(0.0, 38.0, (0.0, -1.0)), (0.0, -38.0, (0.0, 1.0)), (56.0, 0.0, (-1.0, 0.0))]
        .iter()
        .map(|(x, y, f)| placement::tv_weight(DVec3::new(*x, *y, 0.5), DVec3::new(f.0, f.1, 0.0), cam))
        .collect();
    close_v(&tv, &r["tv"], 1e-12, "tv");
}

#[test]
fn pitch_markings() {
    let c = cat();
    let r = reference("pitch");
    for (label, params) in [
        ("default", json!({})),
        ("sunday", json!({"length": 98, "width": 62, "optional_marks": false, "hand_painted": 1.0, "crown": 0.12})),
    ] {
        let p = pitch::Pitch::resolve(&params, Some(&c));
        let (lines, discs) = pitch::marking_paths(&p);
        let x = &r[label];
        let rl = x["lines"].as_array().unwrap();
        assert_eq!(lines.len(), rl.len(), "{label} lines");
        for (i, ((pts, closed), y)) in lines.iter().zip(rl).enumerate() {
            assert_eq!(*closed, y[1].as_bool().unwrap());
            close_v(&pts.iter().flat_map(|q| [q.x, q.y]).collect::<Vec<_>>(), &Value::Array(flat(&y[0]).into_iter().map(|v| json!(v)).collect()), 1e-12, &format!("{label} line {i}"));
            let wob = pitch::wobble(pts, *closed, 0.035 * p.hand_painted, (p.seed * 100 + i as i64) as u64);
            close_v(&wob.iter().flat_map(|q| [q.x, q.y]).collect::<Vec<_>>(), &Value::Array(flat(&x["wobbled"][i]).into_iter().map(|v| json!(v)).collect()), 1e-10, &format!("{label} wobble {i}"));
        }
        let dd: Vec<f64> = discs.iter().flat_map(|(c, r)| [c.x, c.y, *r]).collect();
        close_v(&dd, &Value::Array(flat(&x["discs"]).into_iter().map(|v| json!(v)).collect()), 1e-12, "discs");
        let cs: Vec<f64> = [(0.0, 0.0), (10.0, 20.0), (50.0, -40.0)].iter().map(|(a, b)| p.crown_z(*a, *b)).collect();
        close_v(&cs, &x["crown_samples"], 1e-12, "crown");
    }
}

#[test]
fn seat_patterns() {
    let r = reference("patterns");
    let mut anchors = Vec::new();
    for t in 0..2 {
        for row in 0..12 {
            for b in 0..2 {
                for c in 0..30 {
                    anchors.push(stands::Anchor { pos: DVec3::ZERO, yaw: 0.0, tier: t, row, t: 0.25 + 0.5 * c as f64, pitch: 0.5, bay: b, seat: 0, of: 1, run: 0, riser: 0.0 });
                }
            }
        }
    }
    let refs: Vec<&stands::Anchor> = anchors.iter().collect();
    let tier_rows = [12, 12];
    let specs = [
        json!({"pattern": "solid", "color": "#c8102e"}),
        json!({"pattern": "tiers", "colors": ["#c8102e", "#1d4ea8"]}),
        json!({"pattern": "rows", "colors": ["#c8102e", "#ffffff"], "every": 3}),
        json!({"pattern": "bays", "colors": ["#0b6e4f", "#f2f2ee"]}),
        json!({"pattern": "gradient", "colors": ["#000000", "#ffffff"]}),
        json!({"pattern": "random", "colors": ["#0b6e4f", "#f2f2ee", "#d4af37"], "weights": [6, 2, 1]}),
    ];
    let mut random = Vec::new();
    for spec in &specs {
        let name = spec["pattern"].as_str().unwrap();
        let cols = patterns::base_colors(&refs, spec, &tier_rows, 17).unwrap();
        close_v(&cols.iter().flat_map(|c| v3(*c)).collect::<Vec<_>>(), &Value::Array(flat(&r["base"][name]).into_iter().map(|v| json!(v)).collect()), 1e-12, name);
        if name == "random" {
            random = cols;
        }
    }
    let mut cols = patterns::base_colors(&refs, &json!({"pattern": "solid", "color": "#1d4ea8"}), &tier_rows, 0).unwrap();
    let painted = patterns::apply_mosaic(&mut cols, &anchors, &json!({"run": "south", "tier": 0, "text": "KIT FC", "color": "#ffffff"}), 0, 15.0, &tier_rows, None);
    assert_eq!(painted as u64, r["mosaic"]["painted"].as_u64().unwrap());
    close_v(&cols.iter().flat_map(|c| v3(*c)).collect::<Vec<_>>(), &Value::Array(flat(&r["mosaic"]["colors"]).into_iter().map(|v| json!(v)).collect()), 1e-12, "mosaic");
    let fin = patterns::finish(&random[..50], 0.4, 5);
    close_v(&fin.iter().flat_map(|c| [c.x, c.y, c.z, c.w]).collect::<Vec<_>>(), &Value::Array(flat(&r["finish"]).into_iter().map(|v| json!(v)).collect()), 1e-12, "finish");
    let bmp = patterns::text_bitmap("KIT FC 9");
    let bb: Vec<f64> = bmp.iter().flat_map(|row| row.iter().map(|b| if *b { 1.0 } else { 0.0 })).collect();
    close_v(&bb, &Value::Array(flat(&r["bitmap"]).into_iter().map(|v| json!(v)).collect()), 0.0, "bitmap");
    let crc: Vec<f64> = ["south", "NE"].iter().map(|s| prng::crc32(s) as f64).collect();
    close_v(&crc, &r["crc"], 0.0, "crc");
}

/// Whole layouts: capacities match the Python assembler's (recorded in the stadium exports).
#[test]
fn layout_capacities() {
    let c = cat();
    for (name, seated_total) in [("sunday_league", 802), ("classic", 19285), ("premier", 61192), ("mega", 152854)] {
        let layout: Value = serde_json::from_str(&std::fs::read_to_string(root().join("layouts").join(format!("{name}.json"))).unwrap()).unwrap();
        let st = stadium::build(&layout, &c).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(st.report.capacity.total, seated_total, "{name} capacity");
    }
}

/// With carpets on, the end boards move back far enough for ~1 m carpets, and stay in front of
/// the stands.
#[test]
fn carpets_get_room_behind_the_goal() {
    let c = cat();
    for name in ["premier", "mega", "classic"] {
        let mut layout: Value = serde_json::from_str(&std::fs::read_to_string(root().join("layouts").join(format!("{name}.json"))).unwrap()).unwrap();
        layout["ads"]["carpets"] = json!(true);
        let st = stadium::build(&layout, &c).unwrap();
        let carpets: Vec<&stadium::AdSlot> = st.report.ads.iter().filter(|s| s.kind == "carpet_3d").collect();
        assert_eq!(carpets.len(), 4, "{name}");
        for s in carpets {
            assert!(s.area_m2 >= 5.0 * stadium_gen::placement::CARPET_HEIGHT - 0.05, "{name}: {} m²", s.area_m2);
        }
        let front = st.report.footprint.0 - st.pitch.length / 2.0;
        assert!(st.report.end_boards_m > 5.5 && st.report.end_boards_m <= front - 0.8 + 1e-9, "{name}: end boards {} m, stand front {front} m", st.report.end_boards_m);
        // without carpets the boards stay at the 5.5 m default
        layout["ads"]["carpets"] = json!(false);
        assert_eq!(stadium::build(&layout, &c).unwrap().report.end_boards_m, 5.5);
    }
}
