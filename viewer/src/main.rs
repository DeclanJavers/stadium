//! Stadium Kit viewer and live stadium editor.
//!
//!   stadium-viewer <stadium.game.glb>        view a Blender game export
//!   stadium-viewer <layout.json>             edit a layout live (stands, seats, roofs, ads, …)
//!   stadium-viewer                           the editor with layouts/premier.json
//!
//! Options: --night, --camera broadcast|aerial|pitch|goal, --screenshot out.png (headless),
//! --size 1600x900, --exact-lights, --max-lights N.
//! Interactive: drag to orbit, right-drag to pan, scroll to zoom, 1-4 camera presets, N day/night.

mod build;
mod editor;
mod glb;
mod library;
mod render;

use glam::{Mat4, Vec3};
use render::{sun_matrix, v4, Globals, Renderer, Targets};
use serde_json::Value;
use stadium_gen::catalog::Catalog;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::time::Instant;

/// AgX at Blender exposure 0 (the review renders), in the kit's light units.
const EXPOSURE: f32 = 1.0;
/// Seats closer than this use the fine mesh; the rest use the MSFT_lod coarse mesh.
const LOD_NEAR: f32 = 45.0;

#[derive(Clone, Copy)]
struct Camera {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    dist: f32,
    fov: f32,
}

impl Camera {
    fn eye(&self) -> Vec3 {
        self.target + Vec3::new(self.yaw.cos() * self.pitch.cos(), self.pitch.sin(), self.yaw.sin() * self.pitch.cos()) * self.dist
    }

    fn look(eye: Vec3, target: Vec3, fov_deg: f32) -> Self {
        let d = eye - target;
        Camera { target, yaw: d.z.atan2(d.x), pitch: (d.y / d.length()).asin(), dist: d.length(), fov: fov_deg.to_radians() }
    }

    /// Presets in glTF coordinates (Y up; the kit's Blender -Y south side is glTF +Z).
    fn preset(name: &str, scene: &glb::Scene) -> Self {
        let (lo, hi) = scene.bounds;
        let r = ((hi - lo).length() * 0.5).max(60.0);
        match name {
            "aerial" => Camera::look(Vec3::new(-r * 0.75, r * 0.7, r * 0.95), Vec3::new(0.0, 0.0, 0.0), 40.0),
            "pitch" => Camera::look(Vec3::new(-28.0, 1.7, 18.0), Vec3::new(30.0, 4.0, -10.0), 60.0),
            "goal" => Camera::look(Vec3::new(-80.0, 14.0, 6.0), Vec3::new(0.0, 0.0, 0.0), 50.0),
            "grass" => Camera::look(Vec3::new(-8.0, 1.6, 6.0), Vec3::new(10.0, 0.0, -4.0), 55.0),
            _ => match scene.main_camera {
                // the stadium's own camera gantry (front of the main stand's tier nearest 16 m)
                Some(p) => Camera::look(p, Vec3::new(0.0, 0.0, -6.0), 72.0),
                None => Camera::look(Vec3::new(0.0, 24.0, 66.0), Vec3::new(0.0, 0.0, -4.0), 62.0),
            },
        }
    }
}

fn globals(scene: &glb::Scene, cam: &Camera, size: (u32, u32), night: bool, r: &Renderer, time: f32) -> Globals {
    let aspect = size.0 as f32 / size.1.max(1) as f32;
    let eye = cam.eye();
    let view = Mat4::look_at_rh(eye, cam.target, Vec3::Y);
    let proj = Mat4::perspective_rh(cam.fov, aspect, 0.3, 5000.0);
    let vp = proj * view;
    // sun 38 deg up from the south-west (matches the Blender review renders)
    let (el, az) = (38f32.to_radians(), 215f32.to_radians());
    let sun_b = Vec3::new(el.cos() * az.sin(), -el.cos() * az.cos(), el.sin()); // Blender coords
    let sun = Vec3::new(sun_b.x, sun_b.z, -sun_b.y);
    let (lo, hi) = scene.bounds;
    let pitch_id = scene
        .pitch
        .as_ref()
        .map(|p| match p.pattern.as_str() {
            "stripes" => 1.0,
            "stripes_long" => 2.0,
            "checkerboard" => 3.0,
            "diagonal" => 4.0,
            "diamond" => 5.0,
            "circles" => 6.0,
            _ => 0.0,
        })
        .unwrap_or(0.0);
    let (pw, pa, ps, nat) = scene.pitch.as_ref().map(|p| (p.width, p.angle.to_radians(), p.strength, p.natural as u8 as f32)).unwrap_or((5.0, 0.0, 0.0, 1.0));
    let (sun_e, zen, hor, gnd) = if night {
        (0.0, Vec3::new(0.0008, 0.0012, 0.003), Vec3::new(0.003, 0.003, 0.004), Vec3::new(0.002, 0.002, 0.002))
    } else {
        // ground: the bounce from the bowl floor (sunlit and shaded turf, track, concrete): a
        // green-grey light on the undersides of roofs and the shaded tiers
        (4.5, Vec3::new(0.12, 0.23, 0.50), Vec3::new(0.40, 0.45, 0.52), Vec3::new(0.15, 0.19, 0.10))
    };
    let grass = scene.pitch.as_ref();
    Globals {
        view_proj: vp.to_cols_array_2d(),
        inv_view_proj: vp.inverse().to_cols_array_2d(),
        light_view_proj: sun_matrix(sun, lo, hi).to_cols_array_2d(),
        cam_pos: v4(eye, 1.0),
        sun_dir: v4(sun, sun_e),
        sun_color: [1.0, 0.97, 0.92, 1.0],
        sky_zenith: v4(zen, 0.0),
        sky_horizon: v4(hor, 0.0),
        ground: v4(gnd, 0.0),
        params: [1.0, if night { scene.lights.len().min(max_lights()) as f32 } else { 0.0 }, night as u8 as f32, (!night) as u8 as f32],
        pitch: [pw, pa, ps, pitch_id],
        pitch2: [nat, 0.0, 0.0, 0.0],
        grid_origin: if std::env::args().any(|x| x == "--exact-lights") { [0.0; 4] } else { r.grid.0 },
        grid_size: r.grid.1,
        grass_color: grass.map(|g| v4(g.color, g.wear)).unwrap_or([0.06, 0.12, 0.02, 0.25]),
        grass_dry: grass.map(|g| v4(g.dry, g.wetness)).unwrap_or([0.25, 0.22, 0.07, 0.0]),
        grass_dims: grass.map(|g| [g.half.x, g.half.y, g.height_mm / 1000.0, g.procedural as u8 as f32]).unwrap_or([52.5, 34.0, 0.025, 0.0]),
        sky_origin: r.sky.0,
        sky_size: r.sky.1,
        screen: [size.0 as f32, size.1 as f32, 0.0, time],
    }
}

fn max_lights() -> usize {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == "--max-lights").and_then(|i| a.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(512)
}

struct Args {
    path: PathBuf,
    night: bool,
    camera: String,
    screenshot: Option<PathBuf>,
    size: (u32, u32),
}

fn args() -> Args {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let size = get("--size")
        .and_then(|s| {
            let (w, h) = s.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .unwrap_or((1600, 900));
    let flag_values = ["--camera", "--screenshot", "--size", "--max-lights"];
    let positional = a.iter().enumerate().find(|(i, x)| !x.starts_with("--") && !(*i > 0 && flag_values.contains(&a[i - 1].as_str()))).map(|(_, x)| PathBuf::from(x));
    let path = positional.unwrap_or_else(|| repo_root(&std::env::current_dir().unwrap()).map(|r| r.join("layouts/premier.json")).unwrap_or_else(|| PathBuf::from("layouts/premier.json")));
    Args { path, night: a.iter().any(|x| x == "--night"), camera: get("--camera").unwrap_or_else(|| "broadcast".into()), screenshot: get("--screenshot").map(PathBuf::from), size }
}

/// The repository root (the directory holding catalog/), searched upwards from `start`, the
/// executable and the current directory.
fn repo_root(start: &Path) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    let cwd = std::env::current_dir().ok();
    let starts = [Some(start.to_path_buf()), exe, cwd];
    starts.iter().flatten().find_map(|s| Catalog::find_dir(s)).map(|c| c.parent().unwrap().to_path_buf())
}

async fn gpu(surface: Option<&wgpu::Surface<'static>>, instance: &wgpu::Instance) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: surface, force_fallback_adapter: false, apply_limit_buckets: false })
        .await
        .expect("no GPU adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            experimental_features: Default::default(),
            memory_hints: Default::default(),
            trace: Default::default(),
        })
        .await
        .expect("device");
    (adapter, device, queue)
}

fn is_layout(p: &Path) -> bool {
    p.extension().is_some_and(|e| e == "json")
}

/// Generate a stadium from a layout and turn it into a render scene.
fn build_layout(layout: &Value, cat: &Catalog, lib: &library::Library) -> Result<(stadium_gen::stadium::Report, glb::Scene, Vec<String>, f32, f32), String> {
    let t = Instant::now();
    let st = stadium_gen::stadium::build(layout, cat)?;
    let gen_ms = t.elapsed().as_secs_f32() * 1000.0;
    let t = Instant::now();
    let (scene, warnings) = build::scene(&st, lib);
    Ok((st.report, scene, warnings, gen_ms, t.elapsed().as_secs_f32() * 1000.0))
}

fn load_scene(a: &Args) -> Result<(glb::Scene, Option<(stadium_gen::stadium::Report, Value, Arc<Catalog>, PathBuf)>), String> {
    if !is_layout(&a.path) {
        return Ok((glb::load(&a.path)?, None));
    }
    let root = repo_root(&a.path).ok_or("cannot find the repository (catalog/) for the layout")?;
    let cat = Arc::new(Catalog::load(&root.join("catalog"))?);
    let lib = library::Library::load(&root)?;
    let mut layout: Value = serde_json::from_str(&std::fs::read_to_string(&a.path).map_err(|e| format!("{}: {e}", a.path.display()))?).map_err(|e| e.to_string())?;
    if a.night {
        layout["lighting"] = Value::from("night");
    }
    let (report, scene, warnings, g, s) = build_layout(&layout, &cat, &lib)?;
    for w in warnings.iter().chain(&report.warnings) {
        eprintln!("[editor] warning: {w}");
    }
    println!("[editor] {}: capacity {} ({} seated, {} standing), generate {g:.0} ms, scene {s:.0} ms", a.path.display(), report.capacity.total, report.capacity.seated, report.capacity.standing);
    Ok((scene, Some((report, layout, cat, root))))
}

/// Draws an overlay into the screenshot (the editor UI for `--ui`).
type Overlay<'a> = &'a mut dyn FnMut(&wgpu::Device, &wgpu::Queue, &mut wgpu::CommandEncoder, &wgpu::TextureView, (u32, u32), wgpu::TextureFormat);

fn screenshot(a: &Args, scene: &glb::Scene, mut overlay: Option<Overlay>) {
    let instance = wgpu::Instance::default();
    let (_adapter, device, queue) = pollster::block_on(gpu(None, &instance));
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let t0 = Instant::now();
    let mut r = Renderer::new(&device, &queue, fmt);
    r.set_scene(&device, &queue, scene);
    let (w, h) = a.size;
    let targets = r.targets(&device, w, h);
    let out = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = out.create_view(&Default::default());
    let cam = Camera::preset(&a.camera, scene);
    let g = globals(scene, &cam, (w, h), a.night, &r, 0.0);
    let (near, far) = r.update_lods(&queue, cam.eye(), LOD_NEAR);
    let frame = |r: &Renderer| {
        let mut enc = device.create_command_encoder(&Default::default());
        r.render(&queue, &mut enc, &targets, &view, &g, EXPOSURE);
        queue.submit([enc.finish()]);
    };
    frame(&r);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let setup = t0.elapsed().as_secs_f32();
    if let Some(ov) = overlay.as_mut() {
        let mut enc = device.create_command_encoder(&Default::default());
        ov(&device, &queue, &mut enc, &view, (w, h), fmt);
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    // time a few frames for an honest performance number
    let tf = Instant::now();
    if overlay.is_none() {
        for _ in 0..5 {
            frame(&r);
        }
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let frame_ms = tf.elapsed().as_secs_f32() * 1000.0 / 5.0;
    let row = (4 * w).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &out, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = buf.slice(..).get_mapped_range().expect("map screenshot buffer");
    let mut img = image::RgbaImage::new(w, h);
    for y in 0..h {
        let src = &data[(y * row) as usize..(y * row + 4 * w) as usize];
        img.as_mut()[(y * 4 * w) as usize..((y + 1) * 4 * w) as usize].copy_from_slice(src);
    }
    let path = a.screenshot.as_ref().unwrap();
    img.save(path).expect("save screenshot");
    println!(
        "[viewer] {} ({}x{}, {} camera, {}): setup {:.1}s, {:.1} ms/frame, seats {} fine / {} coarse",
        path.display(),
        w,
        h,
        a.camera,
        if a.night { "night" } else { "day" },
        setup,
        frame_ms,
        near,
        far
    );
}

/// Read back a texture (RGBA8 or BGRA8) and save it as PNG.
fn save_texture(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, size: (u32, u32), path: &Path) {
    let (w, h) = size;
    let row = (4 * w).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = buf.slice(..).get_mapped_range().expect("map");
    let bgra = matches!(tex.format(), wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
    let mut img = image::RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let o = (y * row + 4 * x) as usize;
            let p = &data[o..o + 4];
            img.put_pixel(x, y, if bgra { image::Rgba([p[2], p[1], p[0], 255]) } else { image::Rgba([p[0], p[1], p[2], 255]) });
        }
    }
    let _ = img.save(path);
}

// ---------------------------------------------------------------------------------------------
// interactive window

type BuildResult = (u64, Result<(stadium_gen::stadium::Report, glb::Scene, Vec<String>, f32, f32), String>);

/// Editor state: the layout, a background builder and its results.
struct Edit {
    editor: editor::Editor,
    jobs: mpsc::Sender<(u64, Value)>,
    done: mpsc::Receiver<BuildResult>,
    sent: u64,
    shown: u64,
    changed_at: Instant,
    pending_rev: u64,
}

struct App {
    args: Args,
    scene: Option<glb::Scene>,
    edit: Option<Edit>,
    state: Option<Live>,
}

struct Live {
    window: Arc<winit::window::Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    targets: Targets,
    cam: Camera,
    cam_set: bool,
    night: bool,
    drag: Option<(winit::event::MouseButton, (f64, f64))>,
    last: Instant,
    start: Instant,
    frames: u32,
    snapshot_done: bool,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

impl App {
    fn new(args: Args) -> Result<App, String> {
        if !is_layout(&args.path) {
            let t = Instant::now();
            let scene = glb::load(&args.path)?;
            println!("[viewer] loaded {} in {:.1}s: {} meshes, {} materials, {} images, {} lights", args.path.display(), t.elapsed().as_secs_f32(), scene.meshes.len(), scene.materials.len(), scene.images.len(), scene.lights.len());
            return Ok(App { args, scene: Some(scene), edit: None, state: None });
        }
        let root = repo_root(&args.path).ok_or("cannot find the repository (catalog/)")?;
        let cat = Arc::new(Catalog::load(&root.join("catalog"))?);
        let lib = Arc::new(library::Library::load(&root)?);
        let mut layout: Value = match std::fs::read_to_string(&args.path) {
            Ok(t) => serde_json::from_str(&t).map_err(|e| format!("{}: {e}", args.path.display()))?,
            Err(_) => serde_json::json!({"schema": "stadium-layout/1", "name": "New stadium"}),
        };
        if args.night {
            layout["lighting"] = Value::from("night");
        }
        let (jobs, job_rx) = mpsc::channel::<(u64, Value)>();
        let (done_tx, done) = mpsc::channel::<BuildResult>();
        let (wc, wl) = (cat.clone(), lib.clone());
        std::thread::spawn(move || {
            while let Ok(mut job) = job_rx.recv() {
                while let Ok(newer) = job_rx.try_recv() {
                    job = newer; // only the latest layout matters
                }
                let r = build_layout(&job.1, &wc, &wl);
                if done_tx.send((job.0, r)).is_err() {
                    break;
                }
            }
        });
        let path = args.path.exists().then(|| args.path.clone());
        let mut ed = editor::Editor::new(layout, cat, path, root.join("layouts"));
        ed.saved_revision = ed.revision;
        Ok(App { args, scene: None, edit: Some(Edit { editor: ed, jobs, done, sent: 0, shown: 0, changed_at: Instant::now(), pending_rev: 0 }), state: None })
    }
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let title = if self.edit.is_some() { "Stadium Kit editor" } else { "Stadium Kit viewer" };
        let window = Arc::new(el.create_window(winit::window::Window::default_attributes().with_title(title).with_inner_size(winit::dpi::LogicalSize::new(self.args.size.0, self.args.size.1))).unwrap());
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let (adapter, device, queue) = pollster::block_on(gpu(Some(&surface), &instance));
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let snapshot = std::env::var_os("STADIUM_SNAPSHOT").is_some();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | if snapshot { wgpu::TextureUsages::COPY_SRC } else { wgpu::TextureUsages::empty() },
            format: fmt,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: Default::default(),
        };
        surface.configure(&device, &config);
        let mut renderer = Renderer::new(&device, &queue, fmt);
        let mut cam = Camera::look(Vec3::new(0.0, 40.0, 110.0), Vec3::ZERO, 60.0);
        let mut cam_set = false;
        if let Some(scene) = &self.scene {
            renderer.set_scene(&device, &queue, scene);
            cam = Camera::preset(&self.args.camera, scene);
            cam_set = true;
        }
        let targets = renderer.targets(&device, config.width, config.height);
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, Some(device.limits().max_texture_dimension_2d as usize));
        let egui_renderer = egui_wgpu::Renderer::new(&device, fmt, egui_wgpu::RendererOptions::default());
        self.state = Some(Live {
            window,
            surface,
            device,
            queue,
            config,
            renderer,
            targets,
            cam,
            cam_set,
            night: self.args.night,
            drag: None,
            last: Instant::now(),
            start: Instant::now(),
            frames: 0,
            snapshot_done: false,
            egui_ctx,
            egui_state,
            egui_renderer,
        });
    }

    fn window_event(&mut self, el: &winit::event_loop::ActiveEventLoop, _id: winit::window::WindowId, event: winit::event::WindowEvent) {
        use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
        use winit::keyboard::{Key, NamedKey};
        let Some(s) = self.state.as_mut() else { return };
        let resp = s.egui_state.on_window_event(&s.window, &event);
        let ui_has_pointer = s.egui_ctx.is_pointer_over_egui() || s.egui_ctx.egui_is_using_pointer();
        let ui_has_keys = s.egui_ctx.egui_wants_keyboard_input();
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(sz) => {
                s.config.width = sz.width.max(1);
                s.config.height = sz.height.max(1);
                s.surface.configure(&s.device, &s.config);
                s.targets = s.renderer.targets(&s.device, s.config.width, s.config.height);
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed && !ui_has_keys => match event.logical_key {
                Key::Named(NamedKey::Escape) if self.edit.is_none() => el.exit(),
                Key::Character(c) => {
                    let preset = match c.as_str() {
                        "1" => Some("broadcast"),
                        "2" => Some("aerial"),
                        "3" => Some("pitch"),
                        "4" => Some("goal"),
                        "5" => Some("grass"),
                        _ => None,
                    };
                    if let Some(p) = preset {
                        if let Some(e) = self.edit.as_mut() {
                            e.editor.camera_request = Some(p);
                        } else if let Some(sc) = &self.scene {
                            s.cam = Camera::preset(p, sc);
                        }
                    }
                    if c.as_str() == "n" || c.as_str() == "N" {
                        match self.edit.as_mut() {
                            Some(e) => {
                                let night = !e.editor.night();
                                e.editor.layout["lighting"] = Value::from(if night { "night" } else { "day" });
                                e.editor.revision += 1;
                            }
                            None => s.night = !s.night,
                        }
                    }
                }
                _ => {}
            },
            WindowEvent::MouseInput { state, button, .. } => {
                s.drag = if state == ElementState::Pressed && !ui_has_pointer { Some((button, (f64::NAN, f64::NAN))) } else { None };
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some((button, last)) = s.drag.as_mut() {
                    if !last.0.is_nan() {
                        let (dx, dy) = ((position.x - last.0) as f32, (position.y - last.1) as f32);
                        if *button == winit::event::MouseButton::Left {
                            s.cam.yaw += dx * 0.005;
                            s.cam.pitch = (s.cam.pitch + dy * 0.005).clamp(-0.2, 1.5);
                        } else {
                            let right = Vec3::new(-s.cam.yaw.sin(), 0.0, s.cam.yaw.cos());
                            let fwd = Vec3::new(s.cam.yaw.cos(), 0.0, s.cam.yaw.sin());
                            s.cam.target += (right * -dx + fwd * -dy) * s.cam.dist * 0.0015;
                        }
                    }
                    *last = (position.x, position.y);
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !resp.consumed && !ui_has_pointer => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.002,
                };
                s.cam.dist = (s.cam.dist * (1.0 - d)).clamp(1.0, 2000.0);
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &winit::event_loop::ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

impl App {
    fn redraw(&mut self) {
        let Some(s) = self.state.as_mut() else { return };
        // editor: send layout changes to the builder (debounced), take finished builds
        if let Some(e) = self.edit.as_mut() {
            if e.editor.revision != e.pending_rev {
                e.pending_rev = e.editor.revision;
                e.changed_at = Instant::now();
            }
            if e.pending_rev != e.sent && e.changed_at.elapsed().as_millis() > 120 {
                e.sent = e.pending_rev;
                let _ = e.jobs.send((e.sent, e.editor.layout.clone()));
                e.editor.status = "building…".into();
            }
            while let Ok((rev, result)) = e.done.try_recv() {
                if rev < e.shown {
                    continue;
                }
                e.shown = rev;
                match result {
                    Ok((report, scene, warnings, g, sc)) => {
                        let t = Instant::now();
                        s.renderer.set_scene(&s.device, &s.queue, &scene);
                        let up = t.elapsed().as_secs_f32() * 1000.0;
                        e.editor.build_ms = (g, sc, up);
                        e.editor.error = if warnings.is_empty() { None } else { Some(warnings.join("; ")) };
                        e.editor.report = Some(report);
                        e.editor.status = format!("built in {:.0} ms", g + sc + up);
                        if !s.cam_set {
                            s.cam = Camera::preset(&self.args.camera, &scene);
                            s.cam_set = true;
                        }
                        self.scene = Some(scene);
                    }
                    Err(err) => {
                        e.editor.error = Some(err);
                        e.editor.status = "build failed".into();
                    }
                }
            }
            if let (Some(p), Some(sc)) = (e.editor.camera_request.take(), &self.scene) {
                s.cam = Camera::preset(p, sc);
            }
            s.night = e.editor.night();
        }
        let frame = match s.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                s.surface.configure(&s.device, &s.config);
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let size = (s.config.width, s.config.height);
        let mut enc = s.device.create_command_encoder(&Default::default());
        let exposure = self.edit.as_ref().map(|e| e.editor.exposure).unwrap_or(EXPOSURE);
        if let Some(scene) = &self.scene {
            let g = globals(scene, &s.cam, size, s.night, &s.renderer, s.start.elapsed().as_secs_f32());
            s.renderer.update_lods(&s.queue, s.cam.eye(), LOD_NEAR);
            s.renderer.render(&s.queue, &mut enc, &s.targets, &view, &g, exposure);
        }
        // UI
        let raw = s.egui_state.take_egui_input(&s.window);
        let edit = &mut self.edit;
        let scene_ref = &self.scene;
        let full = s.egui_ctx.run_ui(raw, |ui| match edit.as_mut() {
            Some(e) => {
                egui::Panel::top("top").show(ui, |ui| e.editor.top_bar(ui));
                egui::Panel::left("layout").resizable(true).default_size(330.0).show(ui, |ui| e.editor.controls(ui));
                egui::Panel::right("readouts").resizable(true).default_size(300.0).show(ui, |ui| e.editor.readouts(ui));
            }
            None => {
                egui::Panel::bottom("help").show(ui, |ui| {
                    let cap = scene_ref.as_ref().and_then(|s| s.capacity).map(|c| format!(" · capacity {c}")).unwrap_or_default();
                    ui.weak(format!("drag to orbit · right-drag to pan · scroll to zoom · 1–5 cameras · N day/night{cap}"));
                });
            }
        });
        s.egui_state.handle_platform_output(&s.window, full.platform_output);
        let tris = s.egui_ctx.tessellate(full.shapes, full.pixels_per_point);
        for (id, deltas) in &full.textures_delta.set {
            for delta in deltas {
                s.egui_renderer.update_texture(&s.device, &s.queue, *id, delta);
            }
        }
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [size.0, size.1], pixels_per_point: full.pixels_per_point };
        let cmds = s.egui_renderer.update_buffers(&s.device, &s.queue, &mut enc, &tris, &screen);
        {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: if self.scene.is_some() { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.06, b: 0.07, a: 1.0 }) }, store: wgpu::StoreOp::Store } })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            s.egui_renderer.render(&mut pass, &tris, &screen);
        }
        s.queue.submit(cmds.into_iter().chain([enc.finish()]));
        // STADIUM_SNAPSHOT=out.png: save the window once the stadium has been shown for a while
        if let Some(path) = std::env::var_os("STADIUM_SNAPSHOT") {
            let ready = self.scene.is_some() && s.start.elapsed().as_secs_f32() > 6.0;
            if ready && !s.snapshot_done {
                s.snapshot_done = true;
                save_texture(&s.device, &s.queue, &frame.texture, size, std::path::Path::new(&path));
                println!("[editor] snapshot {}", std::path::Path::new(&path).display());
            }
        }
        s.queue.present(frame);
        for id in &full.textures_delta.free {
            s.egui_renderer.free_texture(id);
        }
        s.frames += 1;
        if s.last.elapsed().as_secs_f32() > 1.0 {
            let fps = s.frames as f32 / s.last.elapsed().as_secs_f32();
            if let Some(e) = self.edit.as_mut() {
                e.editor.fps = fps;
                let name = e.editor.path.as_ref().map(|p| p.file_name().unwrap().to_string_lossy().to_string()).unwrap_or("untitled".into());
                s.window.set_title(&format!("Stadium Kit editor · {name}"));
            } else {
                let cap = self.scene.as_ref().and_then(|s| s.capacity).map(|c| format!(" · capacity {c}")).unwrap_or_default();
                s.window.set_title(&format!("Stadium Kit viewer · {}{} · {:.0} fps", self.args.path.file_name().unwrap().to_string_lossy(), cap, fps));
            }
            s.last = Instant::now();
            s.frames = 0;
        }
    }
}

fn main() {
    let a = args();
    if a.screenshot.is_some() {
        let t = Instant::now();
        let (scene, edit) = load_scene(&a).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1)
        });
        let tris: usize = scene.meshes.iter().enumerate().map(|(i, p)| p.iter().map(|q| q.indices.len() / 3).sum::<usize>() * scene.instances[i].len()).sum();
        println!(
            "[viewer] loaded {} in {:.1}s: {} meshes, {} materials, {} images, {} lights, {:.1}M triangles drawn",
            a.path.display(),
            t.elapsed().as_secs_f32(),
            scene.meshes.len(),
            scene.materials.len(),
            scene.images.len(),
            scene.lights.len(),
            tris as f64 / 1e6
        );
        match edit.filter(|_| std::env::args().any(|x| x == "--ui")) {
            Some((report, layout, cat, root)) => {
                let mut ed = editor::Editor::new(layout, cat, Some(a.path.clone()), root.join("layouts"));
                ed.report = Some(report);
                ed.expand_all = true;
                let mut ov = |device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, size: (u32, u32), fmt: wgpu::TextureFormat| {
                    let ctx = egui::Context::default();
                    let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(size.0 as f32, size.1 as f32))), ..Default::default() };
                    let mut er = egui_wgpu::Renderer::new(device, fmt, egui_wgpu::RendererOptions::default());
                    let mut full = None;
                    for _ in 0..3 {
                        let f = ctx.run_ui(raw.clone(), |ui| {
                            egui::Panel::top("top").show(ui, |ui| ed.top_bar(ui));
                            egui::Panel::left("layout").resizable(true).default_size(330.0).show(ui, |ui| ed.controls(ui));
                            egui::Panel::right("readouts").resizable(true).default_size(300.0).show(ui, |ui| ed.readouts(ui));
                        });
                        // every frame's texture updates (the font atlas arrives in the first)
                        for (id, deltas) in &f.textures_delta.set {
                            for d in deltas {
                                er.update_texture(device, queue, *id, d);
                            }
                        }
                        full = Some(f);
                    }
                    let full = full.unwrap();
                    let tris = ctx.tessellate(full.shapes, full.pixels_per_point);
                    let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [size.0, size.1], pixels_per_point: full.pixels_per_point };
                    let cmds = er.update_buffers(device, queue, enc, &tris, &screen);
                    queue.submit(cmds);
                    let mut pass = enc
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("egui"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        })
                        .forget_lifetime();
                    er.render(&mut pass, &tris, &screen);
                };
                screenshot(&a, &scene, Some(&mut ov));
            }
            None => screenshot(&a, &scene, None),
        }
        return;
    }
    let mut app = App::new(a).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1)
    });
    let el = winit::event_loop::EventLoop::new().unwrap();
    el.run_app(&mut app).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every example layout builds through the asset library: no missing materials or props, and
    /// the instanced seats are exactly the reported seated capacity plus the dugout seats.
    #[test]
    fn layouts_build_with_library() {
        let root = repo_root(Path::new(env!("CARGO_MANIFEST_DIR"))).unwrap();
        let cat = Catalog::load(&root.join("catalog")).unwrap();
        let Ok(lib) = library::Library::load(&root) else {
            eprintln!("skipped: out/library/library.game.glb missing (tools/export_library.py)");
            return;
        };
        let mut names: Vec<PathBuf> = std::fs::read_dir(root.join("layouts")).unwrap().map(|e| e.unwrap().path()).collect();
        names.sort();
        for p in names {
            let layout: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            let (report, scene, warnings, _, _) = build_layout(&layout, &cat, &lib).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            assert!(warnings.is_empty(), "{}: {warnings:?}", p.display());
            assert_eq!(scene.meshes.len(), scene.instances.len());
            let seats: usize = scene.instances.iter().map(|v| v.iter().filter(|i| i.color.w > 0.5).count()).sum();
            // two dugouts, two rows each
            let dug = layout.get("dugouts").filter(|d| d.is_object()).map(|d| 4 * d.get("seats").and_then(|v| v.as_u64()).unwrap_or(14).div_ceil(2)).unwrap_or(0) as usize;
            assert_eq!(seats, report.capacity.seated + dug, "{} seat instances", p.display());
        }
    }

    /// Saving a layout and opening it again gives the same document.
    #[test]
    fn save_and_open_round_trip() {
        let root = repo_root(Path::new(env!("CARGO_MANIFEST_DIR"))).unwrap();
        let cat = Arc::new(Catalog::load(&root.join("catalog")).unwrap());
        let layout: Value = serde_json::from_str(&std::fs::read_to_string(root.join("layouts/classic.json")).unwrap()).unwrap();
        let mut ed = editor::Editor::new(layout.clone(), cat, None, root.join("layouts"));
        let tmp = std::env::temp_dir().join(format!("stadium_roundtrip_{}.json", std::process::id()));
        ed.save(tmp.clone());
        ed.layout = Value::Null;
        ed.open(tmp.clone());
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(ed.layout, layout);
        assert_eq!(ed.revision, ed.saved_revision);
    }
}
