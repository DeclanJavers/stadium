//! Stadium viewer: real-time preview of Stadium Kit glTF exports.
//!
//!   stadium-viewer <stadium.game.glb> [--night] [--camera broadcast|aerial|pitch|goal]
//!                  [--screenshot out.png] [--size 1600x900]
//!
//! Interactive: drag to orbit, right-drag to pan, scroll to zoom, 1-4 camera presets,
//! N day/night, Esc quits.

mod glb;
mod render;

use glam::{Mat4, Vec3};
use render::{sun_matrix, v4, Globals, Renderer, Targets};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// ACES runs hotter than Blender's AgX; this matches the review renders' mid-tones.
const EXPOSURE: f32 = 0.55;
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
            _ => match scene.main_camera {
                // the stadium's own camera gantry (front of the main stand's tier nearest 16 m)
                Some(p) => Camera::look(p, Vec3::new(0.0, 0.0, -6.0), 72.0),
                None => Camera::look(Vec3::new(0.0, 24.0, 66.0), Vec3::new(0.0, 0.0, -4.0), 62.0),
            },
        }
    }
}

fn globals(scene: &glb::Scene, cam: &Camera, aspect: f32, night: bool, grid: ([f32; 4], [f32; 4])) -> Globals {
    let eye = cam.eye();
    let view = Mat4::look_at_rh(eye, cam.target, Vec3::Y);
    let proj = Mat4::perspective_rh(cam.fov, aspect, 0.3, 5000.0);
    let vp = proj * view;
    // sun 38 deg up from the south-west (matches the Blender review renders)
    let (el, az) = (38f32.to_radians(), 215f32.to_radians());
    let sun_b = Vec3::new(el.cos() * az.sin(), -el.cos() * az.cos(), el.sin()); // Blender coords
    let sun = Vec3::new(sun_b.x, sun_b.z, -sun_b.y);
    let (lo, hi) = scene.bounds;
    let pitch_id = scene.pitch.as_ref().map(|p| match p.pattern.as_str() {
        "stripes" => 1.0, "stripes_long" => 2.0, "checkerboard" => 3.0, "diagonal" => 4.0, "diamond" => 5.0, "circles" => 6.0, _ => 0.0,
    }).unwrap_or(0.0);
    let (pw, pa, ps, nat) = scene.pitch.as_ref().map(|p| (p.width, p.angle.to_radians(), p.strength, p.natural as u8 as f32)).unwrap_or((5.0, 0.0, 0.0, 1.0));
    let (sun_e, zen, hor, gnd) = if night {
        (0.0, Vec3::new(0.0008, 0.0012, 0.003), Vec3::new(0.003, 0.003, 0.004), Vec3::new(0.002, 0.002, 0.002))
    } else {
        (4.5, Vec3::new(0.17, 0.25, 0.42), Vec3::new(0.42, 0.45, 0.48), Vec3::new(0.14, 0.15, 0.10))
    };
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
        params: [1.0, if night { (scene.lights.len().min(max_lights())) as f32 } else { 0.0 }, night as u8 as f32, (!night) as u8 as f32],
        pitch: [pw, pa, ps, pitch_id],
        pitch2: [nat, 0.0, 0.0, 0.0],
        grid_origin: if std::env::args().any(|x| x == "--exact-lights") { [0.0; 4] } else { grid.0 },
        grid_size: grid.1,
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
    let size = get("--size").and_then(|s| {
        let (w, h) = s.split_once('x')?;
        Some((w.parse().ok()?, h.parse().ok()?))
    }).unwrap_or((1600, 900));
    Args {
        path: PathBuf::from(a.iter().find(|x| !x.starts_with("--") && x.ends_with(".glb")).cloned().unwrap_or_else(|| {
            eprintln!("usage: stadium-viewer <stadium.glb> [--night] [--camera broadcast|aerial|pitch|goal] [--screenshot out.png] [--size WxH]");
            std::process::exit(2)
        })),
        night: a.iter().any(|x| x == "--night"),
        camera: get("--camera").unwrap_or_else(|| "broadcast".into()),
        screenshot: get("--screenshot").map(PathBuf::from),
        size,
    }
}

async fn gpu(surface: Option<&wgpu::Surface<'static>>, instance: &wgpu::Instance) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: surface, force_fallback_adapter: false })
        .await
        .expect("no GPU adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor { label: None, required_features: wgpu::Features::empty(), required_limits: adapter.limits(), memory_hints: Default::default() }, None)
        .await
        .expect("device");
    (adapter, device, queue)
}

fn screenshot(a: &Args, scene: &glb::Scene) {
    let instance = wgpu::Instance::default();
    let (_adapter, device, queue) = pollster::block_on(gpu(None, &instance));
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let t0 = Instant::now();
    let mut r = Renderer::new(&device, &queue, scene, fmt);
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
    let g = globals(scene, &cam, w as f32 / h as f32, a.night, r.grid);
    let (near, far) = r.update_lods(&queue, cam.eye(), LOD_NEAR);
    let exposure = EXPOSURE;
    r.render(&device, &queue, &targets, &view, &g, exposure);
    // time a few frames for an honest performance number
    let tf = Instant::now();
    for _ in 0..5 {
        r.render(&device, &queue, &targets, &view, &g, exposure);
    }
    device.poll(wgpu::Maintain::Wait);
    let frame_ms = tf.elapsed().as_secs_f32() * 1000.0 / 5.0;
    let row = (4 * w).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture { texture: &out, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::ImageCopyBuffer { buffer: &buf, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = buf.slice(..).get_mapped_range();
    let mut img = image::RgbaImage::new(w, h);
    for y in 0..h {
        let src = &data[(y * row) as usize..(y * row + 4 * w) as usize];
        img.as_mut()[(y * 4 * w) as usize..((y + 1) * 4 * w) as usize].copy_from_slice(src);
    }
    let path = a.screenshot.as_ref().unwrap();
    img.save(path).expect("save screenshot");
    println!("[viewer] {} ({}x{}, {} camera, {}): setup {:.1}s, {:.1} ms/frame, seats {} fine / {} coarse",
             path.display(), w, h, a.camera, if a.night { "night" } else { "day" }, t0.elapsed().as_secs_f32() - frame_ms * 0.005, frame_ms, near, far);
}

// ---------------------------------------------------------------------------------------------
// interactive window

struct App {
    args: Args,
    scene: glb::Scene,
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
    night: bool,
    drag: Option<(winit::event::MouseButton, (f64, f64))>,
    last: Instant,
    frames: u32,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(el.create_window(winit::window::Window::default_attributes()
            .with_title("Stadium Kit viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(self.args.size.0, self.args.size.1))).unwrap());
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let (adapter, device, queue) = pollster::block_on(gpu(Some(&surface), &instance));
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let renderer = Renderer::new(&device, &queue, &self.scene, fmt);
        let targets = renderer.targets(&device, config.width, config.height);
        let cam = Camera::preset(&self.args.camera, &self.scene);
        self.state = Some(Live { window, surface, device, queue, config, renderer, targets, cam, night: self.args.night, drag: None, last: Instant::now(), frames: 0 });
    }

    fn window_event(&mut self, el: &winit::event_loop::ActiveEventLoop, _id: winit::window::WindowId, event: winit::event::WindowEvent) {
        use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
        use winit::keyboard::{Key, NamedKey};
        let Some(s) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(sz) => {
                s.config.width = sz.width.max(1);
                s.config.height = sz.height.max(1);
                s.surface.configure(&s.device, &s.config);
                s.targets = s.renderer.targets(&s.device, s.config.width, s.config.height);
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => match event.logical_key {
                Key::Named(NamedKey::Escape) => el.exit(),
                Key::Character(c) => match c.as_str() {
                    "1" => s.cam = Camera::preset("broadcast", &self.scene),
                    "2" => s.cam = Camera::preset("aerial", &self.scene),
                    "3" => s.cam = Camera::preset("pitch", &self.scene),
                    "4" => s.cam = Camera::preset("goal", &self.scene),
                    "n" | "N" => s.night = !s.night,
                    _ => {}
                },
                _ => {}
            },
            WindowEvent::MouseInput { state, button, .. } => {
                s.drag = if state == ElementState::Pressed { Some((button, (f64::NAN, f64::NAN))) } else { None };
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
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta { MouseScrollDelta::LineDelta(_, y) => y * 0.1, MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.002 };
                s.cam.dist = (s.cam.dist * (1.0 - d)).clamp(3.0, 2000.0);
            }
            WindowEvent::RedrawRequested => {
                let frame = match s.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(_) => {
                        s.surface.configure(&s.device, &s.config);
                        return;
                    }
                };
                let view = frame.texture.create_view(&Default::default());
                let g = globals(&self.scene, &s.cam, s.config.width as f32 / s.config.height as f32, s.night, s.renderer.grid);
                s.renderer.update_lods(&s.queue, s.cam.eye(), LOD_NEAR);
                s.renderer.render(&s.device, &s.queue, &s.targets, &view, &g, EXPOSURE);
                frame.present();
                s.frames += 1;
                if s.last.elapsed().as_secs_f32() > 1.0 {
                    let fps = s.frames as f32 / s.last.elapsed().as_secs_f32();
                    let cap = self.scene.capacity.map(|c| format!(" · capacity {c}")).unwrap_or_default();
                    s.window.set_title(&format!("Stadium Kit viewer · {}{} · {:.0} fps · 1-4 cameras, N night", self.args.path.file_name().unwrap().to_string_lossy(), cap, fps));
                    s.last = Instant::now();
                    s.frames = 0;
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &winit::event_loop::ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

fn main() {
    let a = args();
    let t = Instant::now();
    let scene = glb::load(&a.path).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1)
    });
    let tris: usize = scene.meshes.iter().enumerate().map(|(i, p)| p.iter().map(|q| q.indices.len() / 3).sum::<usize>() * scene.instances[i].len()).sum();
    println!("[viewer] loaded {} in {:.1}s: {} meshes, {} materials, {} images, {} lights, {:.1}M triangles drawn",
             a.path.display(), t.elapsed().as_secs_f32(), scene.meshes.len(), scene.materials.len(), scene.images.len(), scene.lights.len(), tris as f64 / 1e6);
    if a.screenshot.is_some() {
        screenshot(&a, &scene);
        return;
    }
    let el = winit::event_loop::EventLoop::new().unwrap();
    let mut app = App { args: a, scene, state: None };
    el.run_app(&mut app).unwrap();
}
