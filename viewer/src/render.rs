//! wgpu renderer: sun shadow map, 4x MSAA HDR main pass (sky, opaque/mask, blend), ACES tone map.

use crate::glb::{Alpha, Scene, TexRef};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MSAA: u32 = 4;
const SHADOW_SIZE: u32 = 8192;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub light_view_proj: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub ground: [f32; 4],
    pub params: [f32; 4],
    pub pitch: [f32; 4],
    pub pitch2: [f32; 4],
    pub grid_origin: [f32; 4],
    pub grid_size: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightU {
    pos: [f32; 4],
    dir: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialU {
    base_color: [f32; 4],
    emissive: [f32; 4],
    params: [f32; 4],
    flags: [f32; 4],
    flags2: [f32; 4],
    xf: [[f32; 4]; 4],
    rot: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    color: [f32; 4],
}

struct Draw {
    vbuf: std::rc::Rc<wgpu::Buffer>,
    ibuf: std::rc::Rc<wgpu::Buffer>,
    count: u32,
    inst: usize,
    material: usize,
}

struct InstBuf {
    buf: wgpu::Buffer,
    count: u32,
}

/// A mesh with an MSFT_lod coarse level: instances are split near/far every frame.
struct LodSet {
    all: Vec<InstanceRaw>,
    pos: Vec<Vec3>,
    near: usize,
    far: usize,
}

const MAX_LIGHTS: usize = 512;

pub struct Targets {
    msaa_color: wgpu::TextureView,
    msaa_depth: wgpu::TextureView,
    resolve: wgpu::TextureView,
    tonemap_bg: wgpu::BindGroup,
}

pub struct Renderer {
    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    globals_bg_shadow: wgpu::BindGroup,
    shadow_view: wgpu::TextureView,
    material_bgs: Vec<wgpu::BindGroup>,
    material_alpha: Vec<Alpha>,
    draws: Vec<Draw>,
    shadow_draws: Vec<Draw>,
    inst_bufs: Vec<InstBuf>,
    lod_sets: Vec<LodSet>,
    pipe_prepass: wgpu::RenderPipeline,
    pipe_opaque: wgpu::RenderPipeline,
    pipe_blend: wgpu::RenderPipeline,
    pipe_sky: wgpu::RenderPipeline,
    pipe_shadow: wgpu::RenderPipeline,
    pipe_tonemap: wgpu::RenderPipeline,
    tonemap_layout: wgpu::BindGroupLayout,
    exposure_buf: wgpu::Buffer,
    linear_sampler: wgpu::Sampler,
    /// (origin xyz + cell size, extent xyz) of the floodlight irradiance volume; cell 0 = none
    pub grid: ([f32; 4], [f32; 4]),
}

fn mips(img: &crate::glb::Image) -> Vec<(u32, u32, Vec<u8>)> {
    let mut out = vec![(img.width, img.height, img.rgba.clone())];
    let mut cur = image::RgbaImage::from_raw(img.width, img.height, img.rgba.clone()).unwrap();
    while cur.width() > 1 || cur.height() > 1 {
        let (w, h) = ((cur.width() / 2).max(1), (cur.height() / 2).max(1));
        cur = image::imageops::resize(&cur, w, h, image::imageops::FilterType::Triangle);
        out.push((w, h, cur.as_raw().clone()));
    }
    out
}

impl Renderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, output_format: wgpu::TextureFormat) -> Self {
        // output_format: the swapchain / screenshot format the tone-map pass writes
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stadium"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let tshader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tonemap"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tonemap.wgsl").into()),
        });

        // ---- textures (sRGB if used as colour, linear otherwise) with CPU mip chains
        let mut srgb = vec![false; scene.images.len()];
        for m in &scene.materials {
            for t in [m.base_tex, m.emissive_tex].into_iter().flatten() {
                srgb[t.image] = true;
            }
        }
        let max_dim = device.limits().max_texture_dimension_2d;
        let mut views = Vec::new();
        for (i, img) in scene.images.iter().enumerate() {
            let mut levels = mips(img);
            while levels[0].0 > max_dim || levels[0].1 > max_dim {
                levels.remove(0);
            }
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: levels[0].0, height: levels[0].1, depth_or_array_layers: 1 },
                mip_level_count: levels.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if srgb[i] { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm },
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            for (lvl, (w, h, data)) in levels.iter().enumerate() {
                queue.write_texture(
                    wgpu::ImageCopyTexture { texture: &tex, mip_level: lvl as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    data,
                    wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4 * w), rows_per_image: Some(*h) },
                    wgpu::Extent3d { width: *w, height: *h, depth_or_array_layers: 1 },
                );
            }
            views.push(tex.create_view(&Default::default()));
        }
        let solid = |rgba: [u8; 4], fmt| {
            let t = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: fmt,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &rgba,
            );
            t.create_view(&Default::default())
        };
        let white = solid([255, 255, 255, 255], wgpu::TextureFormat::Rgba8UnormSrgb);
        let flat = solid([128, 128, 255, 255], wgpu::TextureFormat::Rgba8Unorm);
        let sampler = |repeat: bool| {
            let mode = if repeat { wgpu::AddressMode::Repeat } else { wgpu::AddressMode::ClampToEdge };
            device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: mode,
                address_mode_v: mode,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                anisotropy_clamp: 16,
                ..Default::default()
            })
        };
        let s_repeat = sampler(true);
        let s_clamp = sampler(false);

        // ---- layouts
        let tex_entry = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let samp_entry = |b: u32| wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None };
        let mat_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                tex_entry(1), samp_entry(2), tex_entry(3), samp_entry(4), tex_entry(5), samp_entry(6), tex_entry(7), samp_entry(8),
            ],
        });
        let tex3 = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
            count: None,
        };
        let glob_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                tex3(4), tex3(5), tex3(6), tex3(7), tex3(8), tex3(9),
                wgpu::BindGroupLayoutEntry { binding: 10, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });

        // ---- materials
        let mut material_bgs = Vec::new();
        let mut material_alpha = Vec::new();
        for m in &scene.materials {
            let xf = |t: &Option<TexRef>| t.map(|t| [t.xform.offset.x, t.xform.offset.y, t.xform.scale.x, t.xform.scale.y]).unwrap_or([0.0, 0.0, 1.0, 1.0]);
            let rot = |t: &Option<TexRef>| t.map(|t| t.xform.rotation).unwrap_or(0.0);
            let is_grass = m.name == "Pitch_Grass";
            let u = MaterialU {
                base_color: m.base_color.to_array(),
                emissive: (m.emissive * m.emissive_strength).extend(0.0).to_array(),
                params: [m.metallic, m.roughness, m.cutoff, m.transmission],
                flags: [m.base_tex.is_some() as u8 as f32, m.mr_tex.is_some() as u8 as f32, m.normal_tex.is_some() as u8 as f32, m.emissive_tex.is_some() as u8 as f32],
                flags2: [
                    match (m.alpha, m.transmission > 0.0) { (Alpha::Mask, _) => 1.0, (Alpha::Blend, _) | (_, true) => 2.0, _ => 0.0 },
                    is_grass as u8 as f32, 0.0, 0.0,
                ],
                xf: [xf(&m.base_tex), xf(&m.mr_tex), xf(&m.normal_tex), xf(&m.emissive_tex)],
                rot: [rot(&m.base_tex), rot(&m.mr_tex), rot(&m.normal_tex), rot(&m.emissive_tex)],
            };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&u), usage: wgpu::BufferUsages::UNIFORM });
            let view = |t: &Option<TexRef>, default: &'static str| -> &wgpu::TextureView {
                match (t, default) { (Some(t), _) => &views[t.image], (None, "flat") => &flat, (None, _) => &white }
            };
            let samp = |t: &Option<TexRef>| match t { Some(t) if !t.repeat => &s_clamp, _ => &s_repeat };
            let (vb, vm, vn, ve) = (view(&m.base_tex, "white"), view(&m.mr_tex, "white"), view(&m.normal_tex, "flat"), view(&m.emissive_tex, "white"));
            material_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&m.name),
                layout: &mat_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(vb) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(samp(&m.base_tex)) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(vm) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(samp(&m.mr_tex)) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(vn) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(samp(&m.normal_tex)) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(ve) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(samp(&m.emissive_tex)) },
                ],
            }));
            material_alpha.push(if m.transmission > 0.0 { Alpha::Blend } else { m.alpha });
        }

        // ---- geometry and instances (meshes with a coarse LOD get near/far buffers per frame)
        let mut inst_bufs: Vec<InstBuf> = Vec::new();
        let mut draws = Vec::new();
        let mut shadow_draws = Vec::new();
        let mut lod_sets = Vec::new();
        let mut geom: Vec<Vec<(std::rc::Rc<wgpu::Buffer>, std::rc::Rc<wgpu::Buffer>, u32, usize)>> = Vec::new();
        for prims in &scene.meshes {
            geom.push(prims.iter().map(|p| {
                let verts: Vec<Vertex> = (0..p.positions.len()).map(|k| Vertex { pos: p.positions[k], normal: p.normals[k], uv: p.uvs[k] }).collect();
                (
                    std::rc::Rc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&verts), usage: wgpu::BufferUsages::VERTEX })),
                    std::rc::Rc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&p.indices), usage: wgpu::BufferUsages::INDEX })),
                    p.indices.len() as u32,
                    p.material.min(scene.materials.len() - 1),
                )
            }).collect());
        }
        let mut push_inst = |raw: &[InstanceRaw], dynamic: bool| -> usize {
            let usage = wgpu::BufferUsages::VERTEX | if dynamic { wgpu::BufferUsages::COPY_DST } else { wgpu::BufferUsages::empty() };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(raw), usage });
            inst_bufs.push(InstBuf { buf, count: raw.len() as u32 });
            inst_bufs.len() - 1
        };
        for (mi, insts) in scene.instances.iter().enumerate() {
            if insts.is_empty() {
                continue;
            }
            let raw: Vec<InstanceRaw> = insts.iter().map(|i| InstanceRaw { model: i.model.to_cols_array_2d(), color: i.color.to_array() }).collect();
            let lod = scene.lods.iter().find(|(b, _)| *b == mi).map(|(_, l)| *l);
            let add = |list: &mut Vec<Draw>, mesh: usize, inst: usize| {
                for (vb, ib, count, material) in &geom[mesh] {
                    list.push(Draw { vbuf: vb.clone(), ibuf: ib.clone(), count: *count, inst, material: *material });
                }
            };
            match lod {
                Some(lm) => {
                    let all = push_inst(&raw, false);
                    let near = push_inst(&raw, true);
                    let far = push_inst(&raw, true);
                    add(&mut draws, mi, near);
                    add(&mut draws, lm, far);
                    add(&mut shadow_draws, lm, all);       // shadows never need the fine mesh
                    lod_sets.push(LodSet { pos: raw.iter().map(|r| Vec3::new(r.model[3][0], r.model[3][1], r.model[3][2])).collect(), all: raw, near, far });
                }
                None => {
                    let b = push_inst(&raw, false);
                    add(&mut draws, mi, b);
                    add(&mut shadow_draws, mi, b);
                }
            }
        }

        // ---- lights
        let lights: Vec<LightU> = scene.lights.iter().map(|l| LightU {
            pos: l.position.extend(l.candela / 683.0).to_array(),
            dir: l.direction.extend(l.outer.cos()).to_array(),
            color: l.color.extend(l.inner.cos()).to_array(),
        }).collect();
        // a fixed-size uniform array: Apple GPUs read uniforms from a fast constant cache
        let mut lights = lights;
        lights.truncate(MAX_LIGHTS);
        lights.resize(MAX_LIGHTS, LightU::zeroed());
        let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lights"),
            contents: bytemuck::cast_slice(&lights),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // ---- globals and shadow map
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("globals"), size: std::mem::size_of::<Globals>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow"),
            size: wgpu::Extent3d { width: SHADOW_SIZE, height: SHADOW_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&Default::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor { compare: Some(wgpu::CompareFunction::LessEqual), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        let (grid_views, grid_info) = build_light_grid(device, queue, &light_buf, scene);
        let grid_sampler = device.create_sampler(&wgpu::SamplerDescriptor { mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        let globals_with = |depth: &wgpu::TextureView| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &glob_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&shadow_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&grid_views[0]) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&grid_views[1]) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&grid_views[2]) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&grid_views[3]) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&grid_views[4]) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&grid_views[5]) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&grid_sampler) },
            ],
        });
        let globals_bg = globals_with(&shadow_view);
        // the shadow pass writes the shadow map, so it binds a dummy depth texture instead
        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy depth"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let globals_bg_shadow = globals_with(&dummy.create_view(&Default::default()));

        // ---- pipelines
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&glob_layout, &mat_layout], push_constant_ranges: &[] });
        let vlayouts = [
            wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2] },
            wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<InstanceRaw>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4] },
        ];
        let make = |label: &str, fs: Option<&str>, target: Option<wgpu::ColorTargetState>, depth_write: bool, samples: u32, bias: wgpu::DepthBiasState, vs: &str, compare: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some(vs), buffers: &vlayouts, compilation_options: Default::default() },
                fragment: fs.map(|f| wgpu::FragmentState { module: &shader, entry_point: Some(f), targets: Box::leak(vec![target.clone()].into_boxed_slice()), compilation_options: Default::default() }),
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: depth_write, depth_compare: compare, stencil: Default::default(), bias }),
                multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
                multiview: None,
                cache: None,
            })
        };
        let opaque_t = wgpu::ColorTargetState { format: HDR, blend: None, write_mask: wgpu::ColorWrites::ALL };
        let blend_t = wgpu::ColorTargetState { format: HDR, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL };
        let pre_t = wgpu::ColorTargetState { format: HDR, blend: None, write_mask: wgpu::ColorWrites::empty() };
        let pipe_prepass = make("prepass", Some("fs_prepass"), Some(pre_t), true, MSAA, Default::default(), "vs_main", wgpu::CompareFunction::LessEqual);
        // after the pre-pass, only the exact front-most surface passes (and no shader discard)
        let pipe_opaque = make("opaque", Some("fs_opaque"), Some(opaque_t.clone()), false, MSAA, Default::default(), "vs_main", wgpu::CompareFunction::Equal);
        let pipe_blend = make("blend", Some("fs_main"), Some(blend_t), false, MSAA, Default::default(), "vs_main", wgpu::CompareFunction::LessEqual);
        let pipe_shadow = {
            // the shadow pass has no colour target
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow"),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_shadow"), buffers: &vlayouts, compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_shadow"), targets: &[], compilation_options: Default::default() }),
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::LessEqual, stencil: Default::default(), bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 } }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&glob_layout], push_constant_ranges: &[] });
        let pipe_sky = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky"),
            layout: Some(&sky_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_sky"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_sky"), targets: &[Some(opaque_t)], compilation_options: Default::default() }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::LessEqual, stencil: Default::default(), bias: Default::default() }),
            multisample: wgpu::MultisampleState { count: MSAA, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });
        let tonemap_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                samp_entry(1),
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let tl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&tonemap_layout], push_constant_ranges: &[] });
        let pipe_tonemap = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap"),
            layout: Some(&tl),
            vertex: wgpu::VertexState { module: &tshader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &tshader, entry_point: Some("fs"), targets: &[Some(wgpu::ColorTargetState { format: output_format, blend: None, write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let exposure_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor { mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });

        Renderer {
            globals_buf, globals_bg, globals_bg_shadow, shadow_view, material_bgs, material_alpha, draws, shadow_draws, inst_bufs, lod_sets,
            pipe_prepass, pipe_opaque, pipe_blend, pipe_sky, pipe_shadow, pipe_tonemap, tonemap_layout, exposure_buf, linear_sampler,
            grid: grid_info,
        }
    }

    /// Split every LOD set into near (fine mesh) and far (coarse mesh) instances for this camera.
    pub fn update_lods(&mut self, queue: &wgpu::Queue, eye: Vec3, near_dist: f32) -> (usize, usize) {
        let d2 = near_dist * near_dist;
        let (mut n_near, mut n_far) = (0, 0);
        for k in 0..self.lod_sets.len() {
            let set = &self.lod_sets[k];
            let (mut near, mut far) = (Vec::new(), Vec::new());
            for (raw, p) in set.all.iter().zip(&set.pos) {
                if (*p - eye).length_squared() < d2 { near.push(*raw) } else { far.push(*raw) }
            }
            let (ni, fi) = (set.near, set.far);
            for (idx, list) in [(ni, &near), (fi, &far)] {
                if !list.is_empty() {
                    queue.write_buffer(&self.inst_bufs[idx].buf, 0, bytemuck::cast_slice(list));
                }
                self.inst_bufs[idx].count = list.len() as u32;
            }
            n_near += near.len();
            n_far += far.len();
        }
        (n_near, n_far)
    }

    pub fn targets(&self, device: &wgpu::Device, width: u32, height: u32) -> Targets {
        let tex = |fmt, samples, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage,
                view_formats: &[],
            })
            .create_view(&Default::default())
        };
        let msaa_color = tex(HDR, MSAA, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let msaa_depth = tex(DEPTH, MSAA, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let resolve = tex(HDR, 1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let tonemap_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.tonemap_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&resolve) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.exposure_buf.as_entire_binding() },
            ],
        });
        Targets { msaa_color, msaa_depth, resolve, tonemap_bg }
    }

    pub fn render(&self, device: &wgpu::Device, queue: &wgpu::Queue, t: &Targets, out: &wgpu::TextureView, globals: &Globals, exposure: f32) {
        queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(globals));
        queue.write_buffer(&self.exposure_buf, 0, bytemuck::cast_slice(&[exposure, 0.0, 0.0, 0.0]));
        let mut enc = device.create_command_encoder(&Default::default());
        if globals.params[3] > 0.5 {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &self.shadow_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipe_shadow);
            pass.set_bind_group(0, &self.globals_bg_shadow, &[]);
            self.draw_list(&mut pass, &self.shadow_draws, |a| a != Alpha::Blend);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &t.msaa_color, resolve_target: Some(&t.resolve), ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &t.msaa_depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.globals_bg, &[]);
            pass.set_pipeline(&self.pipe_prepass);
            self.draw_all(&mut pass, |a| a != Alpha::Blend);
            pass.set_pipeline(&self.pipe_opaque);
            self.draw_all(&mut pass, |a| a != Alpha::Blend);
            pass.set_pipeline(&self.pipe_sky);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&self.pipe_blend);
            self.draw_all(&mut pass, |a| a == Alpha::Blend);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: out, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipe_tonemap);
            pass.set_bind_group(0, &t.tonemap_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([enc.finish()]);
    }

    fn draw_all<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, want: impl Fn(Alpha) -> bool) {
        self.draw_list(pass, &self.draws, want)
    }

    fn draw_list<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, list: &'a [Draw], want: impl Fn(Alpha) -> bool) {
        for d in list {
            let ib = &self.inst_bufs[d.inst];
            if ib.count == 0 || !want(self.material_alpha[d.material]) {
                continue;
            }
            pass.set_bind_group(1, &self.material_bgs[d.material], &[]);
            pass.set_vertex_buffer(0, d.vbuf.slice(..));
            pass.set_vertex_buffer(1, ib.buf.slice(..));
            pass.set_index_buffer(d.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..d.count, 0, 0..ib.count);
        }
    }
}

/// Sun shadow projection covering the scene bounds.
pub fn sun_matrix(sun_dir: Vec3, lo: Vec3, hi: Vec3) -> Mat4 {
    let centre = (lo + hi) * 0.5;
    let radius = (hi - lo).length() * 0.5;
    let eye = centre + sun_dir.normalize() * radius * 2.0;
    let up = if sun_dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let view = Mat4::look_at_rh(eye, centre, up);
    let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.1, radius * 4.0);
    proj * view
}

pub fn v4(v: Vec3, w: f32) -> [f32; 4] {
    Vec4::new(v.x, v.y, v.z, w).to_array()
}


/// Build the floodlight irradiance volume once (compute shader); returns its 6 views and placement.
fn build_light_grid(device: &wgpu::Device, queue: &wgpu::Queue, light_buf: &wgpu::Buffer, scene: &Scene) -> (Vec<wgpu::TextureView>, ([f32; 4], [f32; 4])) {
    let (lo, hi) = scene.bounds;
    let pad = Vec3::splat(4.0);
    let (lo, hi) = (lo - pad, hi + pad);
    let ext = hi - lo;
    let has = !scene.lights.is_empty();
    // about a million cells, never finer than 1.5 m
    let cell = if has { (ext.x * ext.y * ext.z / 1.0e6).cbrt().max(1.5) } else { 1.0 };
    let dims = if has { [(ext.x / cell).ceil() as u32, (ext.y / cell).ceil() as u32, (ext.z / cell).ceil() as u32] } else { [1, 1, 1] };
    let views: Vec<wgpu::TextureView> = (0..6).map(|_| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("light grid"),
            size: wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&Default::default())
    }).collect();
    if !has {
        return (views, ([0.0; 4], [1.0; 4]));
    }
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("light grid"), source: wgpu::ShaderSource::Wgsl(include_str!("lightgrid.wgsl").into()) });
    let gu: [u32; 8] = [lo.x.to_bits(), lo.y.to_bits(), lo.z.to_bits(), cell.to_bits(), dims[0], dims[1], dims[2], scene.lights.len().min(MAX_LIGHTS) as u32];
    let gbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&gu), usage: wgpu::BufferUsages::UNIFORM });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("light grid"), layout: None, module: &module, entry_point: Some("build"), compilation_options: Default::default(), cache: None });
    let mut entries = vec![
        wgpu::BindGroupEntry { binding: 0, resource: light_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: gbuf.as_entire_binding() },
    ];
    for (k, v) in views.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry { binding: 2 + k as u32, resource: wgpu::BindingResource::TextureView(v) });
    }
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &entries });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dims[0].div_ceil(4), dims[1].div_ceil(4), dims[2].div_ceil(4));
    }
    queue.submit([enc.finish()]);
    (views, ([lo.x, lo.y, lo.z, cell], [dims[0] as f32 * cell, dims[1] as f32 * cell, dims[2] as f32 * cell, 0.0]))
}
