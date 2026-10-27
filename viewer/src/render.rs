//! wgpu renderer: sun shadow map, 4x MSAA HDR main pass (sky, opaque/mask, blend), ACES tone map.
//!
//! Pipelines are created once; `set_scene` swaps the scene (geometry, materials, lights) so the
//! editor can rebuild a stadium live. GPU textures are cached by image id across rebuilds.

use crate::glb::{Alpha, Image, Scene, TexRef};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use std::collections::HashMap;
use std::rc::Rc;
use wgpu::util::DeviceExt;

const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MSAA: u32 = 4;
const SHADOW_SIZE: u32 = 4096;
/// sun shadow cascades: near detail, mid ground, the whole scene
pub const CASCADES: usize = 3;
const MAX_LIGHTS: usize = 512;
/// grass blades: one per cell of this size (m) inside this radius (m) of the camera's ground focus
const BLADE_SPACING: f32 = 0.016;
const BLADE_RADIUS: f32 = 6.5;
/// the ground map's texel (m) and largest side (texels)
const GMAP_TEXEL: f32 = 0.08;
const GMAP_MAX: u32 = 4096;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    /// one sun projection per shadow cascade
    pub light_view_proj: [[[f32; 4]; 4]; CASCADES],
    pub cam_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub ground: [f32; 4],
    pub params: [f32; 4],
    pub grid_origin: [f32; 4],
    pub grid_size: [f32; 4],
    /// sky-visibility volume: min corner xyz, w = cell size (0 = none)
    pub sky_origin: [f32; 4],
    pub sky_size: [f32; 4],
    /// x = viewport width, y = height, z = SSAO on, w = time
    pub screen: [f32; 4],
    /// cascade far distances (x: 0, y: 1), w = the cascade a shadow pass renders
    pub shadow: [f32; 4],
    /// world size of a shadow texel per cascade
    pub shadow_texel: [f32; 4],
    /// depth range (m) per cascade
    pub shadow_range: [f32; 4],
    /// the ground map (grass blades, ground occlusion): min corner x, z and size x, z (glTF); size 0 = none
    pub gmap: [f32; 4],
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
    /// procedural grass (glb::GrassParams): a, b, c, colour, dry, half
    grass: [[f32; 4]; 6],
    /// x foliage (alpha-tested leaf cards: wrapped, translucent lighting, normals as authored)
    kind: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    color: [f32; 4],
}

struct Draw {
    vbuf: Rc<wgpu::Buffer>,
    ibuf: Rc<wgpu::Buffer>,
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
    /// the fine mesh draws within this distance: about 30 bounding radii, so a seat switches at
    /// ~15 m and a tree or a car at the scene's cap (detail is only drawn where it can be seen)
    switch: f32,
}

pub struct Targets {
    msaa_color: wgpu::TextureView,
    msaa_depth: wgpu::TextureView,
    resolve: wgpu::TextureView,
    tonemap_bg: wgpu::BindGroup,
}

/// Everything that depends on the scene.
/// Grass blades near the camera: the ground map they grow from and how many to draw.
struct GrassField {
    bg: wgpu::BindGroup,
    /// the map's placement for the shaders (Globals::gmap)
    extent: [f32; 4],
    blades: u32,
    /// highest grass in the map (glTF y): no blades are drawn from far above it
    top: f32,
    _maps: [wgpu::TextureView; 2],
}

struct SceneGpu {
    grass: Option<GrassField>,
    globals_bg: wgpu::BindGroup,
    globals_bg_shadow: Vec<wgpu::BindGroup>,
    material_bgs: Vec<wgpu::BindGroup>,
    material_alpha: Vec<Alpha>,
    /// what each material is to the sun: nets are blended on screen but cast an alpha-tested shadow
    shadow_alpha: Vec<Alpha>,
    draws: Vec<Draw>,
    shadow_draws: Vec<Draw>,
    inst_bufs: Vec<InstBuf>,
    lod_sets: Vec<LodSet>,
    /// non-LOD meshes: (instance buffer, original instances), for selection highlights
    plain: HashMap<usize, (usize, Vec<InstanceRaw>)>,
    highlighted: Vec<usize>,
    _light_buf: wgpu::Buffer,
    _grid: Vec<wgpu::TextureView>,
    _sky: Vec<wgpu::TextureView>,
}

pub struct Renderer {
    globals_buf: wgpu::Buffer,
    shadow_view: wgpu::TextureView,
    shadow_layers: Vec<wgpu::TextureView>,
    shadow_globals: Vec<wgpu::Buffer>,
    dummy_depth: wgpu::TextureView,
    mat_layout: wgpu::BindGroupLayout,
    glob_layout: wgpu::BindGroupLayout,
    pipe_prepass: wgpu::RenderPipeline,
    pipe_opaque: wgpu::RenderPipeline,
    pipe_blend: wgpu::RenderPipeline,
    pipe_highlight: wgpu::RenderPipeline,
    pipe_sky: wgpu::RenderPipeline,
    pipe_shadow: wgpu::RenderPipeline,
    pipe_tonemap: wgpu::RenderPipeline,
    pipe_grid: wgpu::ComputePipeline,
    pipe_skyvis: wgpu::ComputePipeline,
    pipe_skydepth: wgpu::RenderPipeline,
    pipe_grassmap: wgpu::RenderPipeline,
    pipe_blades: wgpu::RenderPipeline,
    blade_layout: wgpu::BindGroupLayout,
    tonemap_layout: wgpu::BindGroupLayout,
    exposure_buf: wgpu::Buffer,
    linear_sampler: wgpu::Sampler,
    s_repeat: wgpu::Sampler,
    s_clamp: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    grid_sampler: wgpu::Sampler,
    white: wgpu::TextureView,
    flat: wgpu::TextureView,
    textures: HashMap<(u64, bool), wgpu::TextureView>,
    /// alpha cutoff per image id for textures that alpha-test (coverage-preserving mips)
    cutoffs: HashMap<u64, f32>,
    scene: Option<SceneGpu>,
    /// (origin xyz + cell size, extent xyz) of the floodlight irradiance volume; cell 0 = none
    pub grid: ([f32; 4], [f32; 4]),
    /// same for the sky-visibility volume
    pub sky: ([f32; 4], [f32; 4]),
    pub gmap: [f32; 4],
    dummy_gmap: wgpu::TextureView,
}

/// Mip chain. For alpha-tested textures (`cutoff`), each level's alpha is rescaled so the same
/// fraction of texels passes the test as at full size: thin features such as net cords then fade
/// to a sparse mesh at a distance instead of vanishing (coverage-preserving mipmaps).
fn mips(img: &Image, cutoff: Option<f32>) -> Vec<(u32, u32, Vec<u8>)> {
    let coverage = |px: &[u8], scale: f32, c: f32| px.chunks(4).filter(|p| p[3] as f32 * scale >= c * 255.0).count() as f32 / (px.len() / 4).max(1) as f32;
    let target = cutoff.map(|c| coverage(&img.rgba, 1.0, c));
    let mut out = vec![(img.width, img.height, img.rgba.clone())];
    let mut cur = image::RgbaImage::from_raw(img.width, img.height, img.rgba.clone()).unwrap();
    while cur.width() > 1 || cur.height() > 1 {
        let (w, h) = ((cur.width() / 2).max(1), (cur.height() / 2).max(1));
        cur = image::imageops::resize(&cur, w, h, image::imageops::FilterType::Triangle);
        let mut data = cur.as_raw().clone();
        if let (Some(c), Some(t)) = (cutoff, target) {
            let (mut lo, mut hi) = (0.0f32, 16.0f32);
            for _ in 0..14 {
                let mid = 0.5 * (lo + hi);
                if coverage(&data, mid, c) < t {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            for p in data.chunks_mut(4) {
                p[3] = (p[3] as f32 * hi).min(255.0) as u8;
            }
        }
        out.push((w, h, data));
    }
    out
}

fn uniform_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }
}

fn tex_entry(binding: u32, dim: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: dim, multisampled: false },
        count: None,
    }
}

fn samp_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None }
}

pub fn depth_state(write: bool, compare: wgpu::CompareFunction, bias: wgpu::DepthBiasState) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: Some(write), depth_compare: Some(compare), stencil: Default::default(), bias }
}

impl Renderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, output_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("stadium"), source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()) });
        let tshader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("tonemap"), source: wgpu::ShaderSource::Wgsl(include_str!("tonemap.wgsl").into()) });
        let solid = |rgba: [u8; 4], fmt| {
            device
                .create_texture_with_data(
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
                )
                .create_view(&Default::default())
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
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: 16,
                ..Default::default()
            })
        };
        let s_repeat = sampler(true);
        let s_clamp = sampler(false);
        let d2 = wgpu::TextureViewDimension::D2;
        let d3 = wgpu::TextureViewDimension::D3;
        let mat_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, d2), samp_entry(2), tex_entry(3, d2), samp_entry(4), tex_entry(5, d2), samp_entry(6), tex_entry(7, d2), samp_entry(8),
            ],
        });
        let glob_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison), count: None },
                uniform_entry(3, wgpu::ShaderStages::FRAGMENT),
                tex_entry(4, d3), tex_entry(5, d3), tex_entry(6, d3), tex_entry(7, d3), tex_entry(8, d3), tex_entry(9, d3),
                samp_entry(10),
                tex_entry(11, d3),
                tex_entry(12, d2),
            ],
        });
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("globals"), size: std::mem::size_of::<Globals>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow"),
            size: wgpu::Extent3d { width: SHADOW_SIZE, height: SHADOW_SIZE, depth_or_array_layers: CASCADES as u32 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let shadow_layers = (0..CASCADES as u32)
            .map(|k| shadow_tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_array_layer: k, array_layer_count: Some(1), ..Default::default() }))
            .collect();
        // each cascade's pass reads its own copy of the globals (shadow.w = the cascade)
        let shadow_globals = (0..CASCADES)
            .map(|_| device.create_buffer(&wgpu::BufferDescriptor { label: Some("shadow globals"), size: std::mem::size_of::<Globals>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }))
            .collect();
        // the shadow pass writes the shadow map, so it binds a dummy depth texture instead
        let dummy_gmap = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("dummy ground map"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let dummy_depth = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("dummy depth"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor { compare: Some(wgpu::CompareFunction::LessEqual), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        let grid_sampler = device.create_sampler(&wgpu::SamplerDescriptor { mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&glob_layout), Some(&mat_layout)], immediate_size: 0 });
        let vlayouts = [
            Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 8 => Float32x4] }),
            Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<InstanceRaw>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4] }),
        ];
        let make = |label: &str, fs: &str, target: wgpu::ColorTargetState, depth: wgpu::DepthStencilState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &vlayouts, compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some(fs), targets: &[Some(target)], compilation_options: Default::default() }),
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: Some(depth),
                multisample: wgpu::MultisampleState { count: MSAA, mask: !0, alpha_to_coverage_enabled: false },
                multiview_mask: None,
                cache: None,
            })
        };
        let opaque_t = wgpu::ColorTargetState { format: HDR, blend: None, write_mask: wgpu::ColorWrites::ALL };
        let blend_t = wgpu::ColorTargetState { format: HDR, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL };
        let pre_t = wgpu::ColorTargetState { format: HDR, blend: None, write_mask: wgpu::ColorWrites::empty() };
        use wgpu::CompareFunction as C;
        let pipe_prepass = make("prepass", "fs_prepass", pre_t, depth_state(true, C::LessEqual, Default::default()));
        // after the pre-pass, only the exact front-most surface passes (and no shader discard)
        let pipe_opaque = make("opaque", "fs_opaque", opaque_t.clone(), depth_state(false, C::Equal, Default::default()));
        let pipe_blend = make("blend", "fs_main", blend_t.clone(), depth_state(false, C::LessEqual, Default::default()));
        // editor selection: the selected parts glow through whatever covers them (seats, roofs)
        let pipe_highlight = make("highlight", "fs_highlight", blend_t, depth_state(false, C::Always, Default::default()));
        let shadow_pipe = |label: &str, size_bias: wgpu::DepthBiasState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_shadow"), buffers: &vlayouts, compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_shadow"), targets: &[], compilation_options: Default::default() }),
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: Some(depth_state(true, C::LessEqual, size_bias)),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipe_shadow = shadow_pipe("shadow", wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 });
        let pipe_skydepth = shadow_pipe("sky depth", wgpu::DepthBiasState { constant: 4, slope_scale: 2.0, clamp: 0.0 });
        // the grass ground map: the scene seen from straight above, topmost surface wins
        let pipe_grassmap = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grass map"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &vlayouts, compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_grassmap"),
                targets: &[
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(depth_state(true, C::Less, Default::default())),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let vtex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let blade_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("blades"), entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX), vtex(1), vtex(2)] });
        let bl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&glob_layout), None, Some(&blade_layout)], immediate_size: 0 });
        let pipe_blades = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grass blades"),
            layout: Some(&bl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_blade"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_blade"), targets: &[Some(opaque_t.clone())], compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(depth_state(true, C::LessEqual, Default::default())),
            multisample: wgpu::MultisampleState { count: MSAA, mask: !0, alpha_to_coverage_enabled: false },
            multiview_mask: None,
            cache: None,
        });
        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&glob_layout)], immediate_size: 0 });
        let pipe_sky = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky"),
            layout: Some(&sky_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_sky"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_sky"), targets: &[Some(opaque_t)], compilation_options: Default::default() }),
            primitive: Default::default(),
            depth_stencil: Some(depth_state(false, C::LessEqual, Default::default())),
            multisample: wgpu::MultisampleState { count: MSAA, mask: !0, alpha_to_coverage_enabled: false },
            multiview_mask: None,
            cache: None,
        });
        let tonemap_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap"),
            entries: &[tex_entry(0, d2), samp_entry(1), uniform_entry(2, wgpu::ShaderStages::FRAGMENT)],
        });
        let tl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&tonemap_layout)], immediate_size: 0 });
        let pipe_tonemap = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap"),
            layout: Some(&tl),
            vertex: wgpu::VertexState { module: &tshader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &tshader, entry_point: Some("fs"), targets: &[Some(wgpu::ColorTargetState { format: output_format, blend: None, write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let grid_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("light grid"), source: wgpu::ShaderSource::Wgsl(include_str!("lightgrid.wgsl").into()) });
        let pipe_grid = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("light grid"), layout: None, module: &grid_module, entry_point: Some("build"), compilation_options: Default::default(), cache: None });
        let sky_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("sky visibility"), source: wgpu::ShaderSource::Wgsl(include_str!("skyvis.wgsl").into()) });
        let pipe_skyvis = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("sky visibility"), layout: None, module: &sky_module, entry_point: Some("build"), compilation_options: Default::default(), cache: None });
        let exposure_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor { mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        Renderer {
            globals_buf, shadow_view, shadow_layers, shadow_globals, dummy_depth, mat_layout, glob_layout, pipe_prepass, pipe_opaque, pipe_blend, pipe_highlight, pipe_sky, pipe_shadow,
            pipe_tonemap, pipe_grid, pipe_skyvis, pipe_skydepth, pipe_grassmap, pipe_blades, blade_layout, tonemap_layout, exposure_buf, linear_sampler, s_repeat, s_clamp, shadow_sampler, grid_sampler,
            white, flat, textures: HashMap::new(), cutoffs: HashMap::new(), scene: None, grid: ([0.0; 4], [1.0; 4]), sky: ([0.0; 4], [1.0; 4]), gmap: [0.0; 4], dummy_gmap,
        }
    }

    fn texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, img: &Image, srgb: bool) {
        if self.textures.contains_key(&(img.id, srgb)) {
            return;
        }
        let max_dim = device.limits().max_texture_dimension_2d;
        let mut levels = mips(img, if srgb { self.cutoffs.get(&img.id).copied() } else { None });
        while levels[0].0 > max_dim || levels[0].1 > max_dim {
            levels.remove(0);
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: levels[0].0, height: levels[0].1, depth_or_array_layers: 1 },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (lvl, (w, h, data)) in levels.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: lvl as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * w), rows_per_image: Some(*h) },
                wgpu::Extent3d { width: *w, height: *h, depth_or_array_layers: 1 },
            );
        }
        self.textures.insert((img.id, srgb), tex.create_view(&Default::default()));
    }

    /// Upload a scene: materials, geometry, instances, lights, light and sky volumes.
    pub fn set_scene(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene) {
        self.scene = None;
        // textures (sRGB if used as colour, linear otherwise); keep only those this scene uses
        let mut srgb = vec![false; scene.images.len()];
        for m in &scene.materials {
            for t in [m.base_tex, m.emissive_tex].into_iter().flatten() {
                srgb[t.image] = true;
            }
        }
        for m in &scene.materials {
            if let (Alpha::Mask, Some(t), false) = (m.alpha, m.base_tex, m.name.contains("net_card")) {
                self.cutoffs.insert(scene.images[t.image].id, m.cutoff);
            }
        }
        let used: std::collections::HashSet<(u64, bool)> = scene.images.iter().zip(&srgb).map(|(i, s)| (i.id, *s)).collect();
        self.textures.retain(|k, _| used.contains(k));
        for (img, s) in scene.images.iter().zip(&srgb) {
            self.texture(device, queue, img, *s);
        }

        let mut material_bgs = Vec::new();
        let mut shadow_alpha = Vec::new();
        let mut material_alpha = Vec::new();
        for m in &scene.materials {
            let xf = |t: &Option<TexRef>| t.map(|t| [t.xform.offset.x, t.xform.offset.y, t.xform.scale.x, t.xform.scale.y]).unwrap_or([0.0, 0.0, 1.0, 1.0]);
            let rot = |t: &Option<TexRef>| t.map(|t| t.xform.rotation).unwrap_or(0.0);
            let is_grass = m.grass.is_some() && m.name != "TINT_Pitch_Paint";
            // 1 = procedural grass, 2 = line paint sprayed on grass
            let procedural = if m.name == "TINT_Pitch_Paint" { 2.0 } else { m.extras.get("procedural").and_then(|v| v.as_bool()).unwrap_or(false) as u8 as f32 };
            // materials that take the per-instance seat colour (the seat shell, not its frame)
            let instance_tint = (m.name.starts_with("TINT_seat") || m.name == "Seat_Wood") as u8 as f32;
            let u = MaterialU {
                base_color: m.base_color.to_array(),
                emissive: (m.emissive * m.emissive_strength).extend(0.0).to_array(),
                params: [m.metallic, m.roughness, m.cutoff, m.transmission],
                flags: [m.base_tex.is_some() as u8 as f32, m.mr_tex.is_some() as u8 as f32, m.normal_tex.is_some() as u8 as f32, m.emissive_tex.is_some() as u8 as f32],
                flags2: [
                    match (m.alpha, m.transmission > 0.0) {
                        (Alpha::Mask, _) if m.name.contains("net_card") => 2.0,
                        (Alpha::Mask, _) => 1.0,
                        (Alpha::Blend, _) | (_, true) => 2.0,
                        _ => 0.0,
                    },
                    is_grass as u8 as f32,
                    procedural,
                    instance_tint,
                ],
                xf: [xf(&m.base_tex), xf(&m.mr_tex), xf(&m.normal_tex), xf(&m.emissive_tex)],
                rot: [rot(&m.base_tex), rot(&m.mr_tex), rot(&m.normal_tex), rot(&m.emissive_tex)],
                grass: m.grass.map(|g| [g.a, g.b, g.c, g.color, g.dry, g.half]).unwrap_or([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.025, 0.0], [0.06, 0.12, 0.02, 0.0], [0.25, 0.22, 0.07, 0.0], [52.5, 34.0, 0.0, 0.0]]),
                kind: [
                    m.name.starts_with("Leaf") as u8 as f32,
                    matches!(m.name.as_str(), "Ads_LED" | "Ads_Ribbon" | "Screen_LED") as u8 as f32,
                    // glazing: double-glazed units reflect from four faces (twin-wall polycarbonate
                    // sheeting likewise)
                    if m.name.starts_with("Facade_DarkGlass") {
                        2.0 // solar-control curtain wall: a coated, mirror-like glass
                    } else {
                        ((m.name.contains("Glass") || m.name.contains("Polycarbonate")) && !m.name.contains("Lit")) as u8 as f32
                    },
                    // structural concrete: weathered in the shader (streaks, tone, stains);
                    // 2 = vehicle paint under a clear coat
                    if m.name.starts_with("TINT_car_paint") || m.name.starts_with("TINT_van_paint") {
                        2.0
                    } else {
                        ["Stand_Concrete", "Stand_Block", "Stand_Soffit", "Facade_Concrete", "Apron_Concrete"].iter().any(|k| m.name.starts_with(k)) as u8 as f32
                    },
                ],
            };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&u), usage: wgpu::BufferUsages::UNIFORM });
            let samp = |t: &Option<TexRef>| match t {
                Some(t) if !t.repeat => &self.s_clamp,
                _ => &self.s_repeat,
            };
            let view = |t: &Option<TexRef>, s: bool, default: bool| pick(&self.textures, scene, t, s, if default { &self.white } else { &self.flat });
            let (vb, vm, vn, ve) = (view(&m.base_tex, true, true), view(&m.mr_tex, false, true), view(&m.normal_tex, false, false), view(&m.emissive_tex, true, true));
            material_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&m.name),
                layout: &self.mat_layout,
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
            // net cards are cords a few millimetres thick: blended, their mipmapped alpha fades to
            // faint continuous lines with distance (an alpha test breaks them into dots)
            let net = m.name.contains("net_card");
            material_alpha.push(if m.transmission > 0.0 || net { Alpha::Blend } else { m.alpha });
            shadow_alpha.push(if net { Alpha::Mask } else if m.transmission > 0.0 { Alpha::Blend } else { m.alpha });
        }

        // geometry and instances (meshes with a coarse LOD get near/far buffers per frame)
        let mut inst_bufs: Vec<InstBuf> = Vec::new();
        let mut draws = Vec::new();
        let mut shadow_draws = Vec::new();
        let mut lod_sets = Vec::new();
        let mut plain = HashMap::new();
        let nmat = scene.materials.len();
        let geom: Vec<Vec<(Rc<wgpu::Buffer>, Rc<wgpu::Buffer>, u32, usize)>> = scene
            .meshes
            .iter()
            .enumerate()
            .map(|(mi, prims)| {
                if scene.instances[mi].is_empty() && !scene.lods.iter().any(|l| l.1 == mi) {
                    return Vec::new();
                }
                prims
                    .iter()
                    .filter(|p| !p.indices.is_empty())
                    .map(|p| {
                        let verts: Vec<Vertex> = (0..p.positions.len()).map(|k| Vertex { pos: p.positions[k], normal: p.normals[k], uv: p.uvs[k], color: p.colors.get(k).copied().unwrap_or([1.0; 4]) }).collect();
                        (
                            Rc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&verts), usage: wgpu::BufferUsages::VERTEX })),
                            Rc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&p.indices), usage: wgpu::BufferUsages::INDEX })),
                            p.indices.len() as u32,
                            p.material.min(nmat - 1),
                        )
                    })
                    .collect()
            })
            .collect();
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
                    add(&mut shadow_draws, lm, all); // shadows never need the fine mesh
                    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                    for p in &scene.meshes[mi] {
                        for q in &p.positions {
                            lo = lo.min(Vec3::from(*q));
                            hi = hi.max(Vec3::from(*q));
                        }
                    }
                    let radius = if lo.x <= hi.x { (hi - lo).length() * 0.5 } else { 1.0 };
                    lod_sets.push(LodSet { pos: raw.iter().map(|r| Vec3::new(r.model[3][0], r.model[3][1], r.model[3][2])).collect(), all: raw, near, far, switch: (radius * 30.0).max(12.0) });
                }
                None => {
                    let b = push_inst(&raw, true);
                    add(&mut draws, mi, b);
                    add(&mut shadow_draws, mi, b);
                    plain.insert(mi, (b, raw));
                }
            }
        }

        // lights: a fixed-size uniform array (Apple GPUs read uniforms from a fast constant cache)
        let mut lights: Vec<LightU> = scene
            .lights
            .iter()
            .map(|l| LightU { pos: l.position.extend(l.candela / 683.0).to_array(), dir: l.direction.extend(l.outer.cos()).to_array(), color: l.color.extend(l.inner.cos()).to_array() })
            .collect();
        lights.truncate(MAX_LIGHTS);
        lights.resize(MAX_LIGHTS, LightU::zeroed());
        let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("lights"), contents: bytemuck::cast_slice(&lights), usage: wgpu::BufferUsages::UNIFORM });
        let (grid_views, grid_info) = self.build_light_grid(device, queue, &light_buf, scene);
        self.grid = grid_info;
        let (sky_views, sky_info) = self.build_sky_volume(device, queue, scene, &shadow_draws, &material_bgs, &material_alpha, &inst_bufs);
        self.sky = sky_info;
        let grass = self.build_grass_field(device, queue, scene, &shadow_draws, &material_bgs, &material_alpha, &inst_bufs, &light_buf, &grid_views, &sky_views);
        self.gmap = grass.as_ref().map(|g| g.extent).unwrap_or([0.0; 4]);
        let gmap_view = grass.as_ref().map(|g| &g._maps[0]).unwrap_or(&self.dummy_gmap);
        let globals_with = |buf: &wgpu::Buffer, depth: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("globals"),
                layout: &self.glob_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.shadow_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: light_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&grid_views[0]) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&grid_views[1]) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&grid_views[2]) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&grid_views[3]) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&grid_views[4]) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&grid_views[5]) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.grid_sampler) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&sky_views[0]) },
                    wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(gmap_view) },
                ],
            })
        };
        let globals_bg = globals_with(&self.globals_buf, &self.shadow_view);
        let globals_bg_shadow = self.shadow_globals.iter().map(|b| globals_with(b, &self.dummy_depth)).collect();
        self.scene = Some(SceneGpu {
            grass,
            globals_bg, globals_bg_shadow, material_bgs, material_alpha, shadow_alpha, draws, shadow_draws, inst_bufs, lod_sets,
            plain, highlighted: Vec::new(),
            _light_buf: light_buf, _grid: grid_views, _sky: sky_views,
        });
    }

    /// Highlight these meshes (selection): their instances get colour `rgb` with w = 2, which the
    /// shader draws as a rim glow. An empty list clears the highlight.
    pub fn set_highlight(&mut self, queue: &wgpu::Queue, meshes: &[usize], rgb: [f32; 3]) {
        let Some(sc) = self.scene.as_mut() else { return };
        for m in std::mem::take(&mut sc.highlighted) {
            if let Some((b, raw)) = sc.plain.get(&m) {
                queue.write_buffer(&sc.inst_bufs[*b].buf, 0, bytemuck::cast_slice(raw));
            }
        }
        for &m in meshes {
            if let Some((b, raw)) = sc.plain.get(&m) {
                let lit: Vec<InstanceRaw> = raw.iter().map(|r| InstanceRaw { model: r.model, color: [rgb[0], rgb[1], rgb[2], 2.0] }).collect();
                queue.write_buffer(&sc.inst_bufs[*b].buf, 0, bytemuck::cast_slice(&lit));
                sc.highlighted.push(m);
            }
        }
    }

    /// Split every LOD set into near (fine mesh) and far (coarse mesh) instances for this camera.
    pub fn update_lods(&mut self, queue: &wgpu::Queue, eye: Vec3, near_dist: f32) -> (usize, usize) {
        let Some(sc) = self.scene.as_mut() else { return (0, 0) };
        let (mut n_near, mut n_far) = (0, 0);
        for set in &sc.lod_sets {
            let d = set.switch.min(near_dist);
            let d2 = d * d;
            let (mut near, mut far) = (Vec::new(), Vec::new());
            for (raw, p) in set.all.iter().zip(&set.pos) {
                if (*p - eye).length_squared() < d2 {
                    near.push(*raw)
                } else {
                    far.push(*raw)
                }
            }
            for (idx, list) in [(set.near, &near), (set.far, &far)] {
                if !list.is_empty() {
                    queue.write_buffer(&sc.inst_bufs[idx].buf, 0, bytemuck::cast_slice(list));
                }
                sc.inst_bufs[idx].count = list.len() as u32;
            }
            n_near += near.len();
            n_far += far.len();
        }
        (n_near, n_far)
    }

    pub fn targets(&self, device: &wgpu::Device, width: u32, height: u32) -> Targets {
        let tex = |fmt, samples, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
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

    /// Record the frame into `enc` (the caller submits, so overlays can draw on top).
    pub fn render(&self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, t: &Targets, out: &wgpu::TextureView, globals: &Globals, exposure: f32) {
        queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(globals));
        queue.write_buffer(&self.exposure_buf, 0, bytemuck::cast_slice(&[exposure, 0.0, 0.0, 0.0]));
        if let Some(sc) = &self.scene {
            if globals.params[3] > 0.5 {
                for k in 0..CASCADES {
                    let mut g = *globals;
                    g.shadow[3] = k as f32;
                    queue.write_buffer(&self.shadow_globals[k], 0, bytemuck::bytes_of(&g));
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("shadow"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &self.shadow_layers[k], depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(&self.pipe_shadow);
                    pass.set_bind_group(0, &sc.globals_bg_shadow[k], &[]);
                    draw_list_by(&mut pass, sc, &sc.shadow_draws, &sc.shadow_alpha, |a| a != Alpha::Blend);
                }
            }
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &t.msaa_color, depth_slice: None, resolve_target: Some(&t.resolve), ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &t.msaa_depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(sc) = &self.scene {
                pass.set_bind_group(0, &sc.globals_bg, &[]);
                pass.set_pipeline(&self.pipe_prepass);
                draw_list(&mut pass, sc, &sc.draws, |a| a != Alpha::Blend);
                pass.set_pipeline(&self.pipe_opaque);
                draw_list(&mut pass, sc, &sc.draws, |a| a != Alpha::Blend);
                // grass blades, only with the eye near the turf (from a gantry they are sub-pixel)
                if let Some(gf) = &sc.grass {
                    if globals.cam_pos[1] < gf.top + 14.0 {
                        pass.set_pipeline(&self.pipe_blades);
                        pass.set_bind_group(2, &gf.bg, &[]);
                        pass.draw(0..15, 0..gf.blades);
                    }
                }
                pass.set_pipeline(&self.pipe_sky);
                pass.draw(0..3, 0..1);
                pass.set_pipeline(&self.pipe_blend);
                draw_list(&mut pass, sc, &sc.draws, |a| a == Alpha::Blend);
                if !sc.highlighted.is_empty() {
                    let bufs: Vec<usize> = sc.highlighted.iter().filter_map(|m| sc.plain.get(m).map(|p| p.0)).collect();
                    let lit: Vec<&Draw> = sc.draws.iter().filter(|d| bufs.contains(&d.inst)).collect();
                    pass.set_pipeline(&self.pipe_highlight);
                    for d in lit {
                        let ib = &sc.inst_bufs[d.inst];
                        pass.set_bind_group(1, &sc.material_bgs[d.material], &[]);
                        pass.set_vertex_buffer(0, d.vbuf.slice(..));
                        pass.set_vertex_buffer(1, ib.buf.slice(..));
                        pass.set_index_buffer(d.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..d.count, 0, 0..ib.count);
                    }
                }
            }
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: out, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipe_tonemap);
            pass.set_bind_group(0, &t.tonemap_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Build the floodlight irradiance volume once (compute shader); returns its 6 views and placement.
    fn build_light_grid(&self, device: &wgpu::Device, queue: &wgpu::Queue, light_buf: &wgpu::Buffer, scene: &Scene) -> (Vec<wgpu::TextureView>, ([f32; 4], [f32; 4])) {
        let (lo, hi) = scene.bounds;
        let pad = Vec3::splat(4.0);
        let (lo, hi) = (lo - pad, hi + pad);
        let ext = hi - lo;
        let has = !scene.lights.is_empty();
        // about a million cells, never finer than 1.5 m
        let cell = if has { (ext.x * ext.y * ext.z / 1.0e6).cbrt().max(1.5) } else { 1.0 };
        let dims = if has { [(ext.x / cell).ceil() as u32, (ext.y / cell).ceil() as u32, (ext.z / cell).ceil() as u32] } else { [1, 1, 1] };
        let views: Vec<wgpu::TextureView> = (0..6).map(|_| volume(device, dims, wgpu::TextureFormat::Rgba16Float)).collect();
        if !has {
            return (views, ([0.0; 4], [1.0; 4]));
        }
        let gu: [u32; 8] = [lo.x.to_bits(), lo.y.to_bits(), lo.z.to_bits(), cell.to_bits(), dims[0], dims[1], dims[2], scene.lights.len().min(MAX_LIGHTS) as u32];
        let gbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&gu), usage: wgpu::BufferUsages::UNIFORM });
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: light_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: gbuf.as_entire_binding() }];
        for (k, v) in views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: 2 + k as u32, resource: wgpu::BindingResource::TextureView(v) });
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.pipe_grid.get_bind_group_layout(0), entries: &entries });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipe_grid);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dims[0].div_ceil(4), dims[1].div_ceil(4), dims[2].div_ceil(4));
        }
        queue.submit([enc.finish()]);
        (views, ([lo.x, lo.y, lo.z, cell], [dims[0] as f32 * cell, dims[1] as f32 * cell, dims[2] as f32 * cell, 0.0]))
    }

    /// Sky visibility volume: render depth maps of the scene from SKY_DIRS directions over the upper
    /// hemisphere, then for every cell of a 3D grid store the cosine-weighted fraction of those
    /// directions with a clear view of the sky. Roofs and stands then shade the space under them,
    /// as the open sky never reaches there in reality.
    #[allow(clippy::too_many_arguments)]
    /// The ground map the grass blades grow from: the scene from straight above (orthographic),
    /// over the built-up area (where structures are, padded), at GMAP_TEXEL metres a texel.
    #[allow(clippy::too_many_arguments)]
    fn build_grass_field(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        draws: &[Draw],
        material_bgs: &[wgpu::BindGroup],
        material_alpha: &[Alpha],
        inst_bufs: &[InstBuf],
        light_buf: &wgpu::Buffer,
        grid_views: &[wgpu::TextureView],
        sky_views: &[wgpu::TextureView],
    ) -> Option<GrassField> {
        let grass_mat = |i: usize| scene.materials.get(i).is_some_and(|m| m.grass.is_some());
        if !scene.materials.iter().any(|m| m.grass.is_some()) {
            return None;
        }
        // extent: everything that is not grass (structures, props), padded; else the grass itself
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        let (mut glo, mut ghi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        let mut gtop = f32::MIN;
        for (mi, prims) in scene.meshes.iter().enumerate() {
            let insts = &scene.instances[mi];
            if insts.is_empty() {
                continue;
            }
            for p in prims {
                if p.positions.is_empty() {
                    continue;
                }
                let (mut a, mut b) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for q in &p.positions {
                    a = a.min(Vec3::from(*q));
                    b = b.max(Vec3::from(*q));
                }
                let is_grass = grass_mat(p.material);
                for inst in insts {
                    for k in 0..8 {
                        let c = Vec3::new(if k & 1 == 0 { a.x } else { b.x }, if k & 2 == 0 { a.y } else { b.y }, if k & 4 == 0 { a.z } else { b.z });
                        let w = inst.model.transform_point3(c);
                        if is_grass {
                            glo = glo.min(w);
                            ghi = ghi.max(w);
                            gtop = gtop.max(w.y);
                        } else {
                            lo = lo.min(w);
                            hi = hi.max(w);
                        }
                    }
                }
            }
        }
        if gtop == f32::MIN {
            return None;
        }
        if lo.x == f32::MAX {
            (lo, hi) = (glo, ghi);
        }
        let pad = 15.0;
        let (mut x0, mut z0, mut x1, mut z1) = ((lo.x - pad).max(glo.x), (lo.z - pad).max(glo.z), (hi.x + pad).min(ghi.x), (hi.z + pad).min(ghi.z));
        let lim = GMAP_MAX as f32 * GMAP_TEXEL;
        let shrink = |a: &mut f32, b: &mut f32| {
            if *b - *a > lim {
                let c = (*a + *b) * 0.5;
                (*a, *b) = (c - lim * 0.5, c + lim * 0.5);
            }
        };
        shrink(&mut x0, &mut x1);
        shrink(&mut z0, &mut z1);
        if x1 <= x0 || z1 <= z0 {
            return None;
        }
        let w = (((x1 - x0) / GMAP_TEXEL).ceil() as u32).clamp(16, GMAP_MAX);
        let h = (((z1 - z0) / GMAP_TEXEL).ceil() as u32).clamp(16, GMAP_MAX);
        // looking straight down -y; texel (u, v) = ((x - x0) / sx, (z - z0) / sz)
        let ytop = hi.y.max(ghi.y) + 10.0;
        let ybot = lo.y.min(glo.y) - 10.0;
        // the matrix by hand: clip x from world x, clip y from world z (v grows with z, and
        // texture rows run down the screen), depth from height (the highest surface is nearest)
        let sx = 2.0 / (x1 - x0);
        let sz = 2.0 / (z1 - z0);
        let vp = Mat4::from_cols(
            glam::Vec4::new(sx, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, -1.0 / (ytop - ybot), 0.0),
            glam::Vec4::new(0.0, -sz, 0.0, 0.0),
            glam::Vec4::new(-1.0 - x0 * sx, 1.0 + z0 * sz, ytop / (ytop - ybot), 1.0),
        );
        let mut g = Globals::zeroed();
        g.view_proj = vp.to_cols_array_2d();
        g.cam_pos = [0.0, ytop, 0.0, 1.0];
        let gb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("grass map globals"), contents: bytemuck::bytes_of(&g), usage: wgpu::BufferUsages::UNIFORM });
        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.glob_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.dummy_depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.shadow_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&grid_views[0]) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&grid_views[1]) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&grid_views[2]) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&grid_views[3]) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&grid_views[4]) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&grid_views[5]) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.grid_sampler) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&sky_views[0]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&self.dummy_gmap) },
            ],
        });
        let tex = |format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("grass map"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let rt = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let ta = tex(wgpu::TextureFormat::Rgba16Float, rt).create_view(&Default::default());
        let tb = tex(wgpu::TextureFormat::Rgba8Unorm, rt).create_view(&Default::default());
        let td = tex(DEPTH, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let clear = wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("grass map"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment { view: &ta, depth_slice: None, resolve_target: None, ops: clear }),
                    Some(wgpu::RenderPassColorAttachment { view: &tb, depth_slice: None, resolve_target: None, ops: clear }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &td, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipe_grassmap);
            pass.set_bind_group(0, &globals_bg, &[]);
            for d in draws {
                let ib = &inst_bufs[d.inst];
                // leaves, nets, glass and cut-out cards don't cover the turf
                let foliage = scene.materials.get(d.material).is_some_and(|m| m.name.starts_with("Leaf"));
                if ib.count == 0 || material_alpha[d.material] != Alpha::Opaque || foliage {
                    continue;
                }
                pass.set_bind_group(1, &material_bgs[d.material], &[]);
                pass.set_vertex_buffer(0, d.vbuf.slice(..));
                pass.set_vertex_buffer(1, ib.buf.slice(..));
                pass.set_index_buffer(d.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..d.count, 0, 0..ib.count);
            }
        }
        queue.submit([enc.finish()]);
        let n = (2.0 * BLADE_RADIUS / BLADE_SPACING).ceil() as u32;
        let u: [f32; 8] = [x0, z0, x1 - x0, z1 - z0, n as f32, BLADE_SPACING, BLADE_RADIUS, 0.0];
        let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("blades"), contents: bytemuck::cast_slice(&u), usage: wgpu::BufferUsages::UNIFORM });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blades"),
            layout: &self.blade_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ta) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&tb) },
            ],
        });
        Some(GrassField { bg, extent: [x0, z0, x1 - x0, z1 - z0], blades: n * n, top: gtop, _maps: [ta, tb] })
    }

    fn build_sky_volume(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        shadow_draws: &[Draw],
        material_bgs: &[wgpu::BindGroup],
        material_alpha: &[Alpha],
        inst_bufs: &[InstBuf],
    ) -> (Vec<wgpu::TextureView>, ([f32; 4], [f32; 4])) {
        const SKY_DIRS: usize = 24;
        const SKY_RES: u32 = 1536;
        let (lo, hi) = scene.bounds;
        let pad = Vec3::splat(2.0);
        let (lo, hi) = (lo - pad, hi + pad);
        let ext = hi - lo;
        let has = scene.meshes.iter().any(|m| !m.is_empty()) && ext.x.is_finite() && ext.x > 0.0;
        // about two million cells, never finer than 0.75 m
        let cell = if has { (ext.x * ext.y * ext.z / 2.0e6).cbrt().max(0.75) } else { 1.0 };
        let dims = if has { [(ext.x / cell).ceil() as u32, (ext.y / cell).ceil() as u32, (ext.z / cell).ceil() as u32] } else { [1, 1, 1] };
        let vol_tex = volume_tex(device, dims, wgpu::TextureFormat::Rgba16Float);
        let vol = vol_tex.create_view(&Default::default());
        if !has {
            return (vec![vol], ([0.0; 4], [1.0; 4]));
        }
        // directions: golden-angle spiral over the hemisphere (y up), cosine-distributed
        let mut mats = Vec::with_capacity(SKY_DIRS);
        let mut dirv = vec![[0.0f32; 4]; 32];
        let centre = (lo + hi) * 0.5;
        let radius = ext.length() * 0.5;
        for k in 0..SKY_DIRS {
            let u = (k as f32 + 0.5) / SKY_DIRS as f32;
            let el = (1.0 - u).sqrt().acos(); // cosine-weighted: sin^2(theta) uniform
            let az = k as f32 * 2.399_963;
            let d = Vec3::new((std::f32::consts::FRAC_PI_2 - el).cos() * az.cos(), (std::f32::consts::FRAC_PI_2 - el).sin(), (std::f32::consts::FRAC_PI_2 - el).cos() * az.sin()).normalize();
            dirv[k] = [d.x, d.y, d.z, 0.0];
            let eye = centre + d * radius * 2.0;
            let up = if d.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
            let view = Mat4::look_at_rh(eye, centre, up);
            let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.1, radius * 4.0);
            mats.push(proj * view);
        }
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky depth"),
            size: wgpu::Extent3d { width: SKY_RES, height: SKY_RES, depth_or_array_layers: SKY_DIRS as u32 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        let mut gbufs = Vec::new();
        for (k, m) in mats.iter().enumerate() {
            // a globals block per direction: vs_shadow reads light_view_proj
            let mut g = Globals::zeroed();
            g.light_view_proj[0] = m.to_cols_array_2d();
            let gb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&g), usage: wgpu::BufferUsages::UNIFORM });
            let empty_light = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (MAX_LIGHTS * std::mem::size_of::<LightU>()) as u64, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false });
            let v3 = volume(device, [1, 1, 1], wgpu::TextureFormat::Rgba16Float);
            let s3 = volume(device, [1, 1, 1], wgpu::TextureFormat::Rgba16Float);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.glob_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gb.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.dummy_depth) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.shadow_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: empty_light.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&v3) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&v3) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&v3) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&v3) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&v3) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&v3) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.grid_sampler) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&s3) },
                    wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&self.dummy_gmap) },
                ],
            });
            let layer = depth.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_array_layer: k as u32, array_layer_count: Some(1), ..Default::default() });
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sky depth"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &layer, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipe_skydepth);
                pass.set_bind_group(0, &bg, &[]);
                for d in shadow_draws {
                    let ib = &inst_bufs[d.inst];
                    // glass and translucent sheeting let the sky through
                    if ib.count == 0 || material_alpha[d.material] == Alpha::Blend {
                        continue;
                    }
                    pass.set_bind_group(1, &material_bgs[d.material], &[]);
                    pass.set_vertex_buffer(0, d.vbuf.slice(..));
                    pass.set_vertex_buffer(1, ib.buf.slice(..));
                    pass.set_index_buffer(d.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..d.count, 0, 0..ib.count);
                }
            }
            gbufs.push((gb, empty_light));
        }
        // the volume: one compute pass over the grid
        let mut mat_data: Vec<[[f32; 4]; 4]> = mats.iter().map(|m| m.to_cols_array_2d()).collect();
        mat_data.resize(32, [[0.0; 4]; 4]);
        let mut ubytes: Vec<u8> = bytemuck::cast_slice(&mat_data).to_vec();
        ubytes.extend_from_slice(bytemuck::cast_slice(&dirv));
        let mbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &ubytes, usage: wgpu::BufferUsages::UNIFORM });
        let gu: [u32; 8] = [lo.x.to_bits(), lo.y.to_bits(), lo.z.to_bits(), cell.to_bits(), dims[0], dims[1], dims[2], SKY_DIRS as u32];
        let gbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&gu), usage: wgpu::BufferUsages::UNIFORM });
        let dview = depth.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let store = vol_tex.create_view(&Default::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipe_skyvis.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: mbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dview) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&store) },
            ],
        });
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipe_skyvis);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dims[0].div_ceil(4), dims[1].div_ceil(4), dims[2].div_ceil(4));
        }
        queue.submit([enc.finish()]);
        (vec![vol], ([lo.x, lo.y, lo.z, cell], [dims[0] as f32 * cell, dims[1] as f32 * cell, dims[2] as f32 * cell, 0.0]))
    }
}

fn pick<'a>(textures: &'a HashMap<(u64, bool), wgpu::TextureView>, scene: &Scene, t: &Option<TexRef>, srgb: bool, default: &'a wgpu::TextureView) -> &'a wgpu::TextureView {
    match t {
        Some(t) => &textures[&(scene.images[t.image].id, srgb)],
        None => default,
    }
}

fn volume(device: &wgpu::Device, dims: [u32; 3], format: wgpu::TextureFormat) -> wgpu::TextureView {
    volume_tex(device, dims, format).create_view(&Default::default())
}

fn volume_tex(device: &wgpu::Device, dims: [u32; 3], format: wgpu::TextureFormat) -> wgpu::Texture {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("volume"),
            size: wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
}

fn draw_list(pass: &mut wgpu::RenderPass<'_>, sc: &SceneGpu, list: &[Draw], want: impl Fn(Alpha) -> bool) {
    draw_list_by(pass, sc, list, &sc.material_alpha, want);
}

/// draw_list, choosing draws by a given per-material alpha mode (the shadow pass has its own)
fn draw_list_by(pass: &mut wgpu::RenderPass<'_>, sc: &SceneGpu, list: &[Draw], alphas: &[Alpha], want: impl Fn(Alpha) -> bool) {
    for d in list {
        let ib = &sc.inst_bufs[d.inst];
        if ib.count == 0 || !want(alphas[d.material]) {
            continue;
        }
        pass.set_bind_group(1, &sc.material_bgs[d.material], &[]);
        pass.set_vertex_buffer(0, d.vbuf.slice(..));
        pass.set_vertex_buffer(1, ib.buf.slice(..));
        pass.set_index_buffer(d.ibuf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..d.count, 0, 0..ib.count);
    }
}

/// A sun shadow projection over the sphere (centre, radius), reaching back towards the sun far
/// enough to catch every caster in the scene (scene_radius); the centre snaps to whole texels so
/// shadows do not crawl as the camera moves. Returns (matrix, texel size, depth range).
pub fn sun_matrix(sun_dir: Vec3, centre: Vec3, radius: f32, scene_centre: Vec3, scene_radius: f32) -> (Mat4, f32, f32) {
    let sun = sun_dir.normalize();
    let up = if sun.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let rot = Mat4::look_at_rh(Vec3::ZERO, -sun, up);
    let radius = radius.max(1.0).ceil();
    let texel = 2.0 * radius / SHADOW_SIZE as f32;
    let c = rot.transform_point3(centre);
    let (cx, cy) = ((c.x / texel).floor() * texel, (c.y / texel).floor() * texel);
    // depth along the light (-z): from the far side of the scene towards the sun to past the sphere
    let d_c = -c.z;
    let d_scene = -rot.transform_point3(scene_centre).z;
    let near = (d_scene - scene_radius).min(d_c - radius) - 1.0;
    let far = d_c + radius + 1.0;
    let proj = Mat4::orthographic_rh(cx - radius, cx + radius, cy - radius, cy + radius, near, far);
    (proj * rot, texel, far - near)
}

pub fn v4(v: Vec3, w: f32) -> [f32; 4] {
    Vec4::new(v.x, v.y, v.z, w).to_array()
}
