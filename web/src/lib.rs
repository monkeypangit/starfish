//! Starfish's neural demosaic in the browser: one raw file in, the network's
//! output on a canvas, with zoom, pan and exposure. The page (`index.html`)
//! does the I/O and the input; this does the decoding, the noise fit and the
//! GPU work, through the same `demosaic_run::Model::run_raw` as the app.
//!
//! Per file: `open` decodes it and names the model export it needs; the page
//! fetches that export once (`add_model`); `develop` fits the noise and
//! queues the network into a mipmapped texture; `developed` says when the
//! GPU has finished; `draw` shows it.
//!
//! Only built for wasm32; see build.sh.
#![cfg(target_arch = "wasm32")]

mod raw;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use demosaic_run::pad::{self, Padding};
use demosaic_run::{Model, RawInput, noise};
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt as _;

/// The network's output, and so the shown image.
const IMAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The factor on the estimated noise sigma for noise reduction `strength`
/// (0..1), as the app's slider (`noise_factor` in main.rs): `(9^s - 1) / 2`,
/// so 0 turns denoising off, 0.5 uses the estimate as is and 1 is 4×.
fn noise_factor(strength: f32) -> f64 {
    (9f64.powf(f64::from(strength)) - 1.0) / 2.0
}

#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&info.to_string().into());
    }));
}

/// The draw shader's uniform (`View` in show.wgsl).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct View {
    matrix: [[f32; 4]; 3],
    center: [f32; 2],
    canvas: [f32; 2],
    scale: f32,
    exposure: f32,
    encode: u32,
    saturation: f32,
}

/// A developed image, ready to draw.
struct Image {
    width: u32,
    height: u32,
    cam_to_srgb: [[f32; 3]; 3],
    bind: wgpu::BindGroup,
}

#[wasm_bindgen]
pub struct Viewer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: String,
    f16: bool,
    /// The first GPU error since the last `take_error`.
    error: Arc<Mutex<Option<String>>>,
    /// By export name, e.g. "xtrans_wgpu/f16".
    models: HashMap<String, Model>,
    show: wgpu::RenderPipeline,
    mip: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    view: wgpu::Buffer,
    /// The opened file, kept for developing it again.
    frame: Option<raw::Frame>,
    /// Its fitted noise (a, b), once known.
    noise: Option<(f64, f64)>,
    image: Option<Image>,
    /// Set when the GPU has finished the last `develop`.
    done: Arc<AtomicBool>,
}

#[wasm_bindgen]
impl Viewer {
    /// Sets up WebGPU on `canvas`.
    pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Result<Viewer, JsError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| JsError::new(&format!("No WebGPU canvas: {e}")))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(|e| JsError::new(&format!("No WebGPU adapter: {e}")))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("starfish web"),
                // the f16 exports run faster where supported
                required_features: adapter.features() & wgpu::Features::SHADER_F16,
                // big sensors pass the default 8192 px texture limit
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| JsError::new(&format!("Failed to create the GPU device: {e}")))?;
        let f16 = device.features().contains(wgpu::Features::SHADER_F16);
        let info = adapter.get_info();
        let gpu = format!(
            "{} ({})",
            if info.name.is_empty() { "WebGPU" } else { &info.name },
            if f16 { "f16" } else { "f32" }
        );

        let error = Arc::new(Mutex::new(None));
        let sink = error.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let message = match e {
                wgpu::Error::OutOfMemory { .. } => "Out of GPU memory".to_owned(),
                e => e.to_string(),
            };
            web_sys::console::error_1(&format!("GPU error: {message}").into());
            sink.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert(message);
        }));
        let sink = error.clone();
        device.set_device_lost_callback(move |reason, message| {
            web_sys::console::error_1(&format!("GPU device lost ({reason:?}): {message}").into());
            sink.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert(format!("The GPU device was lost: {message}"));
        });

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: canvas.width().max(1),
            height: canvas.height().max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let show = pipeline(&device, include_str!("show.wgsl"), format);
        let mip = pipeline(&device, include_str!("mip.wgsl"), IMAGE_FORMAT);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let view = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"),
            size: std::mem::size_of::<View>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Viewer {
            device,
            queue,
            surface,
            config,
            gpu,
            f16,
            error,
            models: HashMap::new(),
            show,
            mip,
            sampler,
            view,
            frame: None,
            noise: None,
            image: None,
            done: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The adapter and the precision the models run at.
    pub fn gpu(&self) -> String {
        self.gpu.clone()
    }

    /// Decodes a raw file, keeping it for `develop`. Returns the model export
    /// it needs, e.g. "xtrans_wgpu/f16".
    pub fn open(&mut self, bytes: &[u8]) -> Result<String, JsError> {
        self.frame = None;
        self.noise = None;
        let frame = raw::decode(bytes).map_err(|e| JsError::new(&e))?;
        let model = match frame.cfa_size {
            (2, 2) => "bayer",
            (6, 6) => "xtrans",
            (w, h) => return Err(JsError::new(&format!("No model for a {w}×{h} CFA"))),
        };
        let limit = self.device.limits().max_texture_dimension_2d as usize;
        if frame.width > limit || frame.height > limit {
            return Err(JsError::new(&format!(
                "{}×{} is larger than this GPU's {limit} px textures",
                frame.width, frame.height
            )));
        }
        self.frame = Some(frame);
        Ok(format!("{model}_wgpu/{}", if self.f16 { "f16" } else { "f32" }))
    }

    /// The camera of the opened file.
    pub fn camera(&self) -> String {
        self.frame.as_ref().map(|f| f.camera.clone()).unwrap_or_default()
    }

    pub fn has_model(&self, export: &str) -> bool {
        self.models.contains_key(export)
    }

    /// Loads a model export from its files, `names[i]` holding `files[i]`.
    pub fn add_model(
        &mut self,
        export: String,
        names: Vec<String>,
        files: Vec<js_sys::Uint8Array>,
    ) -> Result<(), JsError> {
        let files: HashMap<String, js_sys::Uint8Array> = names.into_iter().zip(files).collect();
        let read = |name: &str| {
            files
                .get(name)
                .map(|f| f.to_vec())
                .ok_or_else(|| format!("{export}/{name} was not fetched"))
        };
        let model = Model::load_with(&read, &self.device, &self.queue)
            .map_err(|e| JsError::new(&format!("Failed to load the model: {e}")))?;
        self.models.insert(export, model);
        Ok(())
    }

    /// Queues the network on the opened file, with `export` (loaded by
    /// `add_model`) and noise reduction `strength` (0..1, as the app's
    /// slider); the first time the noise is fitted. Returns a line on what
    /// was done; `developed` says when the GPU has finished.
    pub fn develop(&mut self, export: &str, strength: f32) -> Result<String, JsError> {
        let frame = self.frame.as_ref().ok_or_else(|| JsError::new("No file is open"))?;
        let model = self
            .models
            .get(export)
            .ok_or_else(|| JsError::new(&format!("The {export} model is not loaded")))?;
        let cfa = model.cfa();
        let period = model.multiple() as usize;
        let (dy, dx) =
            pad::phase(&frame.cfa, frame.cfa_size, cfa, period).map_err(|e| JsError::new(&e))?;
        let margin = model.halo() as usize;
        if frame.height < 2 * (margin + period) || frame.width < 2 * (margin + period) {
            return Err(JsError::new(&format!(
                "Image is too small to demosaic ({}×{})",
                frame.width, frame.height
            )));
        }

        let start = now();
        let measured = self.noise.is_none();
        let fit = match self.noise {
            Some(fit) => fit,
            None => {
                let (phased, h, w) =
                    pad::phased(&frame.data, frame.height, frame.width, (dy, dx), period);
                // an image without enough flat area to fit is treated as (almost) noise-free
                let fit = noise::estimate_noise(&phased, h, w, cfa, period)
                    .unwrap_or((noise::A_RANGE.0, noise::B_RANGE.0));
                self.noise = Some(fit);
                fit
            }
        };
        let fitted = now();
        // sigma scales by the factor, so the variance's a and b by its square
        let variance = noise_factor(strength).powi(2);
        let (a, b) = (fit.0 * variance, fit.1 * variance);

        let (width, height) = (frame.width as u32, frame.height as u32);
        let rows = Padding::new(frame.height, dy, margin, period);
        let cols = Padding::new(frame.width, dx, margin, period);
        let input = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("raw frame"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&frame.data),
        );
        let levels = 32 - width.max(height).leading_zeros();
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("demosaiced"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: IMAGE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let raw_input = RawInput {
            frame: &input,
            rows: &rows.source,
            cols: &cols.source,
            gain: frame.wb,
            a,
            b,
        };
        // the browser polls the device itself, so this only queues the windows
        model
            .run_raw(&raw_input, &texture, (rows.before as u32, cols.before as u32), &|| false)
            .map_err(|e| JsError::new(&format!("GPU poll failed: {e}")))?;
        let windows = model.windows(rows.source.len() as u32, cols.source.len() as u32);
        drop(input);
        self.fill_mips(&texture, levels);
        let queued = now();

        self.done.store(false, Ordering::Relaxed);
        let done = self.done.clone();
        self.queue
            .on_submitted_work_done(move || done.store(true, Ordering::Relaxed));

        let view = texture.create_view(&Default::default());
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("show"),
            layout: &self.show.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.view.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.image = Some(Image {
            width,
            height,
            cam_to_srgb: frame.cam_to_srgb,
            bind,
        });
        Ok(format!(
            "noise σ² = {:.2e}·x + {:.2e}{}, noise reduction {:.0}%, {windows} windows queued in {:.0} ms",
            fit.0,
            fit.1,
            if measured { format!(" (fit {:.0} ms)", fitted - start) } else { String::new() },
            strength * 100.0,
            queued - fitted,
        ))
    }

    /// Whether the GPU has finished the last `develop`.
    pub fn developed(&self) -> bool {
        self.done.load(Ordering::Relaxed)
    }

    /// The first GPU error since the last call, if any.
    pub fn take_error(&self) -> Option<String> {
        self.error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// The developed image's size, 0 before there is one.
    pub fn width(&self) -> u32 {
        self.image.as_ref().map_or(0, |i| i.width)
    }

    pub fn height(&self) -> u32 {
        self.image.as_ref().map_or(0, |i| i.height)
    }

    /// Sets the canvas size in device pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    /// Draws the image at `scale` canvas px per image px, with image point
    /// (`cx`, `cy`) at the canvas centre, `ev` stops brighter and with
    /// `saturation` (1 = unchanged).
    pub fn draw(&self, scale: f32, cx: f32, cy: f32, ev: f32, saturation: f32) -> Result<(), JsError> {
        let frame = self
            .surface
            .get_current_texture()
            .map_err(|e| JsError::new(&format!("No canvas texture: {e}")))?;
        let target = frame.texture.create_view(&Default::default());
        let size = (self.config.width, self.config.height);
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.render(&mut enc, &target, size, (scale, cx, cy, ev, saturation));
        self.queue.submit([enc.finish()]);
        frame.present();
        Ok(())
    }

    /// The view `draw` would show on a `width` x `height` canvas, as RGBA8
    /// (sRGB-encoded, row-major): resolves to a `Uint8Array`.
    #[allow(clippy::too_many_arguments)]
    pub fn snapshot(
        &self,
        width: u32,
        height: u32,
        scale: f32,
        cx: f32,
        cy: f32,
        ev: f32,
        saturation: f32,
    ) -> js_sys::Promise {
        let format = self.config.format;
        let bgra = match format.remove_srgb_suffix() {
            wgpu::TextureFormat::Rgba8Unorm => false,
            wgpu::TextureFormat::Bgra8Unorm => true,
            other => {
                return js_sys::Promise::reject(&format!("Cannot read back {other:?}").into());
            }
        };
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("snapshot"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let row = (4 * width).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("snapshot"),
            size: (row * height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        let target = texture.create_view(&Default::default());
        self.render(&mut enc, &target, (width, height), (scale, cx, cy, ev, saturation));
        enc.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            size,
        );
        self.queue.submit([enc.finish()]);
        js_sys::Promise::new(&mut |resolve, reject| {
            let mapped = buffer.clone();
            buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if let Err(e) = result {
                    let _ = reject.call1(&JsValue::NULL, &e.to_string().into());
                    return;
                }
                let mut pixels = Vec::with_capacity((4 * width * height) as usize);
                {
                    let data = mapped.slice(..).get_mapped_range();
                    for line in data.chunks(row as usize) {
                        pixels.extend_from_slice(&line[..4 * width as usize]);
                    }
                }
                mapped.unmap();
                if bgra {
                    pixels.chunks_mut(4).for_each(|p| p.swap(0, 2));
                }
                let _ = resolve.call1(&JsValue::NULL, &js_sys::Uint8Array::from(&pixels[..]));
            });
        })
    }
}

impl Viewer {
    /// Records the view onto `target`, a `width` x `height` texture in the
    /// canvas format.
    fn render(
        &self,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        (width, height): (u32, u32),
        (scale, cx, cy, ev, saturation): (f32, f32, f32, f32, f32),
    ) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("show"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.11,
                        g: 0.11,
                        b: 0.12,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        if let Some(image) = &self.image {
            let m = image.cam_to_srgb;
            let view = View {
                matrix: m.map(|r| [r[0], r[1], r[2], 0.0]),
                center: [cx, cy],
                canvas: [width as f32, height as f32],
                scale,
                exposure: ev.exp2(),
                encode: !self.config.format.is_srgb() as u32,
                saturation,
            };
            self.queue.write_buffer(&self.view, 0, bytemuck::bytes_of(&view));
            pass.set_pipeline(&self.show);
            pass.set_bind_group(0, &image.bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Fills mip levels 1.. of `texture` from level 0.
    fn fill_mips(&self, texture: &wgpu::Texture, levels: u32) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        let level = |l: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: l,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        for l in 1..levels {
            let above = level(l - 1);
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mip"),
                layout: &self.mip.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&above),
                }],
            });
            let target = level(l);
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mip"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.mip);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
    }
}

/// A full-canvas triangle pipeline from a shader with `vs` and `fs`.
fn pipeline(device: &wgpu::Device, source: &str, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview: None,
        cache: None,
    })
}

/// Milliseconds, for timings.
fn now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or(0.0, |p| p.now())
}
