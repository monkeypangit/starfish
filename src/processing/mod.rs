//! GPU processing engine.
//!
//! Runs on its own wgpu device (separate from iced's rendering device) so it
//! can request the adapter's full limits — iced's device only gets the
//! defaults (8192 px textures), which is smaller than many sensors.
//!
//! Pipeline: [`RawFrame`] → [`Demosaic`] → full-resolution `Rgba16Float`
//! texture (white-balanced camera RGB) → lens correction, if the file has a
//! profile (the app asks for CA only; distortion and vignetting are applied
//! where the image is drawn) → read back into an [`Image`].
//!
//! The demosaic stage is the neural joint demosaic + denoise
//! ([`NeuralDemosaic`]); `STARFISH_DEMOSAIC=binning` selects the fast binning
//! one instead.

mod binning;
mod lens;
mod neural;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use crate::lens::{Corrections, LensCorrection};
use crate::raw::{RawFrame, SensorKind, WhiteBalance};

pub use binning::BinningDemosaic;
use lens::LensPass;
pub use neural::NeuralDemosaic;

/// Format produced by every demosaic stage.
pub const IMAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const BYTES_PER_PIXEL: u32 = 8;
/// Upper bound for a single readback buffer; larger images are read in bands.
const MAX_READBACK_BYTES: u64 = 128 << 20;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    /// The first GPU error since the last [`Gpu::take_error`]. wgpu panics on
    /// uncaptured errors by default; recording them lets a failed develop
    /// (typically out of memory on a large file) report an error instead.
    error: Arc<Mutex<Option<String>>>,
}

impl Gpu {
    fn new() -> Result<Self, String> {
        #[cfg_attr(not(all(windows, target_arch = "x86_64")), allow(unused_mut))]
        let mut desc = wgpu::InstanceDescriptor::from_env_or_default();
        // FXC (wgpu's default) cannot compile f16 shaders; see Cargo.toml.
        #[cfg(all(windows, target_arch = "x86_64"))]
        if std::env::var_os("WGPU_DX12_COMPILER").is_none() {
            desc.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::StaticDxc;
        }
        let instance = wgpu::Instance::new(&desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| format!("No GPU adapter available: {e}"))?;
        let info = adapter.get_info();

        let request = |required_limits| {
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("starfish processing"),
                // The neural demosaic's f16 export runs faster where supported.
                required_features: adapter.features() & wgpu::Features::SHADER_F16,
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))
        };
        // Some drivers refuse their own maximum limits; then ask for the
        // defaults with only what the processing needs raised.
        let (device, queue) = request(adapter.limits())
            .or_else(|_| {
                let max = adapter.limits();
                request(wgpu::Limits {
                    max_texture_dimension_2d: max.max_texture_dimension_2d,
                    max_storage_buffer_binding_size: max.max_storage_buffer_binding_size,
                    max_buffer_size: max.max_buffer_size,
                    ..wgpu::Limits::default()
                })
            })
            .map_err(|e| format!("Failed to create GPU device: {e}"))?;

        let error = Arc::new(Mutex::new(None));
        let sink = error.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let message = match e {
                wgpu::Error::OutOfMemory { .. } => "Out of GPU memory".to_owned(),
                e => e.to_string(),
            };
            crate::log::write(&format!("GPU error: {message}"));
            sink.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert(message);
        }));
        let sink = error.clone();
        device.set_device_lost_callback(move |reason, message| {
            crate::log::write(&format!("GPU device lost ({reason:?}): {message}"));
            sink.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert(format!("The GPU device was lost: {message}"));
        });

        Ok(Self {
            device,
            queue,
            info,
            error,
        })
    }

    /// The first GPU error recorded since the last call, if any.
    fn take_error(&self) -> Option<String> {
        self.error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// One line naming the GPU and how it is used, for the metadata panel
    /// and bug reports.
    pub fn summary(&self) -> String {
        let f16 = if self.device.features().contains(wgpu::Features::SHADER_F16) {
            "f16"
        } else {
            "f32"
        };
        format!("{} ({:?}, {f16})", self.info.name, self.info.backend)
    }
}

/// Raw noise model `sigma^2 = a*x + b`, in normalised sensor units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Noise {
    pub a: f64,
    pub b: f64,
}

impl Noise {
    /// The same noise with sigma multiplied by `factor`.
    pub fn scaled(self, factor: f64) -> Self {
        let variance = factor * factor;
        Self {
            a: self.a * variance,
            b: self.b * variance,
        }
    }
}

/// A demosaic stage: turns a sensor mosaic into a full-resolution RGB image.
///
/// Implementations must return a texture of [`IMAGE_FORMAT`] with the same
/// dimensions as the raw frame, usable as `COPY_SRC`, and the noise they
/// denoised for: `noise` if given, else their own estimate (`None` if they
/// don't denoise).
/// The error of a run stopped through its cancel flag.
pub const CANCELLED: &str = "Cancelled";

pub trait Demosaic: Send + Sync {
    fn name(&self) -> &str;
    fn run(
        &self,
        gpu: &Gpu,
        raw: &RawFrame,
        noise: Option<Noise>,
        cancel: &AtomicBool,
    ) -> Result<(wgpu::Texture, Option<Noise>), String>;

    /// The noise this stage estimates in `raw`, `None` if it doesn't denoise.
    fn estimate_noise(&self, _gpu: &Gpu, _raw: &RawFrame) -> Result<Option<Noise>, String> {
        Ok(None)
    }

    /// Drops GPU resources loaded on first use, to be loaded again by the
    /// next run (after a GPU error may have left them invalid).
    fn unload(&self) {}
}

pub struct Engine {
    gpu: Gpu,
    bayer: Box<dyn Demosaic>,
    xtrans: Box<dyn Demosaic>,
    /// Quick low-quality stage, shown while the real one runs.
    preview: BinningDemosaic,
    lens: LensPass,
}

impl Engine {
    fn new() -> Result<Self, String> {
        let gpu = Gpu::new()?;
        crate::log::write(&format!("GPU: {}", gpu.summary()));
        let (bayer, xtrans): (Box<dyn Demosaic>, Box<dyn Demosaic>) =
            if std::env::var("STARFISH_DEMOSAIC").is_ok_and(|v| v == "binning") {
                let binning = || Box::new(BinningDemosaic::new(&gpu.device));
                (binning(), binning())
            } else {
                let f16 = gpu.device.features().contains(wgpu::Features::SHADER_F16);
                (
                    Box::new(NeuralDemosaic::new(SensorKind::Bayer, f16)),
                    Box::new(NeuralDemosaic::new(SensorKind::XTrans, f16)),
                )
            };
        let preview = BinningDemosaic::new(&gpu.device);
        let lens = LensPass::new(&gpu.device);
        Ok(Self {
            gpu,
            bayer,
            xtrans,
            preview,
            lens,
        })
    }

    /// The shared engine, created on first use.
    pub fn get() -> Result<&'static Engine, String> {
        static ENGINE: OnceLock<Result<Engine, String>> = OnceLock::new();
        ENGINE
            .get_or_init(Engine::new)
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The GPU in one line, see [`Gpu::summary`].
    pub fn gpu_summary(&self) -> String {
        self.gpu.summary()
    }

    /// The demosaic stage for `raw`'s sensor.
    fn stage(&self, raw: &RawFrame) -> Result<&dyn Demosaic, String> {
        match raw.cfa.kind() {
            SensorKind::Bayer => Ok(&*self.bayer),
            SensorKind::XTrans => Ok(&*self.xtrans),
            SensorKind::Other => Err(format!(
                "This sensor is not supported yet: Starfish handles Bayer and \
                 X-Trans sensors, and this file has a {}×{} colour filter pattern",
                raw.cfa.width, raw.cfa.height
            )),
        }
    }

    /// The noise the demosaic stage estimates in `raw`, `None` if it doesn't
    /// denoise.
    pub fn estimate_noise(&self, raw: &RawFrame) -> Result<Option<Noise>, String> {
        let stage = self.stage(raw)?;
        self.checked(raw, || stage.estimate_noise(&self.gpu, raw))
    }

    /// Runs `f` and turns a GPU error it caused into an `Err`.
    fn checked<T>(
        &self,
        raw: &RawFrame,
        f: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        // An error from an earlier run must not be blamed on this one.
        self.gpu.take_error();
        let result = f();
        let error = self.gpu.take_error();
        if error.is_some() {
            // A model loaded during the failure may hold invalid buffers.
            self.bayer.unload();
            self.xtrans.unload();
        }
        match error {
            Some(error) if error.starts_with("Out of GPU memory") => Err(format!(
                "Out of GPU memory: this {:.0} MP image needs more video memory than \
                 is free. Close other GPU-heavy programs, or turn on \"Quick demosaic\" under Debug.",
                (raw.width * raw.height) as f64 / 1e6
            )),
            Some(error) => Err(format!("GPU error: {error}")),
            None => result,
        }
    }

    /// Demosaics `raw` and reads the result back to the CPU. `noise`
    /// overrides the demosaic stage's own noise estimate. `lens` picks the
    /// corrections of the file's lens profile to apply, if it has one. `quick`
    /// uses the fast binning stage instead of the neural one (a debugging aid).
    /// Setting `cancel` stops the run early with [`CANCELLED`].
    pub fn develop(
        &self,
        raw: &RawFrame,
        noise: Option<Noise>,
        lens: Corrections,
        quick: bool,
        cancel: &AtomicBool,
    ) -> Result<Image, String> {
        let stage = self.stage(raw)?;
        let stage = if quick { &self.preview } else { stage };
        self.develop_with(stage, raw, noise, lens, quick, cancel)
    }

    /// Like [`Engine::develop`] with the fast binning stage: a rough
    /// full-size image in a fraction of the time.
    pub fn develop_preview(&self, raw: &RawFrame, lens: Corrections) -> Result<Image, String> {
        self.stage(raw)?;
        self.develop_with(&self.preview, raw, None, lens, true, &AtomicBool::new(false))
    }

    fn develop_with(
        &self,
        stage: &dyn Demosaic,
        raw: &RawFrame,
        noise: Option<Noise>,
        lens: Corrections,
        quick: bool,
        cancel: &AtomicBool,
    ) -> Result<Image, String> {
        let limit = self.gpu.device.limits().max_texture_dimension_2d as usize;
        if raw.width > limit || raw.height > limit {
            return Err(format!(
                "Image is {}×{} but the GPU supports at most {limit} px per side",
                raw.width, raw.height
            ));
        }

        self.checked(raw, || {
            self.develop_unchecked(stage, raw, noise, lens, quick, cancel)
        })
    }

    fn develop_unchecked(
        &self,
        stage: &dyn Demosaic,
        raw: &RawFrame,
        noise: Option<Noise>,
        lens: Corrections,
        quick: bool,
        cancel: &AtomicBool,
    ) -> Result<Image, String> {
        let kind = raw.cfa.kind();
        let start = std::time::Instant::now();
        let (texture, noise) = stage.run(&self.gpu, raw, noise, cancel)?;
        let applied = raw
            .lens
            .as_ref()
            .map_or(lens.and(Corrections::NONE), |l| lens.and(l.available()));
        let texture = match (&raw.lens, applied.any()) {
            (Some(profile), true) => self.lens.apply(&self.gpu, &texture, profile, applied),
            _ => texture,
        };
        let queued = start.elapsed();
        let pixels = read_back(&self.gpu, &texture)?;
        crate::log::write(&format!(
            "{}: {}×{} developed in {} ms (read back from {} ms on)",
            stage.name(),
            raw.width,
            raw.height,
            start.elapsed().as_millis(),
            queued.as_millis()
        ));

        Ok(Image {
            id: next_image_id(),
            width: texture.width(),
            height: texture.height(),
            pixels: Arc::new(pixels),
            cam_to_srgb: raw.cam_to_srgb,
            white: raw.white,
            name: raw
                .path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            camera: raw.camera.clone(),
            sensor: kind,
            demosaic: stage.name().to_owned(),
            noise,
            lens: applied,
            profile: raw.lens.clone().map(Arc::new),
            quick,
        })
    }
}

fn texture_desc(
    label: &'static str,
    (width, height): (u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    }
}

/// A demosaiced image ready for display.
#[derive(Clone)]
pub struct Image {
    /// Unique per develop, so the preview knows when to re-upload.
    pub id: u64,
    pub width: u32,
    pub height: u32,
    /// `Rgba16Float` texels as raw `f16` bits, row-major, no padding.
    pub pixels: Arc<Vec<u16>>,
    pub cam_to_srgb: [[f32; 3]; 3],
    pub white: Option<WhiteBalance>,
    pub name: String,
    pub camera: String,
    pub sensor: SensorKind,
    pub demosaic: String,
    /// What the demosaic stage denoised for, if it denoises.
    pub noise: Option<Noise>,
    /// The lens corrections that have been applied (the app asks for CA
    /// only; distortion and vignetting are applied where it is drawn).
    pub lens: Corrections,
    /// The raw file's lens profile, for the corrections applied when drawing.
    pub profile: Option<Arc<LensCorrection>>,
    /// Made by the quick binning stage rather than the neural one.
    pub quick: bool,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("id", &self.id)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

fn next_image_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Copies an `Rgba16Float` texture to CPU memory, in bands of rows so no
/// single staging buffer gets too large.
fn read_back(gpu: &Gpu, texture: &wgpu::Texture) -> Result<Vec<u16>, String> {
    let (width, height) = (texture.width(), texture.height());
    let row_bytes = width * BYTES_PER_PIXEL;
    let padded_row_bytes = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let band_rows = ((MAX_READBACK_BYTES / padded_row_bytes as u64) as u32).clamp(1, height);

    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: padded_row_bytes as u64 * band_rows as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut out: Vec<u16> = Vec::with_capacity((width * height * 4) as usize);
    let mut y = 0;
    while y < height {
        let rows = band_rows.min(height - y);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(rows),
                },
            },
            wgpu::Extent3d {
                width,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);

        let slice = staging.slice(..padded_row_bytes as u64 * rows as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU poll failed: {e}"))?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| format!("Failed to map readback buffer: {e}"))?;

        {
            let mapped = slice.get_mapped_range();
            for row in mapped.chunks_exact(padded_row_bytes as usize) {
                out.extend_from_slice(bytemuck::cast_slice(&row[..row_bytes as usize]));
            }
        }
        staging.unmap();
        y += rows;
    }

    Ok(out)
}

#[cfg(test)]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as f32;
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Develops `STARFISH_TEST_RAW` and writes a downscaled sRGB preview to
    /// `STARFISH_TEST_OUT` (PPM). `STARFISH_TEST_LENS=0` skips the file's lens
    /// correction; `STARFISH_TEST_QUICK=1` uses the quick binning stage;
    /// `STARFISH_TEST_DUMP` writes the full image as planar f32
    /// camera RGB. The image (PPM, dump) has all the lens corrections from
    /// `LensPass`; the export (`STARFISH_TEST_EXPORT`) is developed again
    /// as the app does it, CA only, with distortion and vignetting drawn,
    /// and its histogram computed.
    /// Run with:
    /// `STARFISH_TEST_RAW=file.raf STARFISH_TEST_OUT=out.ppm cargo test -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn develop_file() {
        let path = std::env::var("STARFISH_TEST_RAW").expect("set STARFISH_TEST_RAW");
        let raw = crate::raw::decode(std::path::Path::new(&path)).unwrap();
        println!(
            "{}: {}×{} {} wb={:?}",
            raw.camera,
            raw.width,
            raw.height,
            raw.cfa.kind(),
            raw.wb
        );
        let lens = match std::env::var("STARFISH_TEST_LENS").as_deref() {
            Ok("0") => Corrections::NONE,
            _ => Corrections::ALL,
        };
        let quick = std::env::var("STARFISH_TEST_QUICK").is_ok_and(|v| v != "0");
        let image = Engine::get()
            .unwrap()
            .develop(&raw, None, lens, quick, &AtomicBool::new(false))
            .unwrap();
        println!(
            "demosaic: {} (quick: {})\nlens profile: {:?}, applied: {:?}",
            image.demosaic,
            image.quick,
            raw.lens.as_ref().map(|l| l.description()),
            image.lens
        );
        if let Ok(dump) = std::env::var("STARFISH_TEST_DUMP") {
            let n = (image.width * image.height) as usize;
            let mut planes = vec![0f32; 3 * n];
            for i in 0..n {
                for c in 0..3 {
                    planes[c * n + i] = f16_to_f32(image.pixels[i * 4 + c]);
                }
            }
            std::fs::write(dump, bytemuck::cast_slice::<f32, u8>(&planes)).unwrap();
        }
        assert_eq!(
            image.pixels.len(),
            (image.width * image.height * 4) as usize
        );

        // The app's export with the default adjustments plus some local
        // contrast; `STARFISH_TEST_CROP=1` also turns it a quarter clockwise
        // and crops 3:2 to 1/1.6 of the width, tilted by ~4.6 degrees.
        if let Ok(export) = std::env::var("STARFISH_TEST_EXPORT") {
            use crate::preview::{Adjustments, Aspect, Crop, Rotation};
            let adjustments = Adjustments {
                local_contrast: 0.3,
                ..Adjustments::default()
            };
            let (rotation, crop) = if std::env::var("STARFISH_TEST_CROP").is_ok_and(|v| v != "0") {
                let crop = Crop {
                    zoom: 1.6,
                    angle: 0.08,
                    aspect: Aspect::Ratio(3, 2),
                    ..Crop::default()
                };
                (Rotation::CLOCKWISE, crop)
            } else {
                (Rotation::default(), Crop::default())
            };
            let format = if export.ends_with(".png") {
                crate::export::Format::Png16
            } else {
                crate::export::Format::JpegL
            };
            let path = std::path::Path::new(&export);
            let engine = Engine::get().unwrap();
            let drawn = engine
                .develop(&raw, None, lens.and(Corrections::CHROMATIC), quick, &AtomicBool::new(false))
                .unwrap();
            let view = crate::export::View {
                adjustments,
                rotation,
                crop,
                lens: lens.without(Corrections::CHROMATIC),
            };
            let gpu = &engine.gpu;
            let start = std::time::Instant::now();
            crate::export::export(&gpu.device, &gpu.queue, &drawn, view, format, path).unwrap();
            println!("GPU export: {} ms", start.elapsed().as_millis());
            let start = std::time::Instant::now();
            crate::histogram::Histogram::compute_on(&gpu.device, &gpu.queue, &drawn, view).unwrap();
            println!("histogram: {} ms", start.elapsed().as_millis());
        }

        let Ok(out) = std::env::var("STARFISH_TEST_OUT") else {
            return;
        };
        let step = (image.width / 800).max(1);
        let (w, h) = (image.width / step, image.height / step);
        let m = image.cam_to_srgb;
        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
        for y in 0..h {
            for x in 0..w {
                let i = (((y * step) * image.width + x * step) * 4) as usize;
                let cam: Vec<f32> = (0..3).map(|c| f16_to_f32(image.pixels[i + c])).collect();
                for row in m {
                    let v = row[0] * cam[0] + row[1] * cam[1] + row[2] * cam[2];
                    ppm.push((v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8);
                }
            }
        }
        std::fs::write(out, ppm).unwrap();
    }
}
