//! GPU processing engine.
//!
//! Runs on its own wgpu device (separate from iced's rendering device) so it
//! can request the adapter's full limits — iced's device only gets the
//! defaults (8192 px textures), which is smaller than many sensors.
//!
//! Pipeline: [`RawFrame`] → [`Demosaic`] → full-resolution `Rgba16Float`
//! texture (white-balanced camera RGB) → lens correction, if asked for and
//! the file has a profile (the app asks for none: every correction is drawn
//! where the image is, see [`Image::profile`]) → read back into an
//! [`Image`], whose lateral CA is then measured (`lens::measure`).
//!
//! The demosaic stage is the neural joint demosaic + denoise
//! ([`NeuralDemosaic`]); `STARFISH_DEMOSAIC=binning` selects the fast binning
//! one instead. [`Method`] picks between the network, the light and tiny
//! networks for weaker GPUs and the binning stage per develop.

mod binning;
mod ca_fix;
#[cfg(feature = "gpl-demosaic")]
mod gpl;
mod highlights;
pub mod hot_pixels;
mod lens;
mod mosaic;
mod neural;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use crate::lens::{Corrections, LensCorrection, Measured};
use crate::raw::{RawFrame, SensorKind, WhiteBalance};

pub use binning::BinningDemosaic;
#[cfg(feature = "gpl-demosaic")]
use gpl::{Reference, ReferenceDemosaic};
use lens::LensPass;
use mosaic::MosaicDemosaic;
pub use neural::{NeuralDemosaic, variants};

/// Debugging: whether the networks' output alpha, the grain the grain slider
/// adds back, is the mean of the middle 5 of the 3x3 grains around each pixel
/// instead of its own (`RawInput::filter_grain`), for every develop from now
/// on. Process-wide so it need not pass through every stage;
/// `Image::filter_grain` records what a develop used. On by default.
static FILTER_GRAIN: AtomicBool = AtomicBool::new(true);

pub fn set_filter_grain(on: bool) {
    FILTER_GRAIN.store(on, Ordering::Relaxed);
}

pub fn filter_grain() -> bool {
    FILTER_GRAIN.load(Ordering::Relaxed)
}

/// Debugging: whether a network gets a mosaic with the lateral CA taken out
/// first (`ca_fix`, the mosaic's measured
/// CA; then nothing is drawn) instead of having it measured in its output
/// and drawn, for every develop from now on; `Image::ca_fix`. On by default.
static CA_FIX: AtomicBool = AtomicBool::new(true);

pub fn set_ca_fix(on: bool) {
    CA_FIX.store(on, Ordering::Relaxed);
}

pub fn ca_fix() -> bool {
    CA_FIX.load(Ordering::Relaxed)
}

/// Whether the hot and dead pixels of the mosaic are replaced by their
/// neighbours' median (`hot_pixels`) before anything else, for every develop
/// from now on; `Image::hot_pixel_fix`. On by default.
static HOT_PIXELS: AtomicBool = AtomicBool::new(true);

pub fn set_hot_pixels(on: bool) {
    HOT_PIXELS.store(on, Ordering::Relaxed);
}

pub fn hot_pixels_fix() -> bool {
    HOT_PIXELS.load(Ordering::Relaxed)
}

/// The sites the user picked as defective in the open file, `(y, x)` in the
/// raw frame, and a number that changes with every set, which an [`Image`]
/// records so a re-develop knows it is out of date. They are replaced with
/// the automatic ones, if [`hot_pixels_fix`] is on.
static PICKED: Mutex<(u64, Vec<(usize, usize)>)> = Mutex::new((0, Vec::new()));

pub fn set_picked(sites: Vec<(usize, usize)>) {
    let mut picked = PICKED.lock().unwrap_or_else(PoisonError::into_inner);
    *picked = (picked.0 + 1, sites);
}

pub fn picked() -> (u64, Vec<(usize, usize)>) {
    PICKED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// Debugging: the clipped samples of the mosaic are rebuilt from the
/// unclipped ones (`highlights`), before the CA is taken out (see
/// [`highlights_first`]) and before the demosaic, for every develop from now on; `Image::highlights`. Off by default.
static HIGHLIGHTS: AtomicBool = AtomicBool::new(false);

pub fn set_highlights(on: bool) {
    HIGHLIGHTS.store(on, Ordering::Relaxed);
}

pub fn highlights() -> bool {
    HIGHLIGHTS.load(Ordering::Relaxed)
}

/// With the CA taken out of the mosaic and the highlights rebuilt, the
/// rebuilding runs first, on the mosaic with its CA (each colour read where
/// it saw the same scene point, `highlights::reconstruct`'s `ca`), and `CaFix`
/// gets the result; `STARFISH_HIGHLIGHTS_AFTER_CA=1` gives the old order
/// (the CA out first, then the highlights) to compare.
pub fn highlights_first() -> bool {
    static FIRST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FIRST.get_or_init(|| std::env::var_os("STARFISH_HIGHLIGHTS_AFTER_CA").is_none())
}

/// Raw samples (white = 1, not balanced) at or above this count as clipped,
/// everywhere in the app: the highlight reconstruction, the CA fix's fallback
/// and its measurement. (wgpu-run's grain test has its own
/// copy, `wgpu-run/src/lib.rs`'s `CLIPPED`.)
pub const CLIPPED: f32 = 0.98;

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
    /// The demosaiced image, the noise it denoised for and the lateral CA
    /// taken out of its mosaic, if it was (`CaFix`).
    fn run(
        &self,
        gpu: &Gpu,
        raw: &RawFrame,
        noise: Option<Noise>,
        cancel: &AtomicBool,
    ) -> Result<(wgpu::Texture, Option<Noise>, Option<Measured>), String>;

    /// The noise this stage estimates in `raw`, `None` if it doesn't denoise.
    fn estimate_noise(&self, _gpu: &Gpu, _raw: &RawFrame) -> Result<Option<Noise>, String> {
        Ok(None)
    }

    /// Drops GPU resources loaded on first use, to be loaded again by the
    /// next run (after a GPU error may have left them invalid).
    fn unload(&self) {}
}

/// Which demosaic stage develops an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// A network exported under `models/`: the variant is what sits between
    /// the sensor and `_wgpu` in the folder's name (`""` for
    /// `models/{bayer,xtrans}_wgpu/`, `"_lite"` for `*_lite_wgpu/`), as
    /// [`variants`] found it.
    Network(&'static str),
    /// The fast binning stage, without a network (a debugging aid).
    Quick,
    /// The raw mosaic itself, CA taken out of it if [`set_ca_fix`] is on (a
    /// debugging aid).
    Mosaic,
    /// A reference demosaic to compare the networks with: RCD (Bayer) or
    /// Markesteijn 3-pass (X-Trans), see `gpl/`.
    #[cfg(feature = "gpl-demosaic")]
    Reference(Reference),
}

impl Method {
    /// The reference demosaics for `kind`'s files (every sensor's before one is open).
    #[cfg(feature = "gpl-demosaic")]
    fn references(kind: Option<SensorKind>) -> impl Iterator<Item = Method> {
        [Reference::Rcd, Reference::Markesteijn]
            .into_iter()
            .filter(move |reference| kind.is_none_or(|kind| kind == reference.sensor()))
            .map(Method::Reference)
    }

    #[cfg(not(feature = "gpl-demosaic"))]
    fn references(_: Option<SensorKind>) -> impl Iterator<Item = Method> {
        std::iter::empty()
    }

    /// Whether the stage denoises, so there is a noise to fit and set.
    pub fn denoises(self) -> bool {
        matches!(self, Method::Network(_))
    }

    /// The full network.
    pub const NEURAL: Method = Method::Network("");
    /// The network at half the width, for weaker GPUs.
    const LIGHT: &'static str = "_lite";

    /// What the picker offers for `kind`'s files (every sensor's before one
    /// is open): the exported networks, then the quick stage.
    pub fn available(kind: Option<SensorKind>) -> Vec<Method> {
        let mut variants: Vec<&'static str> = match kind {
            Some(kind) => variants(kind).to_vec(),
            None => [SensorKind::Bayer, SensorKind::XTrans]
                .into_iter()
                .flat_map(|kind| variants(kind).iter().copied())
                .collect(),
        };
        variants.sort_unstable();
        variants.dedup();
        variants
            .into_iter()
            .map(Method::Network)
            .chain([Method::Quick, Method::Mosaic])
            .chain(Method::references(kind))
            .collect()
    }
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Method::Network("") => f.write_str("Neural network"),
            Method::Network("_lite") => f.write_str("Light network"),
            Method::Network("_tiny") => f.write_str("Tiny network"),
            Method::Network(variant) => f.write_str(variant.trim_start_matches('_')),
            Method::Quick => f.write_str("Quick"),
            Method::Mosaic => f.write_str("Mosaic"),
            #[cfg(feature = "gpl-demosaic")]
            Method::Reference(Reference::Rcd) => f.write_str("RCD"),
            #[cfg(feature = "gpl-demosaic")]
            Method::Reference(Reference::Markesteijn) => f.write_str("Markesteijn 3-pass"),
        }
    }
}

pub struct Engine {
    gpu: Gpu,
    /// Each sensor's networks by [`variants`] name.
    bayer: Vec<(&'static str, Box<dyn Demosaic>)>,
    xtrans: Vec<(&'static str, Box<dyn Demosaic>)>,
    /// Quick low-quality stage, shown while the real one runs.
    preview: BinningDemosaic,
    mosaic: MosaicDemosaic,
    #[cfg(feature = "gpl-demosaic")]
    rcd: ReferenceDemosaic,
    #[cfg(feature = "gpl-demosaic")]
    markesteijn: ReferenceDemosaic,
    lens: LensPass,
}

impl Engine {
    fn new() -> Result<Self, String> {
        let gpu = Gpu::new()?;
        crate::log::write(&format!("GPU: {}", gpu.summary()));
        let binning = std::env::var("STARFISH_DEMOSAIC").is_ok_and(|v| v == "binning");
        let f16 = gpu.device.features().contains(wgpu::Features::SHADER_F16);
        let stages = |kind| {
            variants(kind)
                .iter()
                .map(|&variant| -> (&'static str, Box<dyn Demosaic>) {
                    if binning {
                        (variant, Box::new(BinningDemosaic::new(&gpu.device)))
                    } else {
                        (variant, Box::new(NeuralDemosaic::new(kind, variant, f16)))
                    }
                })
                .collect::<Vec<_>>()
        };
        let (bayer, xtrans) = (stages(SensorKind::Bayer), stages(SensorKind::XTrans));
        let preview = BinningDemosaic::new(&gpu.device);
        let mosaic = MosaicDemosaic::new(&gpu.device);
        let lens = LensPass::new(&gpu.device);
        Ok(Self {
            gpu,
            bayer,
            xtrans,
            preview,
            mosaic,
            #[cfg(feature = "gpl-demosaic")]
            rcd: ReferenceDemosaic::new(Reference::Rcd),
            #[cfg(feature = "gpl-demosaic")]
            markesteijn: ReferenceDemosaic::new(Reference::Markesteijn),
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

    /// The method to use when none is chosen: the light network on an
    /// integrated (or software) GPU if `raw`'s sensor has one, else the
    /// full one.
    pub fn automatic(&self, raw: &RawFrame) -> Method {
        match self.gpu.info.device_type {
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu
                if variants(raw.cfa.kind()).contains(&Method::LIGHT) =>
            {
                Method::Network(Method::LIGHT)
            }
            _ => Method::NEURAL,
        }
    }

    /// The demosaic stage `method` uses for `raw`'s sensor. Choosing one
    /// network lets go of the others' GPU buffers, so no two are loaded at
    /// once.
    fn stage(&self, raw: &RawFrame, method: Method) -> Result<&dyn Demosaic, String> {
        let kind = raw.cfa.kind();
        let networks = match kind {
            SensorKind::Bayer => &self.bayer,
            SensorKind::XTrans => &self.xtrans,
            SensorKind::Other => {
                return Err(format!(
                    "This sensor is not supported yet: Starfish handles Bayer and \
                 X-Trans sensors, and this file has a {}×{} colour filter pattern",
                    raw.cfa.width, raw.cfa.height
                ));
            }
        };
        let wanted = match method {
            Method::Network(wanted) => wanted,
            Method::Quick => return Ok(&self.preview),
            Method::Mosaic => return Ok(&self.mosaic),
            #[cfg(feature = "gpl-demosaic")]
            Method::Reference(Reference::Rcd) => return Ok(&self.rcd),
            #[cfg(feature = "gpl-demosaic")]
            Method::Reference(Reference::Markesteijn) => return Ok(&self.markesteijn),
        };
        let mut chosen = None;
        for (variant, stage) in networks {
            if *variant == wanted {
                chosen = Some(&**stage);
            } else {
                stage.unload();
            }
        }
        chosen.ok_or_else(|| {
            format!(
                "No {} export for {kind} sensors",
                neural::export_name(kind, wanted)
            )
        })
    }

    /// The noise `method`'s demosaic stage estimates in `raw`, `None` if it
    /// doesn't denoise.
    pub fn estimate_noise(&self, raw: &RawFrame, method: Method) -> Result<Option<Noise>, String> {
        let stage = self.stage(raw, method)?;
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
            for (_, stage) in self.bayer.iter().chain(&self.xtrans) {
                stage.unload();
            }
        }
        match error {
            Some(error) if error.starts_with("Out of GPU memory") => Err(format!(
                "Out of GPU memory: this {:.0} MP image needs more video memory than \
                 is free. Close other GPU-heavy programs, or set Debug → Demosaic to \"Quick\".",
                (raw.width * raw.height) as f64 / 1e6
            )),
            Some(error) => Err(format!("GPU error: {error}")),
            None => result,
        }
    }

    /// Demosaics `raw` and reads the result back to the CPU. `noise`
    /// overrides the demosaic stage's own noise estimate. `lens` picks the
    /// corrections of the file's lens profile to apply, if it has one.
    /// `method` picks the demosaic stage. Setting `cancel` stops the run early
    /// with [`CANCELLED`].
    pub fn develop(
        &self,
        raw: &RawFrame,
        noise: Option<Noise>,
        lens: Corrections,
        method: Method,
        cancel: &AtomicBool,
    ) -> Result<Image, String> {
        let stage = self.stage(raw, method)?;
        self.develop_with(stage, raw, noise, lens, method, true, cancel)
    }

    /// Like [`Engine::develop`] with the fast binning stage: a rough
    /// full-size image in a fraction of the time. Its lateral CA is not
    /// measured, so it has no CA correction.
    pub fn develop_preview(&self, raw: &RawFrame, lens: Corrections) -> Result<Image, String> {
        let stage = self.stage(raw, Method::Quick)?;
        self.develop_with(
            stage,
            raw,
            None,
            lens,
            Method::Quick,
            false,
            &AtomicBool::new(false),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn develop_with(
        &self,
        stage: &dyn Demosaic,
        raw: &RawFrame,
        noise: Option<Noise>,
        lens: Corrections,
        method: Method,
        measure: bool,
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
            self.develop_unchecked(stage, raw, noise, lens, method, measure, cancel)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn develop_unchecked(
        &self,
        stage: &dyn Demosaic,
        raw: &RawFrame,
        noise: Option<Noise>,
        lens: Corrections,
        method: Method,
        measure: bool,
        cancel: &AtomicBool,
    ) -> Result<Image, String> {
        let kind = raw.cfa.kind();
        let start = std::time::Instant::now();
        let filter_grain = filter_grain();
        let ca_fix = ca_fix();
        let highlights = highlights();
        let hot_pixel_fix = hot_pixels_fix();
        // the stages read `raw.data`, so they get a copy with the defects
        // replaced (found ones and picked ones); the lateral CA is measured
        // on the original first, once
        let (picked_version, picks) = picked();
        let cleaned;
        let mut hot_pixels = None;
        let mut picked_count = 0;
        let raw = if hot_pixel_fix {
            let mut all = raw.hot_pixels().to_vec();
            let manual =
                hot_pixels::replacements(&raw.data, raw.width, raw.height, &raw.cfa, &picks);
            picked_count = manual.len();
            for (i, v) in manual {
                match all.binary_search_by_key(&i, |f| f.0) {
                    Ok(k) => all[k].1 = v,
                    Err(k) => all.insert(k, (i, v)),
                }
            }
            hot_pixels = Some(all.len());
            if all.is_empty() {
                raw
            } else {
                crate::log::write(&format!(
                    "hot and dead pixels: {} replaced ({picked_count} picked)",
                    all.len()
                ));
                let _ = raw.mosaic_ca();
                let mut copy = raw.clone();
                for (i, v) in all {
                    copy.data[i] = v;
                }
                cleaned = copy;
                &cleaned
            }
        } else {
            raw
        };
        let (texture, noise, corrected_ca) = stage.run(&self.gpu, raw, noise, cancel)?;
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
        // In what the stage made, unless the CA was taken out of its mosaic.
        // The networks remove part of it on their own, so it is measured
        // after them.
        let ca = if measure && corrected_ca.is_none() && method != Method::Mosaic {
            crate::lens::measure(texture.width(), texture.height(), &pixels)
        } else {
            None
        };
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
            sensor: kind,
            demosaic: stage.name().to_owned(),
            noise,
            profile: LensCorrection::drawn(
                raw.lens.as_ref(),
                ca.as_ref(),
                texture.width(),
                texture.height(),
            )
            .map(Arc::new),
            ca,
            corrected_ca,
            method,
            filter_grain,
            ca_fix,
            highlights,
            hot_pixel_fix,
            hot_pixels,
            picked_version,
            picked: picked_count,
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
    pub sensor: SensorKind,
    pub demosaic: String,
    /// What the demosaic stage denoised for, if it denoises.
    pub noise: Option<Noise>,
    /// The lens profile for the corrections applied when drawing: the raw
    /// file's distortion and vignetting and the measured CA
    /// ([`LensCorrection::drawn`]).
    pub profile: Option<Arc<LensCorrection>>,
    /// The lateral CA measured in it, if it could be, which is drawn.
    pub ca: Option<Measured>,
    /// The lateral CA taken out of the mosaic before the demosaic (the
    /// mosaic's, see [`set_ca_fix`]); then `ca` is not measured.
    pub corrected_ca: Option<Measured>,
    /// The demosaic stage that made it.
    pub method: Method,
    /// Whether its alpha holds the filtered grain (see [`set_filter_grain`]).
    pub filter_grain: bool,
    /// Whether the CA was to be taken out of the mosaic (see [`set_ca_fix`]).
    pub ca_fix: bool,
    /// Whether the clipped samples of the mosaic were rebuilt (see [`set_highlights`]).
    pub highlights: bool,
    /// Whether the hot and dead pixels were to be replaced (see [`set_hot_pixels`]).
    pub hot_pixel_fix: bool,
    /// How many were replaced; `None` if the switch was off.
    pub hot_pixels: Option<usize>,
    /// The [`set_picked`] version it was made with, and how many of the
    /// picked sites it replaced.
    pub picked_version: u64,
    pub picked: usize,
}

impl Image {
    /// The raw mosaic of `raw` as an image to pick defective sites in: each
    /// site's white balanced sample in its own colour's channel, the others
    /// 0, with no defect removed, CA fix or lens profile. Same size as `raw`,
    /// so a pixel here is the same site there.
    pub fn raw_view(&self, raw: &RawFrame) -> Image {
        use half::f16;
        let mut pixels = vec![0u16; raw.width * raw.height * 4];
        for (i, &v) in raw.data.iter().enumerate() {
            let (y, x) = (i / raw.width, i % raw.width);
            let c =
                raw.cfa.colors[(y % raw.cfa.height) * raw.cfa.width + x % raw.cfa.width] as usize;
            pixels[i * 4 + c] = f16::from_f32(v * raw.wb[c]).to_bits();
        }
        Image {
            id: next_image_id(),
            pixels: Arc::new(pixels),
            noise: None,
            profile: None,
            ca: None,
            corrected_ca: None,
            method: Method::Mosaic,
            demosaic: String::from("Raw mosaic"),
            ..self.clone()
        }
    }
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

/// Copies an `R32Float` texture to CPU memory.
fn read_back_f32(gpu: &Gpu, texture: &wgpu::Texture) -> Result<Vec<f32>, String> {
    let (width, height) = (texture.width(), texture.height());
    let padded_row_bytes = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: padded_row_bytes as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("GPU poll failed: {e:?}"))?;
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("Readback failed: {e}"))?;
    let mapped = slice.get_mapped_range();
    let mut out = Vec::with_capacity((width * height) as usize);
    for row in mapped.chunks_exact(padded_row_bytes as usize) {
        out.extend_from_slice(bytemuck::cast_slice::<u8, f32>(&row[..width as usize * 4]));
    }
    drop(mapped);
    staging.unmap();
    Ok(out)
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
    /// correction; `STARFISH_TEST_QUICK=1` uses the quick binning stage,
    /// `STARFISH_TEST_MOSAIC=1` the mosaic itself,
    /// `STARFISH_TEST_MODEL=_lite` the network exported as `*_lite_wgpu`;
    /// `STARFISH_TEST_DUMP` writes the full image as planar f32
    /// camera RGB. The image (PPM, dump) has all the lens corrections from
    /// `LensPass`, CA from the file's table; the export (`STARFISH_TEST_EXPORT`)
    /// is developed again as the app does it, with every correction drawn
    /// and CA as measured, and its histogram computed.
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
        let set = |var| std::env::var(var).is_ok_and(|v| v != "0");
        // `STARFISH_TEST_GRAIN_FILTER=1`: the grain's 3x3 middle-5 mean
        set_filter_grain(set("STARFISH_TEST_GRAIN_FILTER"));
        // `STARFISH_TEST_CA_FIX=0`: the CA drawn, not taken out of the mosaic
        set_ca_fix(std::env::var("STARFISH_TEST_CA_FIX").as_deref() != Ok("0"));
        // `STARFISH_TEST_HIGHLIGHTS=1`: the clipped samples of the mosaic rebuilt
        set_highlights(set("STARFISH_TEST_HIGHLIGHTS"));
        // `STARFISH_TEST_HOT_PIXELS=0`: the hot and dead pixels left in
        set_hot_pixels(std::env::var("STARFISH_TEST_HOT_PIXELS").as_deref() != Ok("0"));
        let method = if set("STARFISH_TEST_QUICK") {
            Method::Quick
        } else if set("STARFISH_TEST_MOSAIC") {
            Method::Mosaic
        } else if cfg!(feature = "gpl-demosaic") && set("STARFISH_TEST_REFERENCE") {
            Method::available(Some(raw.cfa.kind()))
                .into_iter()
                .find(|method| {
                    !matches!(method, Method::Network(_) | Method::Quick | Method::Mosaic)
                })
                .expect("no reference demosaic for this sensor")
        } else if let Ok(model) = std::env::var("STARFISH_TEST_MODEL") {
            let variant = variants(raw.cfa.kind())
                .iter()
                .find(|variant| **variant == model)
                .unwrap_or_else(|| panic!("no {model} export for this sensor"));
            Method::Network(variant)
        } else {
            Method::NEURAL
        };
        let engine = Engine::get().unwrap();
        let noise = engine.estimate_noise(&raw, method).unwrap();
        let start = std::time::Instant::now();
        let image = engine
            .develop(&raw, noise, lens, method, &AtomicBool::new(false))
            .unwrap();
        println!(
            "develop: {} ms (the noise fitted before)",
            start.elapsed().as_millis()
        );
        println!(
            "demosaic: {} ({:?})\nlens profile: {:?}, CA measured after LensPass: {:?}",
            image.demosaic,
            image.method,
            raw.lens.as_ref().map(|l| l.description()),
            image.ca.map(|ca| ca.summary())
        );
        // `STARFISH_TEST_GRAIN_DUMP`: the alpha, the grain, as f32
        if let Ok(dump) = std::env::var("STARFISH_TEST_GRAIN_DUMP") {
            let grain: Vec<f32> = image
                .pixels
                .chunks_exact(4)
                .map(|p| f16_to_f32(p[3]))
                .collect();
            std::fs::write(dump, bytemuck::cast_slice::<f32, u8>(&grain)).unwrap();
        }
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
        // contrast, fully denoised unless `STARFISH_TEST_GRAIN_AMOUNT` (0..1)
        // puts grain back (`STARFISH_TEST_SHARPNESS` 0..1 sharpens); `STARFISH_TEST_CROP=1` also turns it a quarter clockwise
        // and crops 3:2 to 1/1.6 of the width, tilted by ~4.6 degrees.
        if let Ok(export) = std::env::var("STARFISH_TEST_EXPORT") {
            use crate::preview::{Adjustments, Aspect, Crop, Rotation};
            let mut adjustments = Adjustments {
                local_contrast: 0.3,
                sharpness: std::env::var("STARFISH_TEST_SHARPNESS")
                    .map_or(0.0, |v| v.parse().unwrap()),
                grain: std::env::var("STARFISH_TEST_GRAIN_AMOUNT")
                    .map_or(0.0, |v| v.parse().unwrap()),
                ..Adjustments::default()
            };
            // `STARFISH_TEST_COLOR_EDITS=1`: strong colour mixer and grading settings.
            if std::env::var("STARFISH_TEST_COLOR_EDITS").is_ok_and(|v| v != "0") {
                use crate::color_edit::{GradeParam, MixParam};
                let (mix, grading) = (&mut adjustments.color_mix, &mut adjustments.grading);
                mix.set(0, MixParam::Saturation, -1.0); // red: grey
                mix.set(2, MixParam::Hue, 1.0); // yellow: turned 30 degrees
                mix.set(3, MixParam::Brightness, 1.0); // green: lighter
                mix.set(5, MixParam::Saturation, 1.0); // blue: more colour
                grading.set(0, GradeParam::Hue, 250.0);
                grading.set(0, GradeParam::Amount, 1.0); // blue shadows
                grading.set(2, GradeParam::Hue, 60.0);
                grading.set(2, GradeParam::Amount, 1.0); // orange highlights
            }
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
                crate::export::Format::Jpeg
            };
            let path = std::path::Path::new(&export);
            let engine = Engine::get().unwrap();
            let drawn = engine
                .develop(
                    &raw,
                    None,
                    Corrections::NONE,
                    method,
                    &AtomicBool::new(false),
                )
                .unwrap();
            let start = std::time::Instant::now();
            crate::lens::measure(drawn.width, drawn.height, &drawn.pixels);
            println!(
                "measured CA (drawn): {:?} in {} ms",
                drawn.ca.map(|ca| ca.summary()),
                start.elapsed().as_millis()
            );
            let view = crate::export::View {
                adjustments,
                rotation,
                crop,
                lens,
            };
            let gpu = &engine.gpu;
            let start = std::time::Instant::now();
            crate::export::export(
                &gpu.device,
                &gpu.queue,
                &drawn,
                view,
                format,
                95,
                crate::export::Compression::Low,
                path,
            )
            .unwrap();
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
