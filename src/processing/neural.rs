//! Neural joint demosaic + denoise: runs the networks trained in the `starfish-neural-demosaic` repo and exported into `models/`
//! through `demosaic_run` (the `wgpu-run/` crate).
//!
//! Each network is built for one CFA phase and sizes that are a multiple of
//! its period, so the frame is padded onto them (`demosaic_run::pad`, which
//! the web viewer shares). The output is cropped back to the frame's size on
//! the GPU.
//!
//! The noise is fitted on the CPU. The frame goes up to the GPU as decoded,
//! and the padding as the frame row and column of every padded one; the
//! windows are cut from it, balanced and given their sigma, run through the
//! network and written to the output texture there (`Model::run_raw`).
//!
//! Debug builds read the exports from `models/*_wgpu/` at runtime, so a
//! re-export needs no rebuild; release builds embed them (see `build.rs`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Instant;

use demosaic_run::pad::{self, Padding};
use demosaic_run::{Model, RawInput, noise};
use wgpu::util::DeviceExt as _;

use super::ca_fix::{Ca, CaFix};
use super::{Demosaic, Gpu, IMAGE_FORMAT, Noise, texture_desc};
use crate::lens::Measured;
use crate::raw::{RawFrame, SensorKind};

#[cfg(any(not(debug_assertions), feature = "embed-models"))]
mod embedded {
    include!(concat!(env!("OUT_DIR"), "/models.rs"));
}

pub struct NeuralDemosaic {
    /// Export directory under `models/`, e.g. `xtrans_wgpu/f16`.
    export: String,
    name: String,
    /// Loaded on first use; the lock also serialises runs, which share the
    /// model's GPU buffers.
    model: Mutex<Option<Model>>,
    /// The CA pass (`ca_fix`), made on first use.
    ca_fix: Mutex<Option<CaFix>>,
}

impl NeuralDemosaic {
    /// `variant` picks the network among the exports (see [`variants`]);
    /// `f16` the half-precision export, which needs `SHADER_F16`.
    pub fn new(kind: SensorKind, variant: &str, f16: bool) -> Self {
        let precision = if f16 { "f16" } else { "f32" };
        let label = match variant.trim_start_matches('_') {
            "" => String::new(),
            name => format!(" {name}"),
        };
        Self {
            export: format!("{}/{precision}", export_name(kind, variant)),
            name: format!("Neural {kind}{label} ({precision})"),
            model: Mutex::new(None),
            ca_fix: Mutex::new(None),
        }
    }

    fn load(&self, gpu: &Gpu) -> Result<Model, String> {
        let start = Instant::now();
        let read = |file: &str| read_export(&self.export, file);
        let model = Model::load_with(&read, &gpu.device, &gpu.queue)
            .map_err(|e| format!("Failed to load the {} model: {e}", self.name))?;
        crate::log::write(&format!("{}: model loaded in {}", self.name, ms(start)));
        Ok(model)
    }
}

/// The folder of `kind`'s network `variant`, e.g. `xtrans_lite_wgpu`.
pub fn export_name(kind: SensorKind, variant: &str) -> String {
    let model = match kind {
        SensorKind::Bayer => "bayer",
        SensorKind::XTrans => "xtrans",
        SensorKind::Other => unreachable!("no neural model for {kind}"),
    };
    format!("{model}{variant}_wgpu")
}

/// The variants of `kind`'s network that are exported, found once: the
/// `models/` folders `<bayer|xtrans><variant>_wgpu` with a graph in them
/// in debug builds (a new export shows up on the next start), the ones
/// embedded in release builds. The variant is what sits between the sensor
/// and `_wgpu`: `""` first, then the others by name.
pub fn variants(kind: SensorKind) -> &'static [&'static str] {
    static FOUND: OnceLock<[Vec<&'static str>; 2]> = OnceLock::new();
    let found = FOUND.get_or_init(|| {
        let exports = exports();
        [SensorKind::Bayer, SensorKind::XTrans].map(|kind| {
            let (prefix, suffix) = (export_name(kind, "").replace("_wgpu", ""), "_wgpu");
            let mut variants: Vec<&'static str> = exports
                .iter()
                .filter_map(|export| {
                    let variant = export.strip_prefix(&prefix)?.strip_suffix(suffix)?;
                    (variant.is_empty() || variant.starts_with('_')).then_some(variant)
                })
                .map(|variant| &*Box::leak(variant.to_owned().into_boxed_str()))
                .collect();
            variants.sort_unstable();
            variants.dedup();
            variants
        })
    });
    match kind {
        SensorKind::Bayer => &found[0],
        SensorKind::XTrans => &found[1],
        SensorKind::Other => &[],
    }
}

/// The names of the export folders (`xtrans_lite_wgpu`) that hold a graph.
#[cfg(all(debug_assertions, not(feature = "embed-models")))]
fn exports() -> Vec<String> {
    let models = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    let Ok(entries) = std::fs::read_dir(models) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| {
            ["f16", "f32"]
                .iter()
                .any(|precision| entry.path().join(precision).join("graph.txt").is_file())
        })
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect()
}

#[cfg(any(not(debug_assertions), feature = "embed-models"))]
fn exports() -> Vec<String> {
    embedded::EXPORTS
        .iter()
        .filter_map(|export| export.split_once('/'))
        .map(|(name, _)| name.to_owned())
        .collect()
}

#[cfg(all(debug_assertions, not(feature = "embed-models")))]
fn read_export(export: &str, file: &str) -> Result<Vec<u8>, String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(export)
        .join(file);
    std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(any(not(debug_assertions), feature = "embed-models"))]
fn read_export(export: &str, file: &str) -> Result<Vec<u8>, String> {
    embedded::file(export, file)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| format!("{export}/{file} is not embedded"))
}

impl Demosaic for NeuralDemosaic {
    fn name(&self) -> &str {
        &self.name
    }

    fn unload(&self) {
        *self.model.lock().unwrap_or_else(PoisonError::into_inner) = None;
        *self.ca_fix.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }

    fn estimate_noise(&self, gpu: &Gpu, raw: &RawFrame) -> Result<Option<Noise>, String> {
        let (cfa, period) = {
            let mut slot = self.model.lock().unwrap_or_else(PoisonError::into_inner);
            let model = match &mut *slot {
                Some(model) => model,
                empty => empty.insert(self.load(gpu)?),
            };
            (model.cfa().to_vec(), model.multiple() as usize)
        };
        let (dy, dx) = phase(raw, &cfa, period)?;
        let start = Instant::now();
        let noise = fit_noise(raw, &cfa, period, dy, dx);
        crate::log::write(&format!("{}: noise fitted in {}", self.name, ms(start)));
        Ok(Some(noise))
    }

    fn run(
        &self,
        gpu: &Gpu,
        raw: &RawFrame,
        noise: Option<Noise>,
        cancel: &AtomicBool,
    ) -> Result<(wgpu::Texture, Option<Noise>, Option<Measured>), String> {
        let cancelled = || cancel.load(Ordering::Relaxed);
        let start = Instant::now();
        let mut slot = self.model.lock().unwrap_or_else(PoisonError::into_inner);
        let waited = start.elapsed();
        if cancelled() {
            return Err(super::CANCELLED.into());
        }
        let model = match &mut *slot {
            Some(model) => model,
            empty => empty.insert(self.load(gpu)?),
        };
        let cfa = model.cfa();
        let period = model.multiple() as usize;

        let (dy, dx) = phase(raw, cfa, period)?;
        let margin = model.halo() as usize;
        if raw.height < 2 * (margin + period) || raw.width < 2 * (margin + period) {
            return Err(format!(
                "Image is too small to demosaic ({}×{})",
                raw.width, raw.height
            ));
        }

        let noise = noise.unwrap_or_else(|| fit_noise(raw, cfa, period, dy, dx));

        if cancelled() {
            return Err(super::CANCELLED.into());
        }

        let start = Instant::now();
        let rows = Padding::new(raw.height, dy, margin, period);
        let cols = Padding::new(raw.width, dx, margin, period);
        let frame = gpu.device.create_texture_with_data(
            &gpu.queue,
            &texture_desc(
                "raw frame",
                (raw.width as u32, raw.height as u32),
                wgpu::TextureFormat::R32Float,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ),
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&raw.data),
        );
        // the mosaic with the CA taken out (in the frame's coordinates), if
        // it could be measured
        let measure = Instant::now();
        let measured = if super::ca_fix() {
            raw.mosaic_ca().copied()
        } else {
            None
        };
        let measured_in = ms(measure);
        // the clipped samples rebuilt before the CA is taken out (`highlights`
        // reads each colour where it saw the same scene point), so the guide
        // green has no plateaus; else after
        let first = super::highlights() && super::highlights_first() && measured.is_some();
        let rebuilt = match (&measured, first) {
            (Some(m), true) => Some(super::highlights::frame(
                gpu,
                raw,
                None,
                Some(&Ca::new(raw.height, raw.width, m)),
            )?),
            _ => None,
        };
        let fixed = measured.map(|m| {
            let fix = Instant::now();
            let mut ca = Ca::new(raw.height, raw.width, &m);
            if first {
                ca = ca.unclipped();
            }
            let mut pass = self.ca_fix.lock().unwrap_or_else(PoisonError::into_inner);
            let fixed = pass
                .get_or_insert_with(|| CaFix::new(&gpu.device))
                .apply(gpu, rebuilt.as_ref().unwrap_or(&frame), &raw.cfa, &ca, raw.wb);
            crate::log::write(&format!(
                "{}: lateral CA ({}, measured in {measured_in}) taken out of the mosaic, queued in {}",
                self.name,
                m.summary(),
                ms(fix)
            ));
            (fixed, m)
        });
        // and, if asked, the clipped samples rebuilt
        let reconstructed = if super::highlights() && !first {
            Some(super::highlights::frame(
                gpu,
                raw,
                fixed.as_ref().map(|(texture, _)| texture),
                None,
            )?)
        } else {
            None
        };
        let frame = reconstructed
            .as_ref()
            .or(fixed.as_ref().map(|(texture, _)| texture))
            .unwrap_or(&frame);
        let output = gpu.device.create_texture(&texture_desc(
            "demosaiced",
            (raw.width as u32, raw.height as u32),
            IMAGE_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        ));
        let input = RawInput {
            frame,
            rows: &rows.source,
            cols: &cols.source,
            gain: raw.wb,
            a: noise.a,
            b: noise.b,
            filter_grain: super::filter_grain(),
        };
        let origin = (rows.before as u32, cols.before as u32);
        let queued = start.elapsed();
        let finished = model
            .run_raw(&input, &output, origin, &cancelled)
            .map_err(|e| format!("GPU poll failed: {e}"))?;
        // the last window may still be running: the readback waits for it
        crate::log::write(&format!(
            "{}: {}×{} padded, {} windows {} in {} (waited {} ms for the model, upload {} ms)",
            self.name,
            cols.source.len(),
            rows.source.len(),
            model.windows(rows.source.len() as u32, cols.source.len() as u32),
            if finished { "queued" } else { "cancelled" },
            ms(start),
            waited.as_millis(),
            queued.as_millis(),
        ));
        if !finished {
            return Err(super::CANCELLED.into());
        }
        Ok((output, Some(noise), fixed.map(|(_, m)| m)))
    }
}

fn ms(start: Instant) -> String {
    format!("{} ms", start.elapsed().as_millis())
}

/// Fits the noise on the frame cropped to the model's phase (`dy`, `dx`)
/// and whole periods. An image without enough flat area to fit is treated as
/// (almost) noise-free.
fn fit_noise(raw: &RawFrame, cfa: &[u32], period: usize, dy: usize, dx: usize) -> Noise {
    let (phased, height, width) = pad::phased(&raw.data, raw.height, raw.width, (dy, dx), period);
    let (a, b) = noise::estimate_noise(&phased, height, width, cfa, period)
        .unwrap_or((noise::A_RANGE.0, noise::B_RANGE.0));
    Noise { a, b }
}

/// The (row, column) offset that puts the frame's CFA onto the model's.
fn phase(raw: &RawFrame, cfa: &[u32], period: usize) -> Result<(usize, usize), String> {
    pad::phase(
        &raw.cfa.colors,
        (raw.cfa.width, raw.cfa.height),
        cfa,
        period,
    )
}
