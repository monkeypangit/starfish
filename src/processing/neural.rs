//! Neural joint demosaic + denoise: runs the networks trained in `training/`
//! through `demosaic_run` (the `wgpu-run/` crate).
//!
//! Each network is built for one CFA phase (`Model::cfa`) and sizes that are a
//! multiple of its period, and pads circularly. So the frame is padded: on
//! every side by at least the network's reach (`Model::halo`), so no frame
//! pixel sees the opposite edge, and by up to one period more to land on the
//! model's phase and a whole number of periods. The padding mirrors the frame
//! across its edges in whole periods, which keeps every padded site's CFA
//! colour right. The output is cropped back to the frame's size on the GPU.
//!
//! The noise is fitted on the CPU. The frame goes up to the GPU as decoded,
//! and the padding as the frame row and column of every padded one; the
//! windows are cut from it, balanced and given their sigma, run through the
//! network and written to the output texture there (`Model::run_raw`).
//!
//! Debug builds read the exports from `training/*_wgpu/` at runtime, so a
//! re-export needs no rebuild; release builds embed them (see `build.rs`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use demosaic_run::{Model, RawInput, noise};
use wgpu::util::DeviceExt as _;

use super::{Demosaic, Gpu, IMAGE_FORMAT, Noise, texture_desc};
use crate::raw::{RawFrame, SensorKind};

#[cfg(not(debug_assertions))]
mod embedded {
    include!(concat!(env!("OUT_DIR"), "/models.rs"));
}

pub struct NeuralDemosaic {
    /// Export directory under `training/`, e.g. `xtrans_wgpu/f16`.
    export: String,
    name: String,
    /// Loaded on first use; the lock also serialises runs, which share the
    /// model's GPU buffers.
    model: Mutex<Option<Model>>,
}

impl NeuralDemosaic {
    /// `f16` picks the half-precision export, which needs `SHADER_F16`.
    pub fn new(kind: SensorKind, f16: bool) -> Self {
        let model = match kind {
            SensorKind::Bayer => "bayer",
            SensorKind::XTrans => "xtrans",
            SensorKind::Other => unreachable!("no neural model for {kind}"),
        };
        let precision = if f16 { "f16" } else { "f32" };
        Self {
            export: format!("{model}_wgpu/{precision}"),
            name: format!("Neural {kind} ({precision})"),
            model: Mutex::new(None),
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

#[cfg(debug_assertions)]
fn read_export(export: &str, file: &str) -> Result<Vec<u8>, String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("training")
        .join(export)
        .join(file);
    std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(not(debug_assertions))]
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
    ) -> Result<(wgpu::Texture, Option<Noise>), String> {
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
        let output = gpu.device.create_texture(&texture_desc(
            "demosaiced",
            (raw.width as u32, raw.height as u32),
            IMAGE_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        ));
        let input = RawInput {
            frame: &frame,
            rows: &rows.source,
            cols: &cols.source,
            gain: raw.wb,
            a: noise.a,
            b: noise.b,
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
        Ok((output, Some(noise)))
    }
}

fn ms(start: Instant) -> String {
    format!("{} ms", start.elapsed().as_millis())
}

/// Fits the noise on the frame cropped to the model's phase (`dy`, `dx`)
/// and whole periods. An image without enough flat area to fit is treated as
/// (almost) noise-free.
fn fit_noise(raw: &RawFrame, cfa: &[u32], period: usize, dy: usize, dx: usize) -> Noise {
    let height = (raw.height - dy) / period * period;
    let width = (raw.width - dx) / period * period;
    let mut phased = Vec::with_capacity(width * height);
    for y in dy..dy + height {
        let start = y * raw.width + dx;
        phased.extend_from_slice(&raw.data[start..start + width]);
    }
    let (a, b) = noise::estimate_noise(&phased, height, width, cfa, period)
        .unwrap_or((noise::A_RANGE.0, noise::B_RANGE.0));
    Noise { a, b }
}

/// One axis of the padded mosaic.
struct Padding {
    /// Padded pixels before the frame.
    before: usize,
    /// The frame pixel each padded pixel is copied from.
    source: Vec<u32>,
}

impl Padding {
    /// Pads `len` px, whose CFA starts `phase` px into the model's, by at
    /// least `margin` on both sides: onto the model's phase, to a whole number
    /// of periods.
    fn new(len: usize, phase: usize, margin: usize, period: usize) -> Self {
        // frame pixel `phase` must land on a period start
        let before = margin + (period - (margin + phase) % period) % period;
        let total = (before + len + margin).next_multiple_of(period);
        let source = (0..total)
            .map(|i| mirror(i as isize - before as isize, len, period) as u32)
            .collect();
        Self { before, source }
    }
}

/// The frame pixel for position `i` (frame coordinates, possibly outside
/// `0..len`): mirrored across the nearest edge in whole periods, i.e. the
/// periods next to the edge repeat in reverse order, so the source always
/// has the same CFA phase as `i`.
fn mirror(mut i: isize, len: usize, period: usize) -> usize {
    let (len, period) = (len as isize, period as isize);
    loop {
        if i < 0 {
            i -= period * (2 * i.div_euclid(period) + 1);
        } else if i >= len {
            i -= period * (2 * (i - len).div_euclid(period) + 1);
        } else {
            return i as usize;
        }
    }
}

/// The (row, column) offset that puts the frame's CFA onto the model's.
fn phase(raw: &RawFrame, cfa: &[u32], period: usize) -> Result<(usize, usize), String> {
    let (cw, ch) = (raw.cfa.width, raw.cfa.height);
    if period % cw != 0 || period % ch != 0 {
        return Err(format!("A {cw}×{ch} CFA does not repeat every {period} px"));
    }
    let color = |y: usize, x: usize| raw.cfa.colors[(y % ch) * cw + x % cw];
    (0..period * period)
        .map(|i| (i / period, i % period))
        .find(|&(dy, dx)| {
            (0..period).all(|y| (0..period).all(|x| color(dy + y, dx + x) == cfa[y * period + x]))
        })
        .ok_or_else(|| "The sensor's CFA does not match the model's in any phase".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_keeps_cfa_phase_and_frame() {
        for period in [2, 6] {
            for margin in [0, 22, 30] {
                for phase in 0..period {
                    for len in 2 * (margin + period)..2 * (margin + period) + 2 * period {
                        let pad = Padding::new(len, phase, margin, period);
                        let total = pad.source.len();
                        assert_eq!(total % period, 0);
                        assert_eq!(
                            (pad.before + phase) % period,
                            0,
                            "frame off the model's phase"
                        );
                        assert!(pad.before >= margin && total - pad.before - len >= margin);
                        for (i, &src) in pad.source.iter().enumerate() {
                            let src = src as usize;
                            let frame = i as isize - pad.before as isize;
                            assert!(src < len);
                            assert_eq!(
                                (frame - src as isize).rem_euclid(period as isize),
                                0,
                                "padded pixel {i} copies a different CFA phase"
                            );
                            if (0..len as isize).contains(&frame) {
                                assert_eq!(src as isize, frame);
                            }
                        }
                        // mirrored in whole periods: the period next to each edge comes first
                        if pad.before > 0 {
                            assert_eq!(pad.source[pad.before - 1] as usize, period - 1);
                        }
                        if total > pad.before + len {
                            assert_eq!(pad.source[pad.before + len] as usize, len - period);
                        }
                    }
                }
            }
        }
    }
}
