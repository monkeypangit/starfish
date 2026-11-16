// SPDX-License-Identifier: GPL-3.0-or-later
//! Reference demosaics to compare the networks with: RCD for Bayer and
//! Markesteijn (3 passes) for X-Trans, ports of darktable's CPU code.
//!
//! **Licence.** These files are ports of GPL-3 code (see the headers), unlike
//! the rest of Starfish (MIT OR Apache-2.0), so they are only compiled with
//! the `gpl-demosaic` feature, and a binary built with it is GPL-3. They run
//! on the CPU, in all the cores, on the white balanced mosaic (as darktable
//! runs them, after its white balance module; RCD also divided by the
//! largest gain), and the result is uploaded. They do not denoise. With "Lateral CA
//! in the mosaic" on they get the mosaic with the CA taken out by `CaFix`,
//! as the networks do; off, the CA is measured in the output and drawn.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use wgpu::util::DeviceExt as _;

use super::ca_fix::{Ca, CaFix};
use super::{Demosaic, Gpu, IMAGE_FORMAT, Noise, texture_desc};
use crate::lens::Measured;
use crate::raw::{RawFrame, SensorKind};

mod markesteijn;
mod rcd;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reference {
    /// Ratio corrected demosaicing, Bayer.
    Rcd,
    /// Markesteijn with 3 passes, X-Trans.
    Markesteijn,
}

impl Reference {
    /// The sensor the method is for.
    pub fn sensor(self) -> SensorKind {
        match self {
            Reference::Rcd => SensorKind::Bayer,
            Reference::Markesteijn => SensorKind::XTrans,
        }
    }
}

pub struct ReferenceDemosaic {
    method: Reference,
    /// The CA pass, made on first use.
    ca_fix: Mutex<Option<CaFix>>,
}

impl ReferenceDemosaic {
    pub fn new(method: Reference) -> Self {
        Self {
            method,
            ca_fix: Mutex::new(None),
        }
    }

    /// `data` (`raw`'s mosaic, or that with the highlights rebuilt: then
    /// `unclipped`) with the lateral CA `m` taken out by `CaFix`, as a
    /// network is given it, read back.
    fn ca_fixed(
        &self,
        gpu: &Gpu,
        raw: &RawFrame,
        m: &Measured,
        data: &[f32],
        unclipped: bool,
    ) -> Result<Vec<f32>, String> {
        let (w, h) = (raw.width as u32, raw.height as u32);
        let frame = gpu.device.create_texture_with_data(
            &gpu.queue,
            &texture_desc(
                "raw frame",
                (w, h),
                wgpu::TextureFormat::R32Float,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ),
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(data),
        );
        let mut ca = Ca::new(raw.height, raw.width, m);
        if unclipped {
            ca = ca.unclipped();
        }
        let fixed = self
            .ca_fix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_or_insert_with(|| CaFix::new(&gpu.device))
            .apply(gpu, &frame, &raw.cfa, &ca, raw.wb);
        super::read_back_f32(gpu, &fixed)
    }
}

/// Runs `f(0..count)` in all the cores, results in order; `None` if `cancel`
/// was set.
fn par_map<T: Send>(
    count: usize,
    cancel: &AtomicBool,
    f: impl Fn(usize) -> T + Sync,
) -> Option<Vec<T>> {
    let next = AtomicUsize::new(0);
    let done = Mutex::new((0..count).map(|_| None).collect::<Vec<Option<T>>>());
    let threads = std::thread::available_parallelism()
        .map_or(1, |t| t.get())
        .min(count.max(1));
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= count || cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let value = f(k);
                    done.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)[k] = Some(value);
                }
            });
        }
    });
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    done.into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .into_iter()
        .collect()
}

/// `x` as IEEE half precision bits (round to nearest even).
fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if frac != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = frac | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = (m >> shift) as u16;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        return sign | (half + (rest > mid || (rest == mid && half & 1 == 1)) as u16);
    }
    let half = ((e as u32) << 10 | frac >> 13) as u16;
    let rest = frac & 0x1fff;
    sign | (half + (rest > 0x1000 || (rest == 0x1000 && half & 1 == 1)) as u16)
}

impl Demosaic for ReferenceDemosaic {
    fn unload(&self) {
        *self
            .ca_fix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    fn name(&self) -> &str {
        match self.method {
            Reference::Rcd => "RCD (darktable port)",
            Reference::Markesteijn => "Markesteijn 3-pass (darktable port)",
        }
    }

    fn run(
        &self,
        gpu: &Gpu,
        raw: &RawFrame,
        _noise: Option<Noise>,
        cancel: &AtomicBool,
    ) -> Result<(wgpu::Texture, Option<Noise>, Option<Measured>), String> {
        if raw.cfa.kind() != self.method.sensor() {
            return Err(format!(
                "{} is for {} sensors, this file has {}",
                self.name(),
                self.method.sensor(),
                raw.cfa.kind()
            ));
        }
        let start = std::time::Instant::now();
        // with "Lateral CA in the mosaic" on, the mosaic a network is given;
        // the highlights are rebuilt before the CA is taken out (with its
        // field) or after, as for the networks (`highlights_first`)
        let measured = super::ca_fix().then(|| raw.mosaic_ca().copied()).flatten();
        let first = super::highlights() && super::highlights_first() && measured.is_some();
        let log = |stats: super::highlights::Stats| {
            crate::log::write(&format!(
                "highlights: {} clipped areas, {} samples raised",
                stats.areas, stats.raised
            ))
        };
        let (mut fixed, mut rebuilt) = (None, None);
        match measured {
            Some(m) if first => {
                let mut data = raw.data.clone();
                let ca = Ca::new(raw.height, raw.width, &m);
                log(super::highlights::reconstruct(
                    &mut data, raw.width, raw.height, &raw.cfa, raw.wb, Some(&ca),
                ));
                fixed = Some(self.ca_fixed(gpu, raw, &m, &data, true)?);
            }
            Some(m) => fixed = Some(self.ca_fixed(gpu, raw, &m, &raw.data, false)?),
            None => {}
        }
        let corrected = measured;
        if super::highlights() && !first {
            let mut data = fixed.clone().unwrap_or_else(|| raw.data.clone());
            log(super::highlights::reconstruct(
                &mut data, raw.width, raw.height, &raw.cfa, raw.wb, None,
            ));
            rebuilt = Some(data);
        }
        let data = rebuilt.as_deref().or(fixed.as_deref()).unwrap_or(&raw.data);
        // darktable demosaics after its white balance module, so the data is
        // balanced; RCD also divides by the largest gain (its `scaler`) so
        // the data is about 0..1, which its epsilons are tuned for
        let scaler = if self.method == Reference::Rcd {
            raw.wb.iter().copied().fold(1.0, f32::max)
        } else {
            1.0
        };
        let gain = raw.wb.map(|g| g / scaler);
        let balanced: Vec<f32> = data
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let (y, x) = (i / raw.width, i % raw.width);
                v * gain[raw.cfa.colors[(y % raw.cfa.height) * raw.cfa.width + x % raw.cfa.width]
                    as usize]
            })
            .collect();
        let data = &balanced[..];
        let rgb = match self.method {
            Reference::Rcd => rcd::demosaic(data, raw.width, raw.height, &raw.cfa.colors, cancel),
            Reference::Markesteijn => {
                markesteijn::demosaic(data, raw.width, raw.height, &raw.cfa.colors, cancel)
            }
        }
        .ok_or(super::CANCELLED)?;
        // the white balanced image, with no grain in alpha
        let mut texels = Vec::with_capacity(raw.width * raw.height * 4);
        for px in rgb.chunks_exact(3) {
            texels.extend([
                f16_bits(px[0] * scaler),
                f16_bits(px[1] * scaler),
                f16_bits(px[2] * scaler),
                0,
            ]);
        }
        let texture = gpu.device.create_texture_with_data(
            &gpu.queue,
            &texture_desc(
                "reference demosaic",
                (raw.width as u32, raw.height as u32),
                IMAGE_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            ),
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&texels),
        );
        crate::log::write(&format!(
            "{}: demosaiced in {} ms",
            self.name(),
            start.elapsed().as_millis()
        ));
        Ok((texture, None, corrected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kodak images are in ../demosaic/images, outside the repo.
    fn kodak(n: u32) -> Option<(Vec<f32>, usize, usize)> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../demosaic/images/kodim{n:02}.png"));
        let image = image::open(path).ok()?.to_rgb8();
        let (w, h) = (
            image.width() as usize / 12 * 12,
            image.height() as usize / 12 * 12,
        );
        let mut rgb = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                rgb.extend(
                    image
                        .get_pixel(x as u32, y as u32)
                        .0
                        .map(|v| v as f32 / 255.0),
                );
            }
        }
        Some((rgb, w, h))
    }

    /// PSNR of `out` against `truth`, 24 px in from the edges.
    fn psnr(out: &[f32], truth: &[f32], w: usize, h: usize) -> f64 {
        let (mut sum, mut n) = (0f64, 0f64);
        for y in 24..h - 24 {
            for x in 24..w - 24 {
                for c in 0..3 {
                    let d = (out[(y * w + x) * 3 + c] - truth[(y * w + x) * 3 + c]) as f64;
                    sum += d * d;
                    n += 1.0;
                }
            }
        }
        -10.0 * (sum / n).log10()
    }

    fn mosaic(rgb: &[f32], w: usize, h: usize, colors: &[u32], size: usize) -> Vec<f32> {
        (0..w * h)
            .map(|i| rgb[i * 3 + colors[(i / w % size) * size + i % w % size] as usize])
            .collect()
    }

    /// The mean of each colour's samples in the 3x3 around a pixel (bilinear on Bayer).
    fn box3(mosaic: &[f32], w: usize, h: usize, colors: &[u32], size: usize) -> Vec<f32> {
        let colour = |y: usize, x: usize| colors[(y % size) * size + x % size] as usize;
        let mut out = vec![0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let (mut sum, mut count) = ([0f32; 3], [0f32; 3]);
                for yy in y.saturating_sub(1)..(y + 2).min(h) {
                    for xx in x.saturating_sub(1)..(x + 2).min(w) {
                        sum[colour(yy, xx)] += mosaic[yy * w + xx];
                        count[colour(yy, xx)] += 1.0;
                    }
                }
                for c in 0..3 {
                    out[(y * w + x) * 3 + c] = sum[c] / count[c].max(1.0);
                }
            }
        }
        out
    }

    /// Both methods on the mosaic of Kodak images against the originals.
    #[test]
    fn kodak_psnr() {
        let never = AtomicBool::new(false);
        let bayer = [0, 1, 1, 2];
        // the Fujifilm X-Trans pattern
        let xtrans = [
            1, 2, 1, 1, 0, 1, //
            0, 1, 0, 2, 1, 2, //
            1, 2, 1, 1, 0, 1, //
            1, 0, 1, 1, 2, 1, //
            2, 1, 2, 0, 1, 0, //
            1, 0, 1, 1, 2, 1,
        ];
        for n in [1, 5, 19, 21] {
            let Some((rgb, w, h)) = kodak(n) else {
                eprintln!("no Kodak images, skipped");
                return;
            };
            let out = rcd::demosaic(&mosaic(&rgb, w, h, &bayer, 2), w, h, &bayer, &never).unwrap();
            let rcd = psnr(&out, &rgb, w, h);
            let out = markesteijn::demosaic(&mosaic(&rgb, w, h, &xtrans, 6), w, h, &xtrans, &never)
                .unwrap();
            let markesteijn = psnr(&out, &rgb, w, h);
            let bilinear = psnr(
                &box3(&mosaic(&rgb, w, h, &bayer, 2), w, h, &bayer, 2),
                &rgb,
                w,
                h,
            );
            println!(
                "kodim{n:02}: RCD {rcd:.2} dB, Markesteijn 3-pass {markesteijn:.2} dB (bilinear {bilinear:.2})"
            );
            // far above the bilinear floor: a wrong index in the ports costs several dB
            assert!(rcd > bilinear + 4.0 && markesteijn > bilinear + 3.0);
        }
    }

    /// Demosaics a mosaic file with a reference method, for the post-filter
    /// searches in the scratchpad: `STARFISH_GPL_IN` = u32 h, u32 w, f32
    /// mosaic[h * w], `STARFISH_GPL_METHOD` = rcd | markesteijn,
    /// `STARFISH_GPL_OUT` gets f32 rgb[h * w * 3], interleaved.
    #[test]
    #[ignore]
    fn dump_reference() {
        let (Ok(input), Ok(out), Ok(method)) = (
            std::env::var("STARFISH_GPL_IN"),
            std::env::var("STARFISH_GPL_OUT"),
            std::env::var("STARFISH_GPL_METHOD"),
        ) else {
            return;
        };
        let b = std::fs::read(input).unwrap();
        let h = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let w = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
        let data: Vec<f32> = b[8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let never = AtomicBool::new(false);
        let rgb = if method == "rcd" {
            rcd::demosaic(&data, w, h, &[0, 1, 1, 2], &never).unwrap()
        } else {
            let xtrans = [
                1, 2, 1, 1, 0, 1, 0, 1, 0, 2, 1, 2, 1, 2, 1, 1, 0, 1, 1, 0, 1, 1, 2, 1, 2, 1, 2, 0, 1,
                0, 1, 0, 1, 1, 2, 1,
            ];
            markesteijn::demosaic(&data, w, h, &xtrans, &never).unwrap()
        };
        let bytes: Vec<u8> = rgb.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out, bytes).unwrap();
    }

    #[test]
    fn f16_bits_matches_known_values() {
        for (x, bits) in [
            (0.0, 0),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (0.5, 0x3800),
            (65504.0, 0x7bff),
            (1e-8, 0),
        ] {
            assert_eq!(f16_bits(x), bits, "{x}");
        }
        assert_eq!(f16_bits(6.1035156e-5), 0x0400);
        assert_eq!(f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f16_bits(1e9), 0x7c00);
    }
}
