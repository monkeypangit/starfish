//! Highlight reconstruction on the raw mosaic, in the spirit of darktable's
//! "inpaint opposed" (written from the idea, not ported from its code): where
//! one colour clips and the others do not, the clipped samples are rebuilt
//! from the unclipped ones, with the colour taken from the unclipped ring
//! around the clipped area. It runs on the CPU and works on any CFA. Either
//! after the lateral CA is taken out (`CaFix`), so the channels line up, or
//! before it, with the CA field (`ca`): then every colour's samples are read
//! at the place that saw the scene point of the sample being rebuilt (`at`),
//! and `CaFix` gets a mosaic whose guide green has no plateaus.
//!
//! Per clipped area (a 8-connected set of pixels with a clipped sample within
//! `dilate()` px, 3 by default, 1 until 2026-11-16):
//!
//! 1. The colour of the surroundings: the mean of each colour's unclipped,
//!    white balanced samples within `RING` px of the area. If a colour has
//!    none there, the colour is neutral (balanced 1:1:1).
//! 2. For each clipped sample of colour `c`, the colours `k` with unclipped
//!    samples (and none clipped) in the 5x5 around it give their local means
//!    `v_k`; the estimate of `c` is the mean of `v_k * ring_c / ring_k`: the
//!    local brightness at the surroundings' colour. With no colour left (all
//!    clipped) the estimate is the smallest brightening of the surroundings'
//!    colour that puts every channel at or above its clip level.
//! 3. The sample becomes the larger of itself and the estimate, at most
//!    `HEADROOM` x its clip level (white balanced), so noise or a tiny ring
//!    mean cannot blow a pixel up. Unclipped samples are never touched.

use super::ca_fix::Ca;
use crate::raw::CfaPattern;

/// Raw samples at or above this (white = 1) are clipped.
pub use super::CLIPPED;
/// How far a clipped sample's mask reaches, px (an area is made of the pixels
/// within this of a clipped sample, and the ring starts outside it).
/// `STARFISH_HIGHLIGHTS_DILATE` overrides it, to compare.
fn dilate() -> usize {
    static DILATE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DILATE.get_or_init(|| {
        std::env::var("STARFISH_HIGHLIGHTS_DILATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3)
    })
}
/// The ring around a clipped area the colour is taken from, px.
const RING: isize = 3;
/// The largest value a reconstructed sample takes, in its clip level.
const HEADROOM: f32 = 4.0;
/// Smallest ring mean divided by, balanced.
const FLOOR: f32 = 1e-4;

/// What a run found.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Clipped areas.
    pub areas: usize,
    /// Clipped samples that were raised.
    pub raised: usize,
}

/// Fills in the clipped samples of `data` (`w` x `h`, as decoded: black level
/// subtracted, white = 1, not white balanced; `wb` its gains R, G, B).
/// `ca`: the lateral CA still in `data`, if any.
pub fn reconstruct(
    data: &mut [f32],
    w: usize,
    h: usize,
    cfa: &CfaPattern,
    wb: [f32; 3],
    ca: Option<&Ca>,
) -> Stats {
    reconstruct_dilated(data, w, h, cfa, wb, ca, dilate())
}

/// [`reconstruct`] with the mask reaching `r` px from a clipped sample.
fn reconstruct_dilated(
    data: &mut [f32],
    w: usize,
    h: usize,
    cfa: &CfaPattern,
    wb: [f32; 3],
    ca: Option<&Ca>,
    r: usize,
) -> Stats {
    let (wi, hi) = (w as isize, h as isize);
    // where colour `k`'s samples saw what the sample of colour `c` at (y, x)
    // sees: the offset to add to (y, x) to read them (0 without CA)
    let at = |k: usize, c: usize, y: isize, x: isize| -> (isize, isize) {
        match ca {
            Some(ca) => {
                let (dk, dc) = (ca.field(k, y, x), ca.field(c, y, x));
                ((dk[0] - dc[0]).round() as isize, (dk[1] - dc[1]).round() as isize)
            }
            None => (0, 0),
        }
    };
    let colour = |y: isize, x: isize| {
        cfa.colors[(y as usize % cfa.height) * cfa.width + x as usize % cfa.width] as usize
    };
    let clipped = |data: &[f32], i: usize| data[i] >= CLIPPED;
    let clip_level: [f32; 3] = wb.map(|g| g * CLIPPED);

    // the pixels with a clipped sample within `dilate()`: separable dilation
    let mut hit: Vec<u8> = data.iter().map(|&v| (v >= CLIPPED) as u8).collect();
    if !hit.contains(&1) {
        return Stats::default();
    }
    let mut wide = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r + 1).min(w);
            wide[y * w + x] = hit[y * w + lo..y * w + hi].iter().any(|&v| v != 0) as u8;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let any = (y.saturating_sub(r)..(y + r + 1).min(h)).any(|yy| wide[yy * w + x] != 0);
            hit[y * w + x] = any as u8;
        }
    }
    let mask = hit;

    let mut stats = Stats::default();
    let mut label = vec![0u32; w * h];
    let mut seen = vec![0u32; w * h];
    let mut fills: Vec<(usize, f32)> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut area: Vec<usize> = Vec::new();

    for start in 0..w * h {
        if mask[start] == 0 || label[start] != 0 {
            continue;
        }
        stats.areas += 1;
        let id = stats.areas as u32;
        // flood the area
        area.clear();
        stack.push(start);
        label[start] = id;
        while let Some(p) = stack.pop() {
            area.push(p);
            let (y, x) = ((p / w) as isize, (p % w) as isize);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (yy, xx) = (y + dy, x + dx);
                    if yy < 0 || xx < 0 || yy >= hi || xx >= wi {
                        continue;
                    }
                    let q = yy as usize * w + xx as usize;
                    if mask[q] != 0 && label[q] == 0 {
                        label[q] = id;
                        stack.push(q);
                    }
                }
            }
        }

        // 1. the colour of the surroundings (each colour read where it saw
        // the area's scene points: green's frame position is the scene's)
        let (mut sum, mut count) = ([0f32; 3], [0u32; 3]);
        for &p in &area {
            let (y, x) = ((p / w) as isize, (p % w) as isize);
            let own = colour(y, x);
            for k in 0..3 {
                let (oy, ox) = at(k, own, y, x);
                for yy in (y + oy - RING).max(0)..=(y + oy + RING).min(hi - 1) {
                    for xx in (x + ox - RING).max(0)..=(x + ox + RING).min(wi - 1) {
                        let q = yy as usize * w + xx as usize;
                        // outside every area, so unclipped
                        if colour(yy, xx) != k || mask[q] != 0 || seen[q] == id {
                            continue;
                        }
                        seen[q] = id;
                        sum[k] += data[q] * wb[k];
                        count[k] += 1;
                    }
                }
            }
        }
        let ring = if count.iter().all(|&n| n > 0) {
            [0, 1, 2].map(|c| (sum[c] / count[c] as f32).max(FLOOR))
        } else {
            [1.0; 3]
        };
        // all clipped: the smallest brightening of the ring's colour that
        // takes every channel to its clip level
        let brightening = (0..3).map(|c| clip_level[c] / ring[c]).fold(0.0, f32::max);

        // 2. and 3. each clipped sample
        for &p in &area {
            if !clipped(data, p) {
                continue;
            }
            let (y, x) = ((p / w) as isize, (p % w) as isize);
            let c = colour(y, x);
            let (mut sum, mut count, mut bad) = ([0f32; 3], [0u32; 3], [false; 3]);
            for k in 0..3 {
                // colour k's samples that saw the scene around this sample's
                let (oy, ox) = at(k, c, y, x);
                for yy in (y + oy - 2).max(0)..=(y + oy + 2).min(hi - 1) {
                    for xx in (x + ox - 2).max(0)..=(x + ox + 2).min(wi - 1) {
                        if colour(yy, xx) != k {
                            continue;
                        }
                        let q = yy as usize * w + xx as usize;
                        if clipped(data, q) {
                            bad[k] = true;
                        } else {
                            sum[k] += data[q] * wb[k];
                            count[k] += 1;
                        }
                    }
                }
            }
            let (mut total, mut n) = (0.0, 0);
            for k in 0..3 {
                if !bad[k] && count[k] > 0 {
                    total += sum[k] / count[k] as f32 * ring[c] / ring[k];
                    n += 1;
                }
            }
            let estimate = if n > 0 {
                total / n as f32
            } else {
                brightening * ring[c]
            };
            let balanced = (data[p] * wb[c])
                .max(estimate)
                .min(HEADROOM * clip_level[c]);
            let value = balanced / wb[c];
            if value > data[p] {
                fills.push((p, value));
            }
        }
    }
    stats.raised = fills.len();
    for (p, v) in fills {
        data[p] = v;
    }
    stats
}

/// `raw`'s mosaic (or `base`, the same with the CA taken out, an `R32Float`
/// texture with `COPY_SRC`) with the highlights reconstructed, as a new
/// `R32Float` texture. `ca`: the CA still in the mosaic (`base` is `None`).
pub fn frame(
    gpu: &super::Gpu,
    raw: &crate::raw::RawFrame,
    base: Option<&wgpu::Texture>,
    ca: Option<&Ca>,
) -> Result<wgpu::Texture, String> {
    use wgpu::util::DeviceExt as _;
    let start = std::time::Instant::now();
    let mut data = match base {
        Some(texture) => super::read_back_f32(gpu, texture)?,
        None => raw.data.clone(),
    };
    let read = start.elapsed();
    let stats = reconstruct(&mut data, raw.width, raw.height, &raw.cfa, raw.wb, ca);
    crate::log::write(&format!(
        "highlights: {} clipped areas, {} samples raised, {} ms ({} ms of it reading the mosaic back)",
        stats.areas,
        stats.raised,
        start.elapsed().as_millis(),
        read.as_millis()
    ));
    Ok(gpu.device.create_texture_with_data(
        &gpu.queue,
        &super::texture_desc(
            "highlights frame",
            (raw.width as u32, raw.height as u32),
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        ),
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&data),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BAYER: [u32; 4] = [0, 1, 1, 2];

    /// A warm blob (red far brighter than green, blue weakest) whose red
    /// clips: the clipped red comes back closer to the truth, nothing else
    /// moves. The blob's red / green ratio rises from 1.7 at the ring to 2.0
    /// at the peak, and the ring's ratio is what is used, so the peak stays
    /// somewhat low: the error is about halved, not removed.
    #[test]
    fn rebuilds_a_clipped_channel() {
        let (w, h) = (96usize, 96usize);
        let cfa = CfaPattern {
            width: 2,
            height: 2,
            colors: BAYER.to_vec(),
        };
        let wb = [1.0, 1.0, 1.0];
        let truth = |y: usize, x: usize, c: usize| {
            let (dy, dx) = (y as f32 - 48.0, x as f32 - 48.0);
            let b = (-(dy * dy + dx * dx) / (2.0 * 14.0 * 14.0)).exp();
            match c {
                0 => 0.3 + 1.5 * b,
                1 => 0.3 + 0.6 * b,
                _ => 0.2 + 0.3 * b,
            }
        };
        let mut data = vec![0f32; w * h];
        let mut ideal = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let c = BAYER[(y % 2) * 2 + x % 2] as usize;
                ideal[y * w + x] = truth(y, x, c);
                data[y * w + x] = ideal[y * w + x].min(1.0);
            }
        }
        let before = data.clone();
        let stats = reconstruct_dilated(&mut data, w, h, &cfa, wb, None, 1);
        assert_eq!(stats.areas, 1);
        assert!(stats.raised > 100, "{stats:?}");
        let (mut err_clip, mut err_fixed, mut n) = (0f64, 0f64, 0f64);
        for i in 0..w * h {
            if before[i] >= CLIPPED {
                err_clip += ((before[i] - ideal[i]) as f64).powi(2);
                err_fixed += ((data[i] - ideal[i]) as f64).powi(2);
                n += 1.0;
            } else {
                assert_eq!(data[i], before[i], "unclipped sample {i} moved");
            }
        }
        let (rms_clip, rms_fixed) = ((err_clip / n).sqrt(), (err_fixed / n).sqrt());
        println!("clipped red rms error {rms_clip:.3} -> {rms_fixed:.3}");
        assert!(rms_fixed < rms_clip * 0.65, "{rms_clip} -> {rms_fixed}");
    }

    /// A warm blob whose red is displaced by 3 px against green (a constant
    /// shift, as a decentred lens gives): reading red where it saw the same
    /// scene point as the sample being rebuilt gives a smaller error than
    /// reading it in place.
    #[test]
    fn follows_the_ca_field() {
        let (w, h) = (96usize, 96usize);
        let cfa = CfaPattern {
            width: 2,
            height: 2,
            colors: BAYER.to_vec(),
        };
        let wb = [1.0; 3];
        let shift = 3.0f32;
        let blob = |y: f32, x: f32| (-((y - 48.0).powi(2) + (x - 48.0).powi(2)) / (2.0 * 7.0 * 7.0)).exp();
        // red's sample at (y, x) saw the scene at (y - shift, x)
        // one scene, red twice green and blue half of it; only red clips
        let scene = |y: f32, x: f32| 0.1 + 0.8 * blob(y, x);
        let truth = |y: usize, x: usize, c: usize| match c {
            0 => 2.0 * scene(y as f32 - shift, x as f32),
            1 => scene(y as f32, x as f32),
            _ => 0.5 * scene(y as f32, x as f32),
        };
        let ca = Ca {
            centre: [47.5, 47.5],
            inv_radius: 1.0 / 68.0,
            red: [0.0; 3],
            blue: [0.0; 3],
            reach: [1.0; 2],
            shift: [[shift, 0.0], [0.0, 0.0]],
            clip: f32::INFINITY,
            guide: 1.0,
            bound: -1.0,
        };
        let error = |ca: Option<&Ca>| {
            let mut data = vec![0f32; w * h];
            let mut ideal = vec![0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let c = BAYER[(y % 2) * 2 + x % 2] as usize;
                    ideal[y * w + x] = truth(y, x, c);
                    data[y * w + x] = ideal[y * w + x].min(1.0);
                }
            }
            let before = data.clone();
            reconstruct_dilated(&mut data, w, h, &cfa, wb, ca, 1);
            let (mut err, mut n) = (0f64, 0f64);
            for i in 0..w * h {
                if before[i] >= CLIPPED {
                    err += ((data[i] - ideal[i]) as f64).powi(2);
                    n += 1.0;
                }
            }
            (err / n).sqrt()
        };
        let (plain, aware) = (error(None), error(Some(&ca)));
        println!("clipped red rms error in place {plain:.3}, along the field {aware:.3}");
        assert!(aware < plain * 0.7, "{plain} -> {aware}");
    }

    /// With every channel clipped the sample takes the surroundings' colour
    /// (not a grey / white patch) and stays within the headroom.
    #[test]
    fn all_clipped_keeps_the_hue() {
        let (w, h) = (64usize, 64usize);
        let cfa = CfaPattern {
            width: 2,
            height: 2,
            colors: BAYER.to_vec(),
        };
        let mut data = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let c = BAYER[(y % 2) * 2 + x % 2] as usize;
                let inside = (20..44).contains(&y) && (20..44).contains(&x);
                // warm surroundings: red twice green, blue half
                data[y * w + x] = if inside { 1.0 } else { [0.8, 0.4, 0.2][c] };
            }
        }
        reconstruct_dilated(&mut data, w, h, &cfa, [1.0; 3], None, 1);
        let at = |c: usize| {
            let (mut s, mut n) = (0.0, 0.0);
            for y in 28..36 {
                for x in 28..36 {
                    if BAYER[(y % 2) * 2 + x % 2] as usize == c {
                        s += data[y * w + x];
                        n += 1.0;
                    }
                }
            }
            s / n
        };
        let (r, g, b) = (at(0), at(1), at(2));
        assert!(r >= CLIPPED && r <= HEADROOM, "{r}");
        assert!(
            (r / g - 2.0).abs() < 0.1 && (g / b - 2.0).abs() < 0.1,
            "{r} {g} {b}"
        );
    }
}
