//! Automatic tone: fit exposure, contrast and midpoint so that the developed
//! image looks like the JPEG the camera embedded in the raw file. Saturation,
//! the two knees and vibrance are not fitted: every file gets their defaults.
//!
//! Only distributions are compared (quantiles of the display luma), so the
//! JPEG need not be aligned with the image: its crop, size and lens
//! correction do not matter. The developed samples are kept in linear sRGB
//! and run through the same steps as `develop.wgsl` (exposure, the
//! tone curve, saturation, the display encoding), on the CPU, so
//! trying a candidate costs no new develop. What a global curve cannot do,
//! the camera's local tone mapping, is not matched.
//!
//! 1. A free fit of the three to the quantiles of the display luma. The
//!    exposure found is the most the result may have (and at least 0 EV).
//! 2. The exposure is then the highest up to that where at most
//!    [`CLIP_SHARE`] of the pixels have a channel at the display's white
//!    ([`CLIPPED`]), so the histogram reaches white without the highlights
//!    being spared, and contrast and midpoint are refitted to the luma with
//!    it fixed (the curve is penalised for clipping too). Twice, as the
//!    clipping depends on the curve.

use half::f16;
use image::DynamicImage;

use crate::preview::Adjustments;
use crate::processing::Image;
use crate::tone::{self, Curve};

/// Developed pixels the fit looks at.
const SAMPLES: usize = 4096;
/// Developed pixels the clipped share is counted in: 0.5% of 4096 is 20
/// pixels, too few to go by; of this, 82.
const CLIP_SAMPLES: usize = 16384;
/// Reference pixels the fit looks at, at most.
const REFERENCE_SAMPLES: usize = 16384;
/// Fewest reference pixels worth fitting to.
const MIN_REFERENCE: usize = 256;
/// Quantiles compared: 2%, 4% … 98%.
const QUANTILES: usize = 49;
/// A display value from here up counts as clipped: reference quantiles there
/// are not compared (the camera's white says nothing about the raw's), the
/// developed image just may not clip more than the reference does.
const CLIPPED: f32 = 0.985;
/// The most pixels, as a share, that may have a channel at [`CLIPPED`].
const CLIP_SHARE: f32 = 0.005;
/// Weight of the excess clipped share in the loss.
const CLIP_WEIGHT: f32 = 0.5;
/// Weight of the share clipped over [`CLIP_SHARE`] in the second fit.
const LIMIT_WEIGHT: f32 = 4.0;
/// Weight of the pull towards the defaults, per unit of each parameter's scale.
const PRIOR: f32 = 1e-5;

const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// What is fitted: exposure in stops, contrast, midpoint display-encoded
/// (as the slider has it).
type Params = [f32; 3];
const LOWER: Params = [-3.0, 1.0, 0.1];
const UPPER: Params = [3.0, 3.0, 0.8];
/// First simplex step, and the scale of the pull towards the defaults.
const SCALE: Params = [0.5, 0.2, 0.05];

/// The fitted values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fit {
    pub exposure: f32,
    pub contrast: f32,
    /// Linear, as `Adjustments::midpoint`.
    pub midpoint: f32,
    /// Root mean square difference of the compared luma quantiles, in
    /// display units (0 … 1), with the fit and with the defaults.
    pub rms: f32,
    pub rms_default: f32,
    /// Share of pixels with a channel at the display's white.
    pub clipped: f32,
}

impl Fit {
    /// `adjustments` with the fitted tone and everything else the auto tone
    /// is about at its default; the rest is kept.
    pub fn apply(&self, adjustments: &mut Adjustments) {
        reset(adjustments);
        adjustments.exposure = self.exposure;
        adjustments.contrast = self.contrast;
        adjustments.midpoint = self.midpoint;
    }
}

/// The tone a file gets when nothing can be fitted: the defaults of the tone,
/// the knees, saturation and vibrance.
pub fn reset(adjustments: &mut Adjustments) {
    let d = Adjustments::default();
    adjustments.exposure = d.exposure;
    adjustments.contrast = d.contrast;
    adjustments.midpoint = d.midpoint;
    adjustments.toe_power = d.toe_power;
    adjustments.shoulder_power = d.shoulder_power;
    adjustments.saturation = d.saturation;
    adjustments.vibrance = d.vibrance;
}

/// The fit of `image` to the JPEG embedded in the file, `None` if it has
/// none (or too little of it to go by).
pub fn fit_embedded(image: &Image, preview: &DynamicImage) -> Option<Fit> {
    let reference = reference(preview);
    fit(
        &samples(image, SAMPLES),
        &samples(image, CLIP_SAMPLES),
        &reference,
    )
}

/// The preview's pixels (display-encoded, 0 … 1), without the black
/// bars some cameras put around a thumbnail, at most [`REFERENCE_SAMPLES`].
fn reference(preview: &DynamicImage) -> Vec<[f32; 3]> {
    let rgb = preview.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let pixel = |x: usize, y: usize| {
        rgb.get_pixel(x as u32, y as u32)
            .0
            .map(|v| v as f32 / 255.0)
    };
    let luma = |x: usize, y: usize| luma_of(pixel(x, y));
    fn black(mut line: impl Iterator<Item = f32>) -> bool {
        line.all(|l| l < 3.0 / 255.0)
    }
    let (mut top, mut bottom, mut left, mut right) = (0, height, 0, width);
    while top + 1 < bottom && black((0..width).map(|x| luma(x, top))) {
        top += 1;
    }
    while bottom - 1 > top && black((0..width).map(|x| luma(x, bottom - 1))) {
        bottom -= 1;
    }
    while left + 1 < right && black((top..bottom).map(|y| luma(left, y))) {
        left += 1;
    }
    while right - 1 > left && black((top..bottom).map(|y| luma(right - 1, y))) {
        right -= 1;
    }
    let total = (right - left) * (bottom - top);
    let stride = total.div_ceil(REFERENCE_SAMPLES).max(1);
    (0..total)
        .step_by(stride)
        .map(|i| pixel(left + i % (right - left), top + i / (right - left)))
        .collect()
}

fn luma_of(c: [f32; 3]) -> f32 {
    LUMA[0] * c[0] + LUMA[1] * c[1] + LUMA[2] * c[2]
}

/// An even sample of `count` of `image`'s pixels as linear sRGB, with no adjustments.
fn samples(image: &Image, count: usize) -> Vec<[f32; 3]> {
    let m = Adjustments::default().cam_to_srgb(image.cam_to_srgb, image.white);
    let pixels = image.pixels.len() / 4;
    let stride = (pixels / count).max(1);
    (0..pixels)
        .step_by(stride)
        .map(|i| {
            let p = &image.pixels[i * 4..i * 4 + 3];
            let c = [p[0], p[1], p[2]].map(|bits| f16::from_bits(bits).to_f32());
            m.map(|row| row[0] * c[0] + row[1] * c[1] + row[2] * c[2])
        })
        .collect()
}

/// What the developed image is fitted to.
struct Target {
    /// Quantiles of the luma.
    luma: Vec<f32>,
    /// The share of the reference's luma at white.
    clipped: f32,
}

impl Target {
    fn new(reference: &[[f32; 3]]) -> Self {
        let mut luma: Vec<f32> = reference.iter().map(|&c| luma_of(c)).collect();
        luma.sort_unstable_by(f32::total_cmp);
        Self {
            clipped: clipped(&luma),
            luma: quantiles(&luma),
        }
    }
}

/// Fits the parameters to `reference` (display-encoded pixels) by
/// Nelder-Mead from the defaults. `rgb` and `clip_rgb`: linear sRGB samples
/// of the developed image, a smaller and a larger set.
pub fn fit(rgb: &[[f32; 3]], clip_rgb: &[[f32; 3]], reference: &[[f32; 3]]) -> Option<Fit> {
    if rgb.is_empty() || clip_rgb.is_empty() || reference.len() < MIN_REFERENCE {
        return None;
    }
    let target = Target::new(reference);
    let rgb: Vec<[f32; 3]> = rgb.iter().map(|&c| encoded(c)).collect();
    let clip_rgb: Vec<[f32; 3]> = clip_rgb.iter().map(|&c| encoded(c)).collect();
    let (rgb, clip_rgb) = (&rgb[..], &clip_rgb[..]);
    let d = Adjustments::default();
    let start: Params = [d.exposure, d.contrast, tone::encode(d.midpoint)];
    let mut scratch = Vec::with_capacity(rgb.len());

    // 1. The free fit: the exposure it finds is the cap. From a few starts,
    // as the loss has local minima (a flat curve with more exposure, against
    // a steep one with less).
    let mut free: Option<(f32, Params)> = None;
    for from in [
        start,
        [0.5, 2.5, tone::encode(0.02)],
        [0.0, 2.0, tone::encode(0.06)],
    ] {
        let mut objective = |x: &Params| loss(x, false, rgb, &target, &start, &mut scratch);
        let first = nelder_mead(&mut objective, from, SCALE);
        let found = bounded(&nelder_mead(&mut objective, first, SCALE));
        let at = objective(&found);
        if free.is_none_or(|(best, _)| at < best) {
            free = Some((at, found));
        }
    }
    let mut x = free.map_or(start, |(_, x)| x);
    let cap = x[0].max(0.0);

    // 2. The exposure the clipping allows, then the rest to go with it.
    for _ in 0..2 {
        x[0] = clip_limited(cap, &x, clip_rgb);
        let pinned = x[0];
        let mut objective = |y: &[f32; 2]| {
            let x = [pinned, y[0], y[1]];
            loss(&x, true, rgb, &target, &start, &mut scratch)
        };
        let first = nelder_mead(&mut objective, [x[1], x[2]], [SCALE[1], SCALE[2]]);
        x = bounded(&[pinned, first[0], first[1]]);
    }
    x[0] = clip_limited(cap, &x, clip_rgb);

    let clipped = clip_share(&curve_of(&x), x[0], clip_rgb);
    Some(Fit {
        exposure: x[0],
        contrast: x[1],
        midpoint: tone::decode(x[2]),
        rms: data_loss(&x, false, rgb, &target, &mut scratch).sqrt(),
        rms_default: data_loss(&start, false, rgb, &target, &mut scratch).sqrt(),
        clipped,
    })
}

fn bounded(x: &Params) -> Params {
    std::array::from_fn(|i| x[i].clamp(LOWER[i], UPPER[i]))
}

/// The tone curve for `x`, with the default knees.
fn curve_of(x: &Params) -> Curve {
    Curve::new(&Adjustments {
        contrast: x[1],
        midpoint: tone::decode(x[2]),
        ..Adjustments::default()
    })
}

/// The quantiles compared, of sorted values.
fn quantiles(sorted: &[f32]) -> Vec<f32> {
    (1..=QUANTILES)
        .map(|k| sorted[(k * sorted.len() / (QUANTILES + 1)).min(sorted.len() - 1)])
        .collect()
}

/// The share of sorted values that count as clipped.
fn clipped(sorted: &[f32]) -> f32 {
    let below = sorted.partition_point(|&v| v < CLIPPED);
    (sorted.len() - below) as f32 / sorted.len() as f32
}

/// A linear sRGB sample as `display` takes it: the curve's input encoding
/// (`tone::encode`) before its clamp at white. Encoding is a power, so
/// exposure afterwards is one multiply (see [`gain`]) and the encoding is
/// done once per sample instead of once per candidate.
fn encoded(rgb: [f32; 3]) -> [f32; 3] {
    rgb.map(|v| v.max(0.0).powf(1.0 / 2.4))
}

/// What exposure `exposure` stops multiplies an [`encoded`] sample by.
fn gain(exposure: f32) -> f32 {
    (exposure / 2.4).exp2()
}

/// Display value (0 … 1, sRGB-encoded) of an [`encoded`] sample, as
/// `develop.wgsl` makes it with the default saturation (no grading);
/// `gain` is [`gain`] of the exposure.
fn display(rgb: [f32; 3], curve: &Curve, gain: f32) -> [f32; 3] {
    let c = rgb.map(|v| curve.apply((v * gain).min(1.0)));
    let luma = luma_of(c);
    let saturation = Adjustments::default().saturation;
    c.map(|v| srgb_of_encoded((luma + (v - luma) * saturation).max(0.0).min(1.0)))
}

/// The sRGB encoding of `v^2.4`: the power and the encoding's own
/// cancel above its linear toe, which saves two `powf` calls a channel.
fn srgb_of_encoded(v: f32) -> f32 {
    /// 0.0031308^(1 / 2.4), where the encoding's toe ends.
    const TOE: f32 = 0.089_46;
    if v <= TOE {
        12.92 * v.powf(2.4)
    } else {
        1.055 * v - 0.055
    }
}

/// The share of `rgb` with a channel at the display's white.
/// Counted on all the cores, in chunks.
fn clip_share(curve: &Curve, exposure: f32, rgb: &[[f32; 3]]) -> f32 {
    let gain = gain(exposure);
    let count = |chunk: &[[f32; 3]]| {
        chunk
            .iter()
            .filter(|&&c| display(c, curve, gain).iter().any(|&v| v >= CLIPPED))
            .count()
    };
    let threads = std::thread::available_parallelism()
        .map_or(1, |t| t.get())
        .min(8);
    let size = rgb.len().div_ceil(threads).max(1);
    let total: usize = std::thread::scope(|scope| {
        let handles: Vec<_> = rgb
            .chunks(size)
            .map(|chunk| scope.spawn(move || count(chunk)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or(0)).sum()
    });
    total as f32 / rgb.len() as f32
}

/// The highest exposure in 0 ..= `cap` (0 if `cap` is below it) at which at
/// most [`CLIP_SHARE`] of `rgb` clips under `x`'s curve.
fn clip_limited(cap: f32, x: &Params, rgb: &[[f32; 3]]) -> f32 {
    let curve = curve_of(x);
    let share = |exposure: f32| clip_share(&curve, exposure, rgb);
    if share(cap) <= CLIP_SHARE {
        return cap;
    }
    if share(0.0) > CLIP_SHARE {
        return 0.0;
    }
    // The share rises with exposure.
    let (mut lo, mut hi) = (0.0, cap);
    for _ in 0..12 {
        let mid = 0.5 * (lo + hi);
        if share(mid) <= CLIP_SHARE {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Mean squared luma quantile difference plus the excess-clipping term.
/// With `limit`, a share of pixels with a channel at white over
/// [`CLIP_SHARE`] costs as well (the curve may not clip either).
fn data_loss(
    x: &Params,
    limit: bool,
    rgb: &[[f32; 3]],
    target: &Target,
    scratch: &mut Vec<f32>,
) -> f32 {
    let curve = curve_of(x);
    let gain = gain(x[0]);
    scratch.clear();
    let mut any = 0;
    for &c in rgb {
        let shown = display(c, &curve, gain);
        any += shown.iter().any(|&v| v >= CLIPPED) as usize;
        scratch.push(luma_of(shown));
    }
    scratch.sort_unstable_by(f32::total_cmp);
    let excess = (clipped(scratch) - target.clipped).max(0.0) * CLIP_WEIGHT;
    let over = if limit {
        (any as f32 / rgb.len() as f32 - CLIP_SHARE).max(0.0) * LIMIT_WEIGHT
    } else {
        0.0
    };
    let (mut sum, mut n) = (0.0, 0);
    for (d, &t) in quantiles(scratch).into_iter().zip(&target.luma) {
        if t < CLIPPED {
            sum += (d - t) * (d - t);
            n += 1;
        }
    }
    sum / n.max(1) as f32 + excess * excess + over * over
}

/// `data_loss` at `x` held inside its bounds, a penalty for how far it was
/// outside, and the pull towards `start`.
fn loss(
    x: &Params,
    limit: bool,
    rgb: &[[f32; 3]],
    target: &Target,
    start: &Params,
    scratch: &mut Vec<f32>,
) -> f32 {
    let inside = bounded(x);
    let mut penalty = 0.0;
    for i in 0..x.len() {
        let outside = (x[i] - inside[i]) / SCALE[i];
        let pull = (inside[i] - start[i]) / SCALE[i];
        penalty += 1e-2 * outside * outside + PRIOR * pull * pull;
    }
    data_loss(&inside, limit, rgb, target, scratch) + penalty
}

/// Nelder-Mead minimisation of `f` from `start`, first steps `step`.
fn nelder_mead<const N: usize>(
    f: &mut impl FnMut(&[f32; N]) -> f32,
    start: [f32; N],
    step: [f32; N],
) -> [f32; N] {
    let mut simplex: Vec<([f32; N], f32)> = Vec::with_capacity(N + 1);
    simplex.push((start, f(&start)));
    for i in 0..N {
        let mut x = start;
        x[i] += step[i];
        simplex.push((x, f(&x)));
    }
    for _ in 0..200 {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        if simplex[N].1 - simplex[0].1 < 1e-8 {
            break;
        }
        let mut centroid = [0.0; N];
        for (x, _) in &simplex[..N] {
            for i in 0..N {
                centroid[i] += x[i] / N as f32;
            }
        }
        let towards = |t: f32, from: &[f32; N]| -> [f32; N] {
            std::array::from_fn(|i| centroid[i] + t * (from[i] - centroid[i]))
        };
        let worst = simplex[N];
        let reflected = towards(-1.0, &worst.0);
        let fr = f(&reflected);
        if fr < simplex[0].1 {
            let expanded = towards(-2.0, &worst.0);
            let fe = f(&expanded);
            simplex[N] = if fe < fr {
                (expanded, fe)
            } else {
                (reflected, fr)
            };
        } else if fr < simplex[N - 1].1 {
            simplex[N] = (reflected, fr);
        } else {
            let contracted = if fr < worst.1 {
                towards(-0.5, &worst.0)
            } else {
                towards(0.5, &worst.0)
            };
            let fc = f(&contracted);
            if fc < fr.min(worst.1) {
                simplex[N] = (contracted, fc);
            } else {
                let best = simplex[0].0;
                for (x, v) in simplex.iter_mut().skip(1) {
                    *x = std::array::from_fn(|i| best[i] + 0.5 * (x[i] - best[i]));
                    *v = f(x);
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    simplex[0].0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scene of linear values, deterministic, topping out at 0.6.
    fn scene() -> Vec<[f32; 3]> {
        (0..SAMPLES)
            .map(|i| {
                let t = (i as f32 + 0.5) / SAMPLES as f32;
                let v = 0.002 + 0.6 * t * t * t;
                [v, v * (0.6 + 0.4 * t), v * (0.9 - 0.4 * t)]
            })
            .collect()
    }

    /// The display values of `rgb` at a tone.
    fn reference_of(
        rgb: &[[f32; 3]],
        exposure: f32,
        contrast: f32,
        midpoint_encoded: f32,
    ) -> Vec<[f32; 3]> {
        let curve = curve_of(&[exposure, contrast, midpoint_encoded]);
        rgb.iter()
            .map(|&c| display(encoded(c), &curve, gain(exposure)))
            .collect()
    }

    fn fit_scene(reference: &[[f32; 3]]) -> Fit {
        let rgb = scene();
        fit(&rgb, &rgb, reference).unwrap()
    }

    #[test]
    fn matches_a_known_tone() {
        let fit = fit_scene(&reference_of(&scene(), 0.3, 1.8, 0.4));
        assert!(fit.rms < 0.01, "{fit:?}");
        assert!(fit.rms < fit.rms_default / 3.0, "{fit:?}");
        assert!(
            (fit.contrast - 1.8).abs() < 0.3 && (fit.exposure - 0.3).abs() < 0.3,
            "{fit:?}"
        );
    }

    #[test]
    fn defaults_for_the_default_tone() {
        let d = Adjustments::default();
        let fit = fit_scene(&reference_of(
            &scene(),
            d.exposure,
            d.contrast,
            tone::encode(d.midpoint),
        ));
        assert!(fit.rms < 0.005, "{fit:?}");
        assert!(fit.exposure.abs() < 0.3, "{fit:?}");
    }

    #[test]
    fn exposure_stops_at_the_preview_and_never_goes_below_zero() {
        let rgb = scene();
        // A preview exposed 0.3 EV: no more, though nothing clips.
        let fit = fit_scene(&reference_of(&rgb, 0.3, 1.4, 0.43));
        assert!(fit.exposure <= 0.4 && fit.exposure >= 0.0, "{fit:?}");
        // A darker one: not below 0.
        let fit = fit_scene(&reference_of(&rgb, -1.0, 1.4, 0.43));
        assert!(fit.exposure >= 0.0, "{fit:?}");
    }

    #[test]
    fn exposure_is_held_by_the_clipping() {
        // A preview exposed 2.5 EV, far past what clips; the exposure stops
        // where 0.5% of the pixels clip.
        let rgb = scene();
        let fit = fit_scene(&reference_of(&rgb, 2.5, 1.4, 0.43));
        assert!(fit.clipped <= CLIP_SHARE + 1e-3, "{fit:?}");
        assert!(fit.exposure < 2.5 && fit.exposure > 0.0, "{fit:?}");
        // and no lower than it needs: a bit more would clip more
        let curve = curve_of(&[fit.exposure, fit.contrast, tone::encode(fit.midpoint)]);
        let encoded_rgb: Vec<_> = rgb.iter().map(|&c| encoded(c)).collect();
        assert!(clip_share(&curve, fit.exposure + 0.1, &encoded_rgb) > CLIP_SHARE);
    }

    #[test]
    fn too_little_reference_gives_none() {
        let rgb = scene();
        assert!(fit(&rgb, &rgb, &[[0.5; 3]; 10]).is_none());
    }

    #[test]
    fn applying_a_fit_leaves_saturation_and_the_knees_at_their_defaults() {
        let d = Adjustments::default();
        let mut a = Adjustments {
            saturation: 2.5,
            toe_power: 4.0,
            shoulder_power: 0.7,
            vibrance: 1.0,
            ..d
        };
        Fit {
            exposure: 0.5,
            contrast: 2.0,
            midpoint: 0.05,
            rms: 0.0,
            rms_default: 0.0,
            clipped: 0.0,
        }
        .apply(&mut a);
        assert_eq!((a.exposure, a.contrast, a.midpoint), (0.5, 2.0, 0.05));
        assert_eq!(
            (a.saturation, a.toe_power, a.shoulder_power, a.vibrance),
            (d.saturation, d.toe_power, d.shoulder_power, d.vibrance)
        );
    }

    /// `STARFISH_TEST_RAWS=a.raf,b.nef cargo test --release auto::tests::files -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn files() {
        for path in std::env::var("STARFISH_TEST_RAWS").unwrap().split(',') {
            let path = std::path::Path::new(path);
            let raw = crate::raw::decode(path).unwrap();
            let image = crate::processing::Engine::get()
                .unwrap()
                .develop_preview(&raw, crate::lens::Corrections::NONE)
                .unwrap();
            let start = std::time::Instant::now();
            let jpeg = crate::library::embedded_preview(path);
            let decoded = start.elapsed().as_secs_f32() * 1e3;
            let fit = jpeg.as_ref().and_then(|j| fit_embedded(&image, j));
            println!(
                "{}: jpeg {:?}, decode {decoded:.0} ms, fit {:.0} ms\n  {fit:?}",
                path.display(),
                jpeg.map(|j| (j.width(), j.height())),
                start.elapsed().as_secs_f32() * 1e3 - decoded
            );
        }
    }
}
