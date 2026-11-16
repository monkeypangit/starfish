//! Lateral chromatic aberration measured in the developed image, for every
//! camera: the makers' CA tables did not match what their own cameras correct
//! (Fuji's blue 0.4-0.75x of its table, Sony's unit unknown, Canon and
//! Panasonic have none). It is measured in the demosaic stage's output, not
//! the mosaic, because the networks already pull red and blue part of the
//! way onto green (the X-T30's X-Trans output kept 36-71% of the CA measured
//! in its mosaic, the A6700's Bayer one ~73%): correcting the mosaic's CA
//! over-corrected once the network had run. The mosaic's ([`measure_raw`],
//! `RawFrame::mosaic_ca`) is what `CaFix` takes out before the network.
//!
//! The planes are 2x2 averages of the image (in the mosaic: each colour
//! rebuilt from its own CFA sites by a normalised Gaussian, so every sample
//! counts at its own position). In blocks of 64 px the shift of red and of
//! blue against green is found: the whole-pixel shift of the best
//! correlation, then Gauss-Newton on a Catmull-Rom warp, with a gain and
//! offset per block. Its weight is the structure tensor over the residual,
//! so an edge fixes only the shift across it. A robust fit over all blocks
//! gives `d = t + s(rho) (p - c)`, `s = k0 + k1 rho^2 + k2 rho^4`: the
//! channel at `c + (p - c)(1 + s)` is green's at `p`, the radial scale offset
//! the lens tables give (`lens.wgsl`). The constant shift `t` (0.1-0.3 px,
//! cause unknown) is fitted so it cannot bias `s`, but not corrected.
//!
//! Checked on synthetic Bayer and X-Trans mosaics with a known CA
//! (`measures_known_ca`; also 1.5 px red and 3.3 px blue added to a DIV2K
//! photo): right to 0.05-0.1 px, with no bias in scale. Five shots of one
//! scene from ISO 160 to 25600 agree to 0.01 px. The A6700's JPEGs, which
//! the camera corrects for CA, show none of the 0.2-0.8 px measured in their
//! raws left (< 0.05 px mid-frame).

use demosaic_run::noise::par_map;

use crate::processing::f16_to_f32;
use crate::raw::{CfaPattern, RawFrame};

/// Gaussian of the mosaic's planes, in sensor pixels ...
const SIGMA: f32 = 1.5;
/// ... its ±5 px combined with the 2 px pooling.
const TAPS: usize = 12;
/// Block side, in half-resolution pixels.
const BLOCK: usize = 32;
/// The whole-pixel search reaches this far (half-resolution pixels) ...
const SEARCH: i32 = 3;
/// ... and Gauss-Newton two more; blocks keep this far from the edge.
const MARGIN: usize = SEARCH as usize + 5;
const ITERATIONS: usize = 6;
/// Blocks that correlate less, colour edges mostly, are left out.
const MIN_NCC: f32 = 0.8;
/// A sample this close to white makes its block clipped.
const CLIP: f32 = crate::processing::CLIPPED;
/// A channel needs this many blocks, this many of them beyond `OUTER` ...
const MIN_BLOCKS: usize = 200;
const MIN_OUTER: usize = 30;
const OUTER: f32 = 0.6;
/// ... and a shift no larger than this anywhere (sensor pixels).
const MAX_SHIFT: f32 = 8.0;
/// Huber threshold of the fit, in standard deviations.
const HUBER: f64 = 2.0;

/// The lateral CA of red and blue against green.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    /// Half-diagonal of the frame in pixels: the unit of rho.
    pub radius: f32,
    /// `[k0, k1, k2]` of red and blue; zero for a channel not measured.
    pub coefficients: [[f32; 3]; 2],
    /// The largest rho with blocks, per channel; `s` is held beyond.
    pub reach: [f32; 2],
    /// Blocks per channel.
    pub blocks: [usize; 2],
    /// Constant shift `t` (y, x) in pixels, per channel.
    pub shift: [[f32; 2]; 2],
}

impl Measured {
    /// The radial scale offset of channel 0 (red) or 1 (blue) at `rho`.
    pub fn scale(&self, channel: usize, rho: f32) -> f32 {
        let [k0, k1, k2] = self.coefficients[channel];
        let r2 = rho.min(self.reach[channel]).powi(2);
        k0 + r2 * (k1 + r2 * k2)
    }

    /// The shift along the radius at `rho`, in pixels.
    pub fn pixels(&self, channel: usize, rho: f32) -> f32 {
        self.scale(channel, rho) * rho * self.radius
    }

    /// For the metadata panel: the shifts at mid-frame and in the corner.
    pub fn summary(&self) -> String {
        let at = |c| format!("{:+.2} / {:+.2}", self.pixels(c, 0.5), self.pixels(c, 1.0));
        format!("R {}, B {} px", at(0), at(1))
    }
}

/// The lateral CA in a developed image of `width` × `height`, `Rgba16Float`
/// `pixels` (`Image::pixels`), if enough of it has detail to measure it.
pub fn measure(width: u32, height: u32, pixels: &[u16]) -> Option<Measured> {
    measure_planes(&Planes::from_image(width as usize, height as usize, pixels))
}

/// The lateral CA in `raw`'s mosaic, before any demosaic.
pub fn measure_raw(raw: &RawFrame) -> Option<Measured> {
    measure_mosaic(&Mosaic {
        data: &raw.data,
        width: raw.width,
        height: raw.height,
        cfa: &raw.cfa,
        wb: raw.wb,
    })
}

/// What [`measure_raw`] needs of a [`RawFrame`].
struct Mosaic<'a> {
    data: &'a [f32],
    width: usize,
    height: usize,
    cfa: &'a CfaPattern,
    wb: [f32; 3],
}

fn measure_mosaic(raw: &Mosaic) -> Option<Measured> {
    measure_planes(&Planes::new(raw))
}

fn measure_planes(planes: &Planes) -> Option<Measured> {
    let start = std::time::Instant::now();
    let blocks = planes.blocks();
    let radius = (planes.sensor.0 as f32).hypot(planes.sensor.1 as f32) / 2.0;
    let mut measured = Measured {
        radius,
        coefficients: [[0.0; 3]; 2],
        reach: [0.0; 2],
        blocks: [0; 2],
        shift: [[0.0; 2]; 2],
    };
    for channel in 0..2 {
        let blocks: Vec<&Block> = blocks.iter().filter_map(|b| b[channel].as_ref()).collect();
        let rho = |b: &Block| b.p[0].hypot(b.p[1]) / radius;
        let outer = blocks.iter().filter(|b| rho(b) > OUTER).count();
        measured.blocks[channel] = blocks.len();
        if blocks.len() < MIN_BLOCKS || outer < MIN_OUTER {
            continue;
        }
        let theta = fit(&blocks, radius);
        measured.shift[channel] = [theta[0] as f32, theta[1] as f32];
        measured.coefficients[channel] = [theta[2] as f32, theta[3] as f32, theta[4] as f32];
        measured.reach[channel] = blocks.iter().map(|b| rho(b)).fold(0.0, f32::max);
        let largest = (0..=20)
            .map(|i| measured.pixels(channel, i as f32 / 20.0).abs())
            .fold(0.0, f32::max);
        if !largest.is_finite() || largest > MAX_SHIFT {
            measured.coefficients[channel] = [0.0; 3];
        }
    }
    crate::log::write(&format!(
        "lateral CA: {} (blocks R {}, B {}; shift R {:?}, B {:?} px) in {} ms",
        measured.summary(),
        measured.blocks[0],
        measured.blocks[1],
        measured.shift[0],
        measured.shift[1],
        start.elapsed().as_millis()
    ));
    (measured.coefficients != [[0.0; 3]; 2]).then_some(measured)
}

/// The colour planes at half resolution.
struct Planes {
    width: usize,
    height: usize,
    /// The frame's size, sensor pixels.
    sensor: (usize, usize),
    /// Red, green, blue, white balanced.
    planes: [Vec<f32>; 3],
    /// Whether a raw sample under the pixel is near white.
    clipped: Vec<bool>,
}

impl Planes {
    /// 2x2 averages of a developed image.
    fn from_image(sensor_width: usize, sensor_height: usize, pixels: &[u16]) -> Self {
        let (width, height) = (sensor_width / 2, sensor_height / 2);
        let rows = par_map(height, |i| {
            let mut out: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(width));
            let mut clipped = Vec::with_capacity(width);
            for j in 0..width {
                let (mut sum, mut clip) = ([0f32; 3], false);
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let at = ((2 * i + dy) * sensor_width + 2 * j + dx) * 4;
                    for (c, s) in sum.iter_mut().enumerate() {
                        let v = f16_to_f32(pixels[at + c]);
                        *s += v;
                        clip |= v > CLIP;
                    }
                }
                for (plane, s) in out.iter_mut().zip(sum) {
                    plane.push(s / 4.0);
                }
                clipped.push(clip);
            }
            (out, clipped)
        });
        let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(width * height));
        let mut clipped = Vec::with_capacity(width * height);
        for (row, row_clipped) in rows {
            for (plane, part) in planes.iter_mut().zip(row) {
                plane.extend(part);
            }
            clipped.extend(row_clipped);
        }
        Self {
            width,
            height,
            sensor: (sensor_width, sensor_height),
            planes,
            clipped,
        }
    }

    /// Each colour of a mosaic from its own sites.
    fn new(raw: &Mosaic) -> Self {
        let (width, height) = (raw.width / 2, raw.height / 2);
        // Pooled pixel j covers sensor pixels 2j and 2j + 1; tap t is sensor
        // pixel 2j - 5 + t.
        let g = |d: f32| (-d * d / (2.0 * SIGMA * SIGMA)).exp();
        let kernel: [f32; TAPS] = std::array::from_fn(|t| g(t as f32 - 5.0) + g(t as f32 - 6.0));
        // each sensor column's colour, per row of the CFA
        let cfa = &raw.cfa;
        let colours: Vec<Vec<usize>> = (0..cfa.height)
            .map(|py| {
                (0..raw.width)
                    .map(|x| cfa.colors[py * cfa.width + x % cfa.width] as usize)
                    .collect()
            })
            .collect();

        const BAND: usize = 32;
        let bands = par_map(height.div_ceil(BAND), |band| {
            let (i0, i1) = (band * BAND, ((band + 1) * BAND).min(height));
            // Sensor rows y0.. feed output rows i0..i1.
            let y0 = 2 * i0 as isize - 5;
            let rows = 2 * (i1 - i0) + TAPS - 2;
            // Per sensor row and colour, the weighted sum and weight at each pooled column.
            let mut sum = vec![0f32; 3 * rows * width];
            let mut weight = vec![0f32; 3 * rows * width];
            for r in 0..rows {
                let y = y0 + r as isize;
                if y < 0 || y >= raw.height as isize {
                    continue;
                }
                let y = y as usize;
                let line = &raw.data[y * raw.width..(y + 1) * raw.width];
                let colour = &colours[y % cfa.height];
                for j in 0..width {
                    // the taps on the frame
                    let x0 = 2 * j as isize - 5;
                    let first = (-x0).max(0) as usize;
                    let last = TAPS.min((raw.width as isize - x0) as usize);
                    let (mut s, mut w) = ([0f32; 3], [0f32; 3]);
                    for t in first..last {
                        let x = (x0 + t as isize) as usize;
                        let c = colour[x];
                        s[c] += kernel[t] * line[x];
                        w[c] += kernel[t];
                    }
                    for c in 0..3 {
                        let at = (c * rows + r) * width + j;
                        sum[at] = s[c] * raw.wb[c];
                        weight[at] = w[c];
                    }
                }
            }
            let mut out: [Vec<f32>; 3] =
                std::array::from_fn(|_| Vec::with_capacity((i1 - i0) * width));
            let mut clipped = Vec::with_capacity((i1 - i0) * width);
            for i in i0..i1 {
                for (c, plane) in out.iter_mut().enumerate() {
                    for j in 0..width {
                        let (mut s, mut w) = (0.0, 0.0);
                        for (t, &k) in kernel.iter().enumerate() {
                            let at = (c * rows + 2 * (i - i0) + t) * width + j;
                            s += k * sum[at];
                            w += k * weight[at];
                        }
                        plane.push(if w > 0.0 { s / w } else { 0.0 });
                    }
                }
                for j in 0..width {
                    let sample = |y: usize, x: usize| raw.data[y * raw.width + x];
                    let (y, x) = (2 * i, 2 * j);
                    clipped.push(
                        [
                            sample(y, x),
                            sample(y, x + 1),
                            sample(y + 1, x),
                            sample(y + 1, x + 1),
                        ]
                        .iter()
                        .any(|&v| v > CLIP),
                    );
                }
            }
            (out, clipped)
        });

        let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(width * height));
        let mut clipped = Vec::with_capacity(width * height);
        for (band, band_clipped) in bands {
            for (plane, part) in planes.iter_mut().zip(band) {
                plane.extend(part);
            }
            clipped.extend(band_clipped);
        }
        Self {
            width,
            height,
            sensor: (raw.width, raw.height),
            planes,
            clipped,
        }
    }

    /// Every block's shifts of red and blue, where measurable.
    fn blocks(&self) -> Vec<[Option<Block>; 2]> {
        let fits = |n: usize| n.saturating_sub(2 * MARGIN) / BLOCK;
        let (rows, columns) = (fits(self.height), fits(self.width));
        par_map(rows, |row| {
            (0..columns)
                .map(|column| self.block(MARGIN + row * BLOCK, MARGIN + column * BLOCK))
                .collect::<Vec<_>>()
        })
        .into_iter()
        .flatten()
        .collect()
    }

    /// The block whose top left is (y0, x0).
    fn block(&self, y0: usize, x0: usize) -> [Option<Block>; 2] {
        const N: usize = BLOCK * BLOCK;
        // Clipping blurs into its surroundings: look 4 px further.
        let clipped = (y0 - 4..y0 + BLOCK + 4).any(|y| {
            self.clipped[y * self.width + x0 - 4..y * self.width + x0 + BLOCK + 4]
                .iter()
                .any(|&c| c)
        });
        if clipped {
            return [None, None];
        }
        let green = &self.planes[1];
        let mut gn: Vec<f32> = (0..N)
            .map(|k| green[(y0 + k / BLOCK) * self.width + x0 + k % BLOCK])
            .collect();
        if !normalise(&mut gn) {
            return [None, None];
        }
        // Block centre relative to the image centre, sensor pixels: pooled
        // pixel i is centred on sensor pixel 2i + 0.5.
        let centre = |v: usize, n: usize| {
            2.0 * (v as f32 + (BLOCK as f32 - 1.0) / 2.0) + 0.5 - (n as f32 - 1.0) / 2.0
        };
        let p = [centre(y0, self.sensor.1), centre(x0, self.sensor.0)];
        [0, 2].map(|c| {
            self.shift(&self.planes[c], &gn, y0, x0)
                .map(|(d, w)| Block {
                    p,
                    d: [2.0 * d[0], 2.0 * d[1]],
                    w: w.map(|r| r.map(|v| v / 4.0)),
                })
        })
    }

    /// Where `plane` matches the normalised green `gn` of the block at (y0,
    /// x0): the shift (half-resolution pixels) and its weight matrix.
    fn shift(
        &self,
        plane: &[f32],
        gn: &[f32],
        y0: usize,
        x0: usize,
    ) -> Option<([f32; 2], [[f32; 2]; 2])> {
        let n = gn.len() as f32;
        let w = self.width;
        // The correlation at a whole-pixel shift; gn has zero mean.
        let ncc = |dy: i32, dx: i32| {
            let (mut s1, mut s2, mut sg) = (0.0, 0.0, 0.0);
            for yy in 0..BLOCK {
                let row = ((y0 + yy) as i32 + dy) as usize * w;
                for xx in 0..BLOCK {
                    let c = plane[row + ((x0 + xx) as i32 + dx) as usize];
                    s1 += c;
                    s2 += c * c;
                    sg += c * gn[yy * BLOCK + xx];
                }
            }
            let var = s2 / n - (s1 / n).powi(2);
            if var > 0.0 { sg / n / var.sqrt() } else { -1.0 }
        };
        // Climb to the best whole-pixel shift.
        let (mut at, mut best): ((i32, i32), f32) = ((0, 0), ncc(0, 0));
        loop {
            let mut next = None;
            for (dy, dx) in [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ] {
                let (y, x) = (at.0 + dy, at.1 + dx);
                if y.abs() > SEARCH || x.abs() > SEARCH {
                    continue;
                }
                let v = ncc(y, x);
                if v > best {
                    best = v;
                    next = Some((y, x));
                }
            }
            match next {
                Some(n) => at = n,
                None => break,
            }
        }

        // Gauss-Newton from there, the gain and offset taken out by
        // normalising the warped block.
        let mut d = [at.0 as f32, at.1 as f32];
        const S: usize = BLOCK + 2;
        let mut cn = vec![0f32; BLOCK * BLOCK];
        let (mut a, mut ncc_final, mut var);
        let mut iteration = 0;
        loop {
            let warped = self.warp(plane, y0 - 1, x0 - 1, S, d);
            for k in 0..BLOCK * BLOCK {
                cn[k] = warped[(k / BLOCK + 1) * S + k % BLOCK + 1];
            }
            let (mean, sd) = moments(&cn);
            if sd <= 0.0 {
                return None;
            }
            a = [[0.0f32; 2]; 2];
            let mut b = [0.0f32; 2];
            let (mut r2, mut cg) = (0.0, 0.0);
            for yy in 0..BLOCK {
                for xx in 0..BLOCK {
                    let k = yy * BLOCK + xx;
                    let c = (cn[k] - mean) / sd;
                    let at = (yy + 1) * S + xx + 1;
                    let jy = (warped[at + S] - warped[at - S]) / (2.0 * sd);
                    let jx = (warped[at + 1] - warped[at - 1]) / (2.0 * sd);
                    let r = c - gn[k];
                    a[0][0] += jy * jy;
                    a[0][1] += jy * jx;
                    a[1][1] += jx * jx;
                    b[0] -= jy * r;
                    b[1] -= jx * r;
                    r2 += r * r;
                    cg += c * gn[k];
                }
            }
            a[1][0] = a[0][1];
            ncc_final = cg / n;
            var = r2 / n + 1e-6;
            iteration += 1;
            if iteration > ITERATIONS {
                break;
            }
            // Levenberg damping: an edge leaves the shift along it alone.
            let lambda = 1e-3 * (a[0][0] + a[1][1]);
            let (p, q, r) = (a[0][0] + lambda, a[0][1], a[1][1] + lambda);
            let det = p * r - q * q;
            if det.is_nan() || det <= 0.0 {
                return None;
            }
            let step = [(r * b[0] - q * b[1]) / det, (p * b[1] - q * b[0]) / det];
            d[0] += step[0].clamp(-1.0, 1.0);
            d[1] += step[1].clamp(-1.0, 1.0);
            if d[0].abs() > (SEARCH + 2) as f32 || d[1].abs() > (SEARCH + 2) as f32 {
                return None;
            }
            if step[0].abs().max(step[1].abs()) < 1e-3 {
                break;
            }
        }
        (ncc_final >= MIN_NCC).then(|| (d, a.map(|row| row.map(|v| v / var))))
    }

    /// `plane` at (y0 + i + d0, x0 + j + d1) for i, j < size: Catmull-Rom,
    /// separable as the shift is the same everywhere. It has to be
    /// Catmull-Rom: its slope at the samples is the central difference, so
    /// a small shift moves the block as far as it says. Keys' cubic with
    /// a = -0.75 (PyTorch's bicubic) moves it 1.5x as far, and measured real
    /// CA 1.4x too small; Lanczos agrees with Catmull-Rom to 3%.
    fn warp(&self, plane: &[f32], y0: usize, x0: usize, size: usize, d: [f32; 2]) -> Vec<f32> {
        let weights = |f: f32| {
            [
                f * (-0.5 + f * (1.0 - 0.5 * f)),
                1.0 + f * f * (-2.5 + 1.5 * f),
                f * (0.5 + f * (2.0 - 1.5 * f)),
                f * f * (-0.5 + 0.5 * f),
            ]
        };
        let (iy, ix) = (d[0].floor(), d[1].floor());
        let (wy, wx) = (weights(d[0] - iy), weights(d[1] - ix));
        // Sensor-plane origin of tap 0 (one before the floor).
        let top = (y0 as isize + iy as isize - 1) as usize;
        let left = (x0 as isize + ix as isize - 1) as usize;
        let mut rows = vec![0f32; (size + 3) * size];
        for r in 0..size + 3 {
            let line = &plane[(top + r) * self.width + left..];
            for j in 0..size {
                rows[r * size + j] = wx[0] * line[j]
                    + wx[1] * line[j + 1]
                    + wx[2] * line[j + 2]
                    + wx[3] * line[j + 3];
            }
        }
        let mut out = vec![0f32; size * size];
        for i in 0..size {
            for j in 0..size {
                out[i * size + j] = (0..4).map(|t| wy[t] * rows[(i + t) * size + j]).sum();
            }
        }
        out
    }
}

/// One block's shift of a channel against green, in sensor pixels.
struct Block {
    /// Centre (y, x) relative to the image centre.
    p: [f32; 2],
    d: [f32; 2],
    /// Inverse covariance of `d`.
    w: [[f32; 2]; 2],
}

fn moments(v: &[f32]) -> (f32, f32) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// Zero mean, unit rms; false if flat.
fn normalise(v: &mut [f32]) -> bool {
    let (mean, sd) = moments(v);
    if sd.is_nan() || sd <= 0.0 {
        return false;
    }
    v.iter_mut().for_each(|x| *x = (*x - mean) / sd);
    true
}

/// `[ty, tx, k0, k1, k2]` of `d = t + (k0 + k1 rho^2 + k2 rho^4) p`,
/// weighted by each block's matrix and robust (Huber, iterated).
fn fit(blocks: &[&Block], radius: f32) -> [f64; 5] {
    let model = |b: &Block| -> [[f64; 5]; 2] {
        let r2 = (b.p[0].hypot(b.p[1]) / radius).powi(2) as f64;
        let s = [1.0, r2, r2 * r2];
        let (y, x) = (b.p[0] as f64, b.p[1] as f64);
        [
            [1.0, 0.0, y * s[0], y * s[1], y * s[2]],
            [0.0, 1.0, x * s[0], x * s[1], x * s[2]],
        ]
    };
    let mut weights = vec![1.0f64; blocks.len()];
    let mut theta = [0.0; 5];
    for _ in 0..10 {
        let mut m = [[0.0f64; 5]; 5];
        let mut v = [0.0f64; 5];
        for (b, &rw) in blocks.iter().zip(&weights) {
            let x = model(b);
            let w = b.w.map(|r| r.map(|v| v as f64 * rw));
            let d = [b.d[0] as f64, b.d[1] as f64];
            // X^T W
            let xtw: [[f64; 2]; 5] = std::array::from_fn(|i| {
                std::array::from_fn(|j| x[0][i] * w[0][j] + x[1][i] * w[1][j])
            });
            for i in 0..5 {
                for k in 0..5 {
                    m[i][k] += xtw[i][0] * x[0][k] + xtw[i][1] * x[1][k];
                }
                v[i] += xtw[i][0] * d[0] + xtw[i][1] * d[1];
            }
        }
        let Some(solution) = solve(m, v) else {
            break;
        };
        theta = solution;
        for (b, rw) in blocks.iter().zip(weights.iter_mut()) {
            let x = model(b);
            let e: [f64; 2] = std::array::from_fn(|r| {
                b.d[r] as f64 - (0..5).map(|i| x[r][i] * theta[i]).sum::<f64>()
            });
            let w = b.w.map(|r| r.map(|v| v as f64));
            let maha = (e[0] * (w[0][0] * e[0] + w[0][1] * e[1])
                + e[1] * (w[1][0] * e[0] + w[1][1] * e[1]))
                .max(0.0)
                .sqrt();
            *rw = if maha < HUBER { 1.0 } else { HUBER / maha };
        }
    }
    theta
}

/// `m x = v` by Gaussian elimination with partial pivoting.
fn solve<const N: usize>(mut m: [[f64; N]; N], mut v: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let pivot = (col..N).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[pivot][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, pivot);
        v.swap(col, pivot);
        let pivot_row = m[col];
        for row in col + 1..N {
            let f = m[row][col] / pivot_row[col];
            for (a, p) in m[row].iter_mut().zip(pivot_row).skip(col) {
                *a -= f * p;
            }
            v[row] -= f * v[col];
        }
    }
    let mut x = [0.0; N];
    for row in (0..N).rev() {
        let s: f64 = (row + 1..N).map(|k| m[row][k] * x[k]).sum();
        x[row] = (v[row] - s) / m[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smooth random texture: values on a 6 px grid, Catmull-Rom between.
    fn scene(y: f32, x: f32) -> f32 {
        let value = |i: i32, j: i32| {
            let h = (i as u32).wrapping_mul(0x9e37_79b9) ^ (j as u32).wrapping_mul(0x85eb_ca6b);
            let h = (h ^ (h >> 15)).wrapping_mul(0x2c1b_3c6d);
            ((h ^ (h >> 13)) & 0xffff) as f32 / 65535.0
        };
        let weights = |f: f32| {
            [
                f * (-0.5 + f * (1.0 - 0.5 * f)),
                1.0 + f * f * (-2.5 + 1.5 * f),
                f * (0.5 + f * (2.0 - 1.5 * f)),
                f * f * (-0.5 + 0.5 * f),
            ]
        };
        let (gy, gx) = (y / 6.0, x / 6.0);
        let (iy, ix) = (gy.floor(), gx.floor());
        let (wy, wx) = (weights(gy - iy), weights(gx - ix));
        let mut sum = 0.0;
        for (a, wa) in wy.iter().enumerate() {
            for (b, wb) in wx.iter().enumerate() {
                sum += wa * wb * value(iy as i32 + a as i32 - 1, ix as i32 + b as i32 - 1);
            }
        }
        sum
    }

    /// A Bayer and an X-Trans mosaic with a known CA, each colour with its
    /// own gain and offset: the measured shifts are within 0.1 px of it.
    #[test]
    fn measures_known_ca() {
        let truth = Measured {
            radius: 0.0,
            coefficients: [[1.5e-3, 0.0, 0.0], [-5e-4, 2.5e-3, 0.0]],
            reach: [2.0; 2],
            blocks: [0; 2],
            shift: [[0.0; 2]; 2],
        };
        let bayer = CfaPattern {
            width: 2,
            height: 2,
            colors: vec![0, 1, 1, 2],
        };
        let xtrans = CfaPattern {
            width: 6,
            height: 6,
            colors: vec![
                1, 1, 0, 1, 1, 2, 1, 1, 2, 1, 1, 0, 2, 0, 1, 0, 2, 1, //
                1, 1, 2, 1, 1, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1, 2, 0, 1,
            ],
        };
        let (width, height) = (1536, 1032);
        let radius = (width as f32).hypot(height as f32) / 2.0;
        let truth = Measured { radius, ..truth };
        let (cy, cx) = ((height as f32 - 1.0) / 2.0, (width as f32 - 1.0) / 2.0);
        for cfa in [bayer, xtrans] {
            let data: Vec<f32> = (0..width * height)
                .map(|i| {
                    let (y, x) = ((i / width) as f32, (i % width) as f32);
                    let c = cfa.colors[(i / width % cfa.height) * cfa.width + i % width % cfa.width]
                        as usize;
                    // channel c at q is the scene at c + (q - c) / (1 + s)
                    let s = match c {
                        0 => truth.scale(0, (y - cy).hypot(x - cx) / radius),
                        2 => truth.scale(1, (y - cy).hypot(x - cx) / radius),
                        _ => 0.0,
                    };
                    let v = scene(cy + (y - cy) / (1.0 + s), cx + (x - cx) / (1.0 + s));
                    [0.3 * v + 0.05, 0.6 * v, 0.2 * v + 0.1][c]
                })
                .collect();
            let mosaic = Mosaic {
                data: &data,
                width,
                height,
                cfa: &cfa,
                wb: [2.0, 1.0, 1.5],
            };
            let measured = measure_mosaic(&mosaic).expect("measured");
            println!(
                "{}x{}: {} vs {}",
                cfa.width,
                cfa.height,
                measured.summary(),
                truth.summary()
            );
            for c in 0..2 {
                for rho in [0.3, 0.5, 0.7, 0.9] {
                    let (got, want) = (measured.pixels(c, rho), truth.pixels(c, rho));
                    assert!(
                        (got - want).abs() < 0.1,
                        "{}x{} channel {c} rho {rho}: {got} vs {want}",
                        cfa.width,
                        cfa.height
                    );
                }
            }
        }
    }

    /// Measures `STARFISH_TEST_RAW`'s mosaic and prints the shifts by radius:
    /// `STARFISH_TEST_RAW=file.raf cargo test --release measure_file -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn measure_file() {
        let path = std::env::var("STARFISH_TEST_RAW").expect("set STARFISH_TEST_RAW");
        let raw = crate::raw::decode(std::path::Path::new(&path)).unwrap();
        let start = std::time::Instant::now();
        let measured = measure_raw(&raw);
        println!("{path}: {} ms", start.elapsed().as_millis());
        // The file's own table, for comparison.
        let (unit, table) = match &raw.lens {
            Some(lens) => {
                let ca = crate::lens::Corrections {
                    chromatic: true,
                    ..crate::lens::Corrections::NONE
                };
                lens.lut(raw.width as u32, raw.height as u32, ca)
            }
            None => (
                (raw.width as f32).hypot(raw.height as f32) / 2.0,
                vec![[1.0, 0.0, 0.0, 1.0]; crate::lens::LUT_SIZE],
            ),
        };
        if let Some(m) = measured {
            println!("blocks {:?}, shift {:?}", m.blocks, m.shift);
            println!(
                "radius {} coefficients {:?} reach {:?}",
                m.radius, m.coefficients, m.reach
            );
            println!("  rho   measured R / B (px)   table R / B (px)");
            for rho in [0.1, 0.28, 0.42, 0.55, 0.65, 0.75, 0.85, 0.95] {
                let r = rho * m.radius / unit;
                let i = (r / crate::lens::LUT_RHO_MAX * (crate::lens::LUT_SIZE - 1) as f32).round()
                    as usize;
                let px = |c: usize| table[i][c] * rho * m.radius;
                println!(
                    "  {rho:.2}  {:+.3} / {:+.3}     {:+.3} / {:+.3}",
                    m.pixels(0, rho),
                    m.pixels(1, rho),
                    px(1),
                    px(2)
                );
            }
        }
    }

    /// Measures a mosaic dumped by `raw::tests::dump_mosaic` (or written in
    /// its format), `STARFISH_TEST_MOSAIC`.
    #[test]
    #[ignore]
    fn measure_dump() {
        let path = std::env::var("STARFISH_TEST_MOSAIC").expect("set STARFISH_TEST_MOSAIC");
        let bytes = std::fs::read(path).unwrap();
        let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        let (height, width, cw, ch) = (
            word(0) as usize,
            word(1) as usize,
            word(2) as usize,
            word(3) as usize,
        );
        let colors: Vec<u32> = (0..cw * ch).map(|i| word(4 + i)).collect();
        let float = |i: usize| f32::from_bits(word(i));
        let at = 4 + cw * ch;
        let wb = [float(at), float(at + 1), float(at + 2)];
        let data: Vec<f32> = (0..width * height)
            .map(|i| float(at + 4 + 2 * crate::lens::LUT_SIZE + i))
            .collect();
        let cfa = CfaPattern {
            width: cw,
            height: ch,
            colors,
        };
        let m = measure_mosaic(&Mosaic {
            data: &data,
            width,
            height,
            cfa: &cfa,
            wb,
        })
        .unwrap();
        println!("blocks {:?}, shift {:?}", m.blocks, m.shift);
        println!(
            "radius {} coefficients {:?} reach {:?}",
            m.radius, m.coefficients, m.reach
        );
        for rho in [0.1, 0.28, 0.42, 0.55, 0.65, 0.75, 0.85, 0.95] {
            println!(
                "  {rho:.2}  R {:+.3}  B {:+.3}",
                m.pixels(0, rho),
                m.pixels(1, rho)
            );
        }
    }

    /// Measures a planar f32 camera RGB dump (`develop_file`'s
    /// `STARFISH_TEST_DUMP`, `STARFISH_TEST_LENS=0` for the network's own
    /// output) of `STARFISH_TEST_SIZE=WxH`: the CA left after the demosaic.
    #[test]
    #[ignore]
    fn measure_rgb_dump() {
        let path = std::env::var("STARFISH_TEST_RGB").expect("set STARFISH_TEST_RGB");
        let size = std::env::var("STARFISH_TEST_SIZE").expect("set STARFISH_TEST_SIZE");
        let (w, h) = size.split_once('x').unwrap();
        let (sw, sh): (usize, usize) = (w.parse().unwrap(), h.parse().unwrap());
        let bytes = std::fs::read(path).unwrap();
        let v = |i: usize| f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        let (width, height) = (sw / 2, sh / 2);
        let planes: [Vec<f32>; 3] = std::array::from_fn(|c| {
            (0..width * height)
                .map(|k| {
                    let (i, j) = (k / width, k % width);
                    let at = |y: usize, x: usize| v(c * sw * sh + y * sw + x);
                    (at(2 * i, 2 * j)
                        + at(2 * i, 2 * j + 1)
                        + at(2 * i + 1, 2 * j)
                        + at(2 * i + 1, 2 * j + 1))
                        / 4.0
                })
                .collect()
        });
        let clipped = (0..width * height)
            .map(|k| (0..3).any(|c| planes[c][k] > 0.95))
            .collect();
        let planes = Planes {
            width,
            height,
            sensor: (sw, sh),
            planes,
            clipped,
        };
        let m = measure_planes(&planes).unwrap();
        for rho in [0.1, 0.28, 0.42, 0.55, 0.65, 0.75, 0.85, 0.95] {
            println!(
                "  {rho:.2}  R {:+.3}  B {:+.3}",
                m.pixels(0, rho),
                m.pixels(1, rho)
            );
        }
    }
}
