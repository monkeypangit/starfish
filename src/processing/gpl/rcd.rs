// SPDX-License-Identifier: GPL-3.0-or-later
//
// Ratio Corrected Demosaicing, a port of darktable's `rcd.c` (Copyright (C)
// 2010-2026 darktable developers), which is Luis Sanz Rodriguez's RCD 2.3
// (https://github.com/LuisSR/RCD-Demosaicing, GPL-3) with Ingo Weyrich's
// tiling. See the module docs for the licence this brings with it.
//
// Differences from darktable: the image is mirrored `BORDER` px beyond its
// edges (reflecting about the edge pixel keeps the Bayer phase) instead of
// filling the outer ten pixels with PPG, and it runs in strips of the full
// width instead of square tiles. The arithmetic in each strip is the same.

/// Pixels around a strip that are not trusted (darktable's RCD_BORDER).
const BORDER: usize = 10;
/// Rows of the image one strip produces.
const STRIP: usize = 64;
const EPS: f32 = 1e-5;
const EPSSQ: f32 = 1e-10;

fn sq(x: f32) -> f32 {
    x * x
}

/// darktable's `interpolatef`: `a` of `b`, the rest of `c`.
fn lerp(a: f32, b: f32, c: f32) -> f32 {
    a * (b - c) + c
}

fn clip(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Mirrors `n` into `0..size` about the edge pixels (reflect-101), which
/// keeps the parity of the index and so the Bayer phase.
fn mirror(n: isize, size: usize) -> usize {
    let size = size as isize;
    let period = 2 * (size - 1);
    let m = n.rem_euclid(period.max(1));
    (if m < size { m } else { period - m }) as usize
}

/// The demosaic of a Bayer frame of `w` x `h` samples, `colors` its 2x2 pattern
/// (0 = R, 1 = G, 2 = B): interleaved R, G, B (as sampled, no white balance),
/// `None` if cancelled.
pub fn demosaic(
    data: &[f32],
    w: usize,
    h: usize,
    colors: &[u32],
    cancel: &std::sync::atomic::AtomicBool,
) -> Option<Vec<f32>> {
    // padded by BORDER on each side; the stride is even, as the half-size
    // buffers below index by position / 2
    let (pw, ph) = (w + 2 * BORDER, h + 2 * BORDER);
    let stride = pw.next_multiple_of(2);
    let mut padded = vec![0f32; stride * ph];
    for y in 0..ph {
        let sy = mirror(y as isize - BORDER as isize, h);
        for x in 0..pw {
            let sx = mirror(x as isize - BORDER as isize, w);
            padded[y * stride + x] = data[sy * w + sx];
        }
    }
    // the colour at padded (y, x): BORDER is even, so the pattern is the frame's
    let colour = |y: usize, x: usize| colors[(y % 2) * 2 + x % 2] as usize;

    let strips = h.div_ceil(STRIP);
    let results = super::par_map(strips, cancel, |k| {
        let first = k * STRIP;
        let last = (first + STRIP).min(h);
        let y0 = first; // = BORDER + first - BORDER
        let rows = last - first + 2 * BORDER;
        let block = &padded[y0 * stride..(y0 + rows) * stride];
        let rgb = strip(block, rows, pw, stride, |r, c| colour(y0 + r, c));
        // the image's rows of this strip, interleaved
        let mut out = Vec::with_capacity((last - first) * w * 3);
        for r in BORDER..BORDER + (last - first) {
            for c in BORDER..BORDER + w {
                let i = r * stride + c;
                out.extend([rgb[0][i].max(0.0), rgb[1][i].max(0.0), rgb[2][i].max(0.0)]);
            }
        }
        out
    })?;
    Some(results.concat())
}

/// RCD on `rows` x `cols` of `cfa` (row stride `stride`, even); `fc` gives a
/// site's colour (0 = R, 1 = G, 2 = B). The returned planes are valid more
/// than `BORDER` px from the block's edges.
fn strip(
    input: &[f32],
    rows: usize,
    cols: usize,
    stride: usize,
    fc: impl Fn(usize, usize) -> usize,
) -> [Vec<f32>; 3] {
    let n = stride * rows;
    let w1 = stride as isize;
    let (w2, w3, w4) = (2 * w1, 3 * w1, 4 * w1);
    let mut cfa = vec![0f32; n];
    let mut rgb = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
    let mut vh_dir = vec![0f32; n];
    let half = n / 2 + stride;
    let mut pq_dir = vec![0f32; half];
    let mut lpf = vec![0f32; half];
    let mut p_hpf = vec![0f32; half];
    let mut q_hpf = vec![0f32; half];

    // Step 0: fill the data and make sure it is not negative
    for row in 0..rows {
        let (c0, c1) = (fc(row, 0), fc(row, 1));
        for col in 0..cols {
            let i = row * stride + col;
            let v = input[i].max(0.0);
            cfa[i] = v;
            rgb[c0][i] = v;
            rgb[c1][i] = v;
        }
    }
    let at = |a: &[f32], i: isize| a[i as usize];

    // STEP 1: vertical and horizontal interpolation directions
    // 1.1: the squared vertical and horizontal colour difference high pass
    let mut vsq = vec![0f32; n];
    let mut hsq = vec![0f32; n];
    for row in 3..rows - 3 {
        for col in 4..cols - 4 {
            let i = (row * stride + col) as isize;
            vsq[i as usize] = sq((at(&cfa, i - w3) - at(&cfa, i - w1) - at(&cfa, i + w1)
                + at(&cfa, i + w3))
                - 3.0 * (at(&cfa, i - w2) + at(&cfa, i + w2))
                + 6.0 * at(&cfa, i));
        }
    }
    for row in 4..rows - 4 {
        for col in 3..cols - 3 {
            let i = (row * stride + col) as isize;
            hsq[i as usize] = sq((at(&cfa, i - 3) - at(&cfa, i - 1) - at(&cfa, i + 1)
                + at(&cfa, i + 3))
                - 3.0 * (at(&cfa, i - 2) + at(&cfa, i + 2))
                + 6.0 * at(&cfa, i));
        }
    }
    // 1.2: the directional discrimination strength
    for row in 4..rows - 4 {
        for col in 4..cols - 4 {
            let i = (row * stride + col) as isize;
            let v_stat = EPSSQ.max(at(&vsq, i - w1) + at(&vsq, i) + at(&vsq, i + w1));
            let h_stat = EPSSQ.max(at(&hsq, i - 1) + at(&hsq, i) + at(&hsq, i + 1));
            vh_dir[i as usize] = v_stat / (v_stat + h_stat);
        }
    }

    // STEP 2: the low pass filter, at the red and blue sites
    for row in 2..rows - 2 {
        let mut col = 2 + (fc(row, 0) & 1);
        while col < cols - 2 {
            let i = (row * stride + col) as isize;
            lpf[(i / 2) as usize] = at(&cfa, i)
                + 0.5 * (at(&cfa, i - w1) + at(&cfa, i + w1) + at(&cfa, i - 1) + at(&cfa, i + 1))
                + 0.25
                    * (at(&cfa, i - w1 - 1)
                        + at(&cfa, i - w1 + 1)
                        + at(&cfa, i + w1 - 1)
                        + at(&cfa, i + w1 + 1));
            col += 2;
        }
    }

    // The refined vertical / horizontal discrimination at `i`
    let refined = |dir: &[f32], i: isize| {
        let central = at(dir, i);
        let neighbourhood = 0.25
            * (at(dir, i - w1 - 1)
                + at(dir, i - w1 + 1)
                + at(dir, i + w1 - 1)
                + at(dir, i + w1 + 1));
        if (0.5 - central).abs() < (0.5 - neighbourhood).abs() {
            neighbourhood
        } else {
            central
        }
    };

    // STEP 3: green at the red and blue sites
    for row in 4..rows - 4 {
        let mut col = 4 + (fc(row, 0) & 1);
        while col < cols - 4 {
            let i = (row * stride + col) as isize;
            let lp = (i / 2) as usize;
            let cfai = at(&cfa, i);
            let c = |o: isize| at(&cfa, i + o);

            // cardinal gradients
            let n_grad = EPS
                + (c(-w1) - c(w1)).abs()
                + (cfai - c(-w2)).abs()
                + (c(-w1) - c(-w3)).abs()
                + (c(-w2) - c(-w4)).abs();
            let s_grad = EPS
                + (c(w1) - c(-w1)).abs()
                + (cfai - c(w2)).abs()
                + (c(w1) - c(w3)).abs()
                + (c(w2) - c(w4)).abs();
            let w_grad = EPS
                + (c(-1) - c(1)).abs()
                + (cfai - c(-2)).abs()
                + (c(-1) - c(-3)).abs()
                + (c(-2) - c(-4)).abs();
            let e_grad = EPS
                + (c(1) - c(-1)).abs()
                + (cfai - c(2)).abs()
                + (c(1) - c(3)).abs()
                + (c(2) - c(4)).abs();

            // cardinal pixel estimations
            let lpfi = lpf[lp];
            let l = |o: usize, up: bool| if up { lpf[lp - o] } else { lpf[lp + o] };
            let n_est = c(-w1) * (lpfi + lpfi) / (EPS + lpfi + l(stride, true));
            let s_est = c(w1) * (lpfi + lpfi) / (EPS + lpfi + l(stride, false));
            let w_est = c(-1) * (lpfi + lpfi) / (EPS + lpfi + l(1, true));
            let e_est = c(1) * (lpfi + lpfi) / (EPS + lpfi + l(1, false));

            // vertical and horizontal estimations
            let v_est = (s_grad * n_est + n_grad * s_est) / (n_grad + s_grad);
            let h_est = (w_grad * e_est + e_grad * w_est) / (e_grad + w_grad);

            rgb[1][i as usize] = lerp(clip(refined(&vh_dir, i)), h_est, v_est);
            col += 2;
        }
    }

    // STEP 4: red and blue
    // 4.0: the squared P / Q diagonals colour difference high pass
    for row in 3..rows - 3 {
        let mut col = 3;
        while col < cols - 3 {
            let i = (row * stride + col) as isize;
            let c = |o: isize| at(&cfa, i + o);
            p_hpf[(i / 2) as usize] = sq((c(-w3 - 3) - c(-w1 - 1) - c(w1 + 1) + c(w3 + 3))
                - 3.0 * (c(-w2 - 2) + c(w2 + 2))
                + 6.0 * c(0));
            q_hpf[(i / 2) as usize] = sq((c(-w3 + 3) - c(-w1 + 1) - c(w1 - 1) + c(w3 - 3))
                - 3.0 * (c(-w2 + 2) + c(w2 - 2))
                + 6.0 * c(0));
            col += 2;
        }
    }
    // 4.1: the P / Q diagonals directional discrimination strength
    for row in 4..rows - 4 {
        let mut col = 4 + (fc(row, 0) & 1);
        while col < cols - 4 {
            let i = (row * stride + col) as isize;
            let (i2, i3, i4) = (i / 2, (i - w1 - 1) / 2, (i + w1 - 1) / 2);
            let p_stat = EPSSQ.max(at(&p_hpf, i3) + at(&p_hpf, i2) + at(&p_hpf, i4 + 1));
            let q_stat = EPSSQ.max(at(&q_hpf, i3 + 1) + at(&q_hpf, i2) + at(&q_hpf, i4));
            pq_dir[i2 as usize] = p_stat / (p_stat + q_stat);
            col += 2;
        }
    }
    // 4.2: red and blue at the blue and red sites
    for row in 4..rows - 4 {
        let mut col = 4 + (fc(row, 0) & 1);
        while col < cols - 4 {
            let i = (row * stride + col) as isize;
            let c = 2 - fc(row, col);
            let (pq, pq2, pq3) = (i / 2, (i - w1 - 1) / 2, (i + w1 - 1) / 2);

            // refined P / Q diagonal local discrimination
            let central = at(&pq_dir, pq);
            let neighbourhood = 0.25
                * (at(&pq_dir, pq2)
                    + at(&pq_dir, pq2 + 1)
                    + at(&pq_dir, pq3)
                    + at(&pq_dir, pq3 + 1));
            let disc = if (0.5 - central).abs() < (0.5 - neighbourhood).abs() {
                neighbourhood
            } else {
                central
            };

            let rc = |o: isize| at(&rgb[c], i + o);
            let rg = |o: isize| at(&rgb[1], i + o);

            // diagonal gradients
            let nw_grad = EPS
                + (rc(-w1 - 1) - rc(w1 + 1)).abs()
                + (rc(-w1 - 1) - rc(-w3 - 3)).abs()
                + (rg(0) - rg(-w2 - 2)).abs();
            let ne_grad = EPS
                + (rc(-w1 + 1) - rc(w1 - 1)).abs()
                + (rc(-w1 + 1) - rc(-w3 + 3)).abs()
                + (rg(0) - rg(-w2 + 2)).abs();
            let sw_grad = EPS
                + (rc(-w1 + 1) - rc(w1 - 1)).abs()
                + (rc(w1 - 1) - rc(w3 - 3)).abs()
                + (rg(0) - rg(w2 - 2)).abs();
            let se_grad = EPS
                + (rc(-w1 - 1) - rc(w1 + 1)).abs()
                + (rc(w1 + 1) - rc(w3 + 3)).abs()
                + (rg(0) - rg(w2 + 2)).abs();

            // diagonal colour differences
            let nw_est = rc(-w1 - 1) - rg(-w1 - 1);
            let ne_est = rc(-w1 + 1) - rg(-w1 + 1);
            let sw_est = rc(w1 - 1) - rg(w1 - 1);
            let se_est = rc(w1 + 1) - rg(w1 + 1);

            // P / Q estimations
            let p_est = (nw_grad * se_est + se_grad * nw_est) / (nw_grad + se_grad);
            let q_est = (ne_grad * sw_est + sw_grad * ne_est) / (ne_grad + sw_grad);

            rgb[c][i as usize] = rg(0) + lerp(clip(disc), q_est, p_est);
            col += 2;
        }
    }
    // 4.3: red and blue at the green sites
    for row in 4..rows - 4 {
        let mut col = 4 + (fc(row, 1) & 1);
        while col < cols - 4 {
            let i = (row * stride + col) as isize;
            let disc = refined(&vh_dir, i);
            let g = |o: isize| at(&rgb[1], i + o);
            let rgb1 = g(0);
            let n1 = EPS + (rgb1 - g(-w2)).abs();
            let s1 = EPS + (rgb1 - g(w2)).abs();
            let w1_ = EPS + (rgb1 - g(-2)).abs();
            let e1 = EPS + (rgb1 - g(2)).abs();
            let (g_n, g_s, g_w, g_e) = (g(-w1), g(w1), g(-1), g(1));

            for c in [0, 2] {
                let r = |o: isize| at(&rgb[c], i + o);
                let sn_abs = (r(-w1) - r(w1)).abs();
                let ew_abs = (r(-1) - r(1)).abs();

                // cardinal gradients
                let n_grad = n1 + sn_abs + (r(-w1) - r(-w3)).abs();
                let s_grad = s1 + sn_abs + (r(w1) - r(w3)).abs();
                let w_grad = w1_ + ew_abs + (r(-1) - r(-3)).abs();
                let e_grad = e1 + ew_abs + (r(1) - r(3)).abs();

                // cardinal colour differences
                let n_est = r(-w1) - g_n;
                let s_est = r(w1) - g_s;
                let w_est = r(-1) - g_w;
                let e_est = r(1) - g_e;

                // vertical and horizontal estimations
                let v_est = (n_grad * s_est + s_grad * n_est) / (n_grad + s_grad);
                let h_est = (e_grad * w_est + w_grad * e_est) / (e_grad + w_grad);

                let value = rgb1 + lerp(clip(disc), h_est, v_est);
                rgb[c][i as usize] = value;
            }
            col += 2;
        }
    }
    rgb
}
