//! The network's noise input: estimate_noise (rawload.py's flat-block fit of
//! sigma^2 = a*x + b on the greens, with a robust line) and inputs(), which turns
//! an un-balanced mosaic into the balanced mosaic and sigma map Model::run takes.

/// The noise ranges the models were trained on (training/common.py A_RANGE, B_RANGE).
pub const A_RANGE: (f64, f64) = (1e-5, 1e-2);
pub const B_RANGE: (f64, f64) = (1e-8, 1e-4);

/// The network's input from an un-balanced h x w mosaic phased to `cfa`
/// (period p): sigma = sqrt(a*x + b) in sensor units, then mosaic and sigma
/// both balanced by the site's gain. Returns (mosaic, sigma).
pub fn inputs(
    m: &[f32],
    h: usize,
    w: usize,
    cfa: &[u32],
    p: usize,
    gain: [f32; 3],
    a: f64,
    b: f64,
) -> (Vec<f32>, Vec<f32>) {
    let mut mos = vec![0f32; h * w];
    let mut sig = vec![0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let v = m[y * w + x];
            let g = gain[cfa[(y % p) * p + x % p] as usize];
            sig[y * w + x] = ((a * v.max(0.0) as f64 + b).sqrt() as f32) * g;
            mos[y * w + x] = v * g;
        }
    }
    (mos, sig)
}

/// torch.quantile's linear interpolation; sorts `v`.
fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let pos = q * (v.len() - 1) as f64;
    let (i, f) = (pos.floor() as usize, pos - pos.floor());
    if i + 1 < v.len() {
        v[i] + f * (v[i + 1] - v[i])
    } else {
        v[i]
    }
}

/// torch.median: the lower of the two middle values.
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[(v.len() - 1) / 2]
}

/// `f(i)` for every i in 0..n, in order, spread over the cores. In one thread
/// on wasm, which has no threads to spawn.
pub fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if cfg!(target_arch = "wasm32") {
        return (0..n).map(f).collect();
    }
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let chunk = n.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let f = &f;
        let parts: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|i0| scope.spawn(move || (i0..(i0 + chunk).min(n)).map(f).collect::<Vec<_>>()))
            .collect();
        parts.into_iter().flat_map(|p| p.join().unwrap()).collect()
    })
}

/// Each block's mean, variance and brightest sample, and the samples (or
/// pairs) per block.
type Blocks = (Vec<f64>, Vec<f64>, Vec<f64>, f64);

/// rawload._blocks_bayer: the RGGB green sub-lattice at (0, 1) as a dense
/// image, differenced along rows.
fn blocks_bayer(m: &[f32], h: usize, w: usize, block: usize) -> Blocks {
    let (gh, gw) = (h / 2, w / 2);
    let g = |y: usize, x: usize| m[(2 * y) * w + 2 * x + 1] as f64;
    let (bh, bw) = (gh / block, (gw - 1) / block);
    let rows = par_map(bh, |by| {
        let (mut lvl, mut var, mut peak) = (vec![], vec![], vec![]);
        for bx in 0..bw {
            let (mut s, mut d2, mut top) = (0.0, 0.0, f64::MIN);
            for y in by * block..(by + 1) * block {
                for x in bx * block..(bx + 1) * block {
                    s += g(y, x);
                    d2 += (g(y, x) - g(y, x + 1)).powi(2);
                    top = top.max(g(y, x));
                }
            }
            let n = (block * block) as f64;
            lvl.push(s / n);
            var.push(d2 / n / 2.0);
            peak.push(top);
        }
        (lvl, var, peak)
    });
    let (mut lvl, mut var, mut peak) = (vec![], vec![], vec![]);
    for (l, v, t) in rows {
        lvl.extend(l);
        var.extend(v);
        peak.extend(t);
    }
    (lvl, var, peak, (block * block) as f64)
}

/// rawload._blocks_cfa: every horizontally neighbouring green pair at
/// distance 1 or 2, pooled over full-res block x block tiles.
fn blocks_cfa(
    m: &[f32],
    h: usize,
    w: usize,
    cfa: &[u32],
    p: usize,
    block: usize,
) -> Result<Blocks, String> {
    let green = |y: usize, x: usize| cfa[(y % p) * p + x % p] == 1;
    let rows = par_map(h / block, |by| {
        let (mut lvl, mut var, mut peak, mut counts) = (vec![], vec![], vec![], vec![]);
        for bx in 0..w / block {
            let (mut d2, mut n, mut s, mut ng, mut top) = (0.0, 0usize, 0.0, 0usize, f64::MIN);
            for y in by * block..(by + 1) * block {
                for x in bx * block..(bx + 1) * block {
                    if !green(y, x) {
                        continue;
                    }
                    let v = m[y * w + x] as f64;
                    s += v;
                    ng += 1;
                    top = top.max(v);
                    for dist in [1, 2] {
                        if x + dist < w && green(y, x + dist) {
                            d2 += (v - m[y * w + x + dist] as f64).powi(2);
                            n += 1;
                        }
                    }
                }
            }
            if n >= 8 {
                var.push(d2 / n as f64 / 2.0);
                lvl.push(s / ng.max(1) as f64);
                peak.push(top);
                counts.push(n as f64);
            }
        }
        (lvl, var, peak, counts)
    });
    let (mut lvl, mut var, mut peak, mut counts) = (vec![], vec![], vec![], vec![]);
    for (l, v, t, c) in rows {
        lvl.extend(l);
        var.extend(v);
        peak.extend(t);
        counts.extend(c);
    }
    if lvl.len() < 2 {
        return Err("noise estimate failed (no green pairs); pass --a/--b".into());
    }
    let mean_n = counts.iter().sum::<f64>() / counts.len() as f64;
    Ok((lvl, var, peak, mean_n))
}

/// rawload.estimate_noise: fit sigma^2 = a*x + b under the scene's texture
/// (`floor_line`), clipped blocks left out, clamped to the trained `A_RANGE`
/// / `B_RANGE`. Returns (a, b).
pub fn estimate_noise(
    m: &[f32],
    h: usize,
    w: usize,
    cfa: &[u32],
    p: usize,
) -> Result<(f64, f64), String> {
    let (block, q, bins, min_blocks) = (8, 0.02, 24, 32);
    let (mean, var, peak, n) = if p == 2 {
        blocks_bayer(m, h, w, block)
    } else {
        blocks_cfa(m, h, w, cfa, p, block)?
    };
    // A block reaching the clip level has lost its noise (a near-zero
    // variance far below the line, which made the old fit's slope negative).
    // The white level is not always where the data clips (Nikon Z5: 1.06), so
    // the brightest sample stands for it.
    let clip = 0.95 * peak.iter().fold(f64::MIN, |a, &b| a.max(b));
    let keep: Vec<usize> = (0..mean.len()).filter(|&k| peak[k] < clip).collect();
    let mean: Vec<f64> = keep.iter().map(|&k| mean[k]).collect();
    const Z: f64 = -2.053748910631823; // Normal icdf(0.02)
    let var: Vec<f64> = keep
        .iter()
        .map(|&k| var[k] / (1.0 + Z * (2.0 / n).sqrt())) // undo the quantile's bias
        .collect();
    if mean.len() < 2 {
        return Err("noise estimate failed (all blocks clipped); pass --a/--b".into());
    }

    let mut sorted = mean.clone();
    let lo = quantile(&mut sorted, 0.01);
    let hi = quantile(&mut sorted, 0.99);
    let (mut xs, mut ys, mut ns) = (vec![], vec![], vec![]);
    for i in 0..bins {
        let e0 = lo + (hi - lo) * i as f64 / bins as f64;
        let e1 = lo + (hi - lo) * (i + 1) as f64 / bins as f64;
        let sel: Vec<usize> = (0..mean.len())
            .filter(|&k| mean[k] >= e0 && mean[k] < e1)
            .collect();
        if sel.len() >= min_blocks {
            xs.push(median(
                &mut sel.iter().map(|&k| mean[k]).collect::<Vec<_>>(),
            ));
            ys.push(quantile(
                &mut sel.iter().map(|&k| var[k]).collect::<Vec<_>>(),
                q,
            ));
            ns.push(sel.len() as f64);
        }
    }
    if xs.len() < 2 {
        return Err("noise estimate failed (too few flat blocks); pass --a/--b".into());
    }
    let (mut a, mut b) = floor_line(&xs, &ys, &ns).unwrap_or((0.0, 0.0));
    if a <= 0.0 {
        // pure read noise (or a fit the scene defeated): flat sigma
        a = 0.0;
        b = median(&mut ys.clone());
    }
    // the network was trained on these ranges only
    Ok((a.clamp(A_RANGE.0, A_RANGE.1), b.clamp(B_RANGE.0, B_RANGE.1)))
}

/// The noise line under the bins (level `xs`, variance `ys`, `ns` blocks).
/// Texture only ever raises a bin's variance, and a level range with no flat
/// area (bright blossom against twigs, foliage between dark ground and sky)
/// has nothing else to give: fitted through every bin, such bins set the
/// slope (300x too high on one A7 III file). So every line through two bins is
/// a candidate if it predicts a positive variance at every bin and no bin lies
/// well below it; the one with the most blocks on it (within `TOLERANCE`)
/// wins, and the line is refitted to those bins, weighted by block count. A
/// lower envelope, as Liu et al. (2006) and Azzari & Foi (2014) fit one.
fn floor_line(xs: &[f64], ys: &[f64], ns: &[f64]) -> Option<(f64, f64)> {
    const TOLERANCE: f64 = 1.5;
    let mut best: Option<(f64, Vec<bool>)> = None;
    for i in 0..xs.len() {
        for j in i + 1..xs.len() {
            let a = (ys[j] - ys[i]) / (xs[j] - xs[i]);
            let b = ys[i] - a * xs[i];
            let fit = |k: usize| a * xs[k] + b;
            if (0..xs.len()).any(|k| fit(k) <= 0.0 || ys[k] < fit(k) / TOLERANCE) {
                continue;
            }
            let on: Vec<bool> = (0..xs.len()).map(|k| ys[k] <= fit(k) * TOLERANCE).collect();
            let blocks: f64 = ns.iter().zip(&on).filter(|(_, &o)| o).map(|(n, _)| n).sum();
            if best.as_ref().is_none_or(|(most, _)| blocks > *most) {
                best = Some((blocks, on));
            }
        }
    }
    let (_, on) = best?;
    let pick = |v: &[f64]| -> Vec<f64> {
        v.iter()
            .zip(&on)
            .filter(|(_, &o)| o)
            .map(|(&x, _)| x)
            .collect()
    };
    Some(weighted_line(&pick(xs), &pick(ys), &pick(ns)))
}

/// Least squares y = a*x + b, each point weighted by `n`. Returns (a, b).
fn weighted_line(xs: &[f64], ys: &[f64], n: &[f64]) -> (f64, f64) {
    let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for ((&x, &y), &w) in xs.iter().zip(ys).zip(n) {
        sw += w;
        sx += w * x;
        sy += w * y;
        sxx += w * x * x;
        sxy += w * x * y;
    }
    let a = (sw * sxy - sx * sy) / (sw * sxx - sx * sx);
    (a, (sy - a * sx) / sw)
}
