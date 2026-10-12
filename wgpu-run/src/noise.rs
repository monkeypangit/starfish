//! The network's noise input: estimate_noise (a port of rawload.py's
//! flat-block fit of sigma^2 = a*x + b on the greens) and inputs(), which turns
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

/// `f(i)` for every i in 0..n, in order, spread over the cores.
fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
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

/// rawload._blocks_bayer: the RGGB green sub-lattice at (0, 1) as a dense
/// image, differenced along rows.
fn blocks_bayer(m: &[f32], h: usize, w: usize, block: usize) -> (Vec<f64>, Vec<f64>, f64) {
    let (gh, gw) = (h / 2, w / 2);
    let g = |y: usize, x: usize| m[(2 * y) * w + 2 * x + 1] as f64;
    let (bh, bw) = (gh / block, (gw - 1) / block);
    let rows = par_map(bh, |by| {
        let (mut lvl, mut var) = (vec![], vec![]);
        for bx in 0..bw {
            let (mut s, mut d2) = (0.0, 0.0);
            for y in by * block..(by + 1) * block {
                for x in bx * block..(bx + 1) * block {
                    s += g(y, x);
                    d2 += (g(y, x) - g(y, x + 1)).powi(2);
                }
            }
            let n = (block * block) as f64;
            lvl.push(s / n);
            var.push(d2 / n / 2.0);
        }
        (lvl, var)
    });
    let (mut lvl, mut var) = (vec![], vec![]);
    for (l, v) in rows {
        lvl.extend(l);
        var.extend(v);
    }
    (lvl, var, (block * block) as f64)
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
) -> Result<(Vec<f64>, Vec<f64>, f64), String> {
    let green = |y: usize, x: usize| cfa[(y % p) * p + x % p] == 1;
    let rows = par_map(h / block, |by| {
        let (mut lvl, mut var, mut counts) = (vec![], vec![], vec![]);
        for bx in 0..w / block {
            let (mut d2, mut n, mut s, mut ng) = (0.0, 0usize, 0.0, 0usize);
            for y in by * block..(by + 1) * block {
                for x in bx * block..(bx + 1) * block {
                    if !green(y, x) {
                        continue;
                    }
                    let v = m[y * w + x] as f64;
                    s += v;
                    ng += 1;
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
                counts.push(n as f64);
            }
        }
        (lvl, var, counts)
    });
    let (mut lvl, mut var, mut counts) = (vec![], vec![], vec![]);
    for (l, v, c) in rows {
        lvl.extend(l);
        var.extend(v);
        counts.extend(c);
    }
    if lvl.len() < 2 {
        return Err("noise estimate failed (no green pairs); pass --a/--b".into());
    }
    let mean_n = counts.iter().sum::<f64>() / counts.len() as f64;
    Ok((lvl, var, mean_n))
}

/// rawload.estimate_noise: fit sigma^2 = a*x + b. Returns (a, b).
pub fn estimate_noise(
    m: &[f32],
    h: usize,
    w: usize,
    cfa: &[u32],
    p: usize,
) -> Result<(f64, f64), String> {
    let (block, q, bins, min_blocks) = (8, 0.02, 24, 32);
    let (mean, mut var, n) = if p == 2 {
        blocks_bayer(m, h, w, block)
    } else {
        blocks_cfa(m, h, w, cfa, p, block)?
    };
    const Z: f64 = -2.053748910631823; // Normal icdf(0.02)
    var.iter_mut()
        .for_each(|v| *v /= 1.0 + Z * (2.0 / n).sqrt()); // undo the quantile's bias

    let mut sorted = mean.clone();
    let lo = quantile(&mut sorted, 0.01);
    let hi = quantile(&mut sorted, 0.99);
    let (mut xs, mut ys) = (vec![], vec![]);
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
        }
    }
    if xs.len() < 2 {
        return Err("noise estimate failed (too few flat blocks); pass --a/--b".into());
    }
    // least squares y = a*x + b
    let k = xs.len() as f64;
    let (sx, sy) = (xs.iter().sum::<f64>(), ys.iter().sum::<f64>());
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| x * y).sum();
    let mut a = (k * sxy - sx * sy) / (k * sxx - sx * sx);
    let mut b = (sy - a * sx) / k;
    if a <= 0.0 {
        // pure read noise (or a fit the scene defeated): flat sigma
        a = 0.0;
        b = median(&mut ys.clone());
    }
    Ok((a.max(0.0), b.max(1e-8)))
}
