//! Hot and dead pixel removal on the raw mosaic, before anything else looks at
//! it (the noise fit, the CA, the network): a sensor site that is stuck high or
//! low is far outside the samples of its own colour around it, which a
//! network would otherwise take for detail and draw as a coloured speck.
//!
//! A site is a *defect* when it lies outside the range of its same-colour
//! neighbours (every site of its colour within `RADIUS` px, so 8 on Bayer red
//! and blue, more on green and X-Trans) by more than `K` times that range plus
//! `FLOOR`. The range is the neighbourhood's own noise and texture, so a noisy
//! or textured area needs a bigger outlier, and a flat one a smaller; a thin
//! line or an edge has neighbours of its own value along it and is kept. The
//! defect is replaced by the median of those neighbours. Works on any CFA:
//! the neighbours come from the pattern, not from a fixed layout.
//!
//! By default (`EXTREME_ONLY`) a defect must also be at an extreme: stuck on
//! (at or above the clip level, with neighbours that are not clipped) or dead
//! (at black, with neighbours clearly above it). Partly elevated "warm" pixels
//! are then left alone, which keeps real detail safe.
//!
//! Known limit: a point of light sharper than one pixel (no blur at all)
//! cannot be told from a hot pixel, and two defects of one colour within
//! `RADIUS` px of each other protect each other.

use crate::raw::CfaPattern;

/// Same-colour neighbours are searched this far (Chebyshev), px.
const RADIUS: isize = 3;
/// Fewer neighbours than this and the site is left alone (image corners).
const MIN_NEIGHBOURS: usize = 3;
/// How many times the neighbours' range the site must lie beyond them.
const K: f32 = 3.0;
/// And at least this much beyond them (white = 1).
const FLOOR: f32 = 0.005;
/// Only sites at an extreme are defects: stuck on or dead.
const EXTREME_ONLY: bool = true;
/// Stuck on: at or above this (white = 1), as the highlight code's clip level.
const STUCK_ON: f32 = 0.99;
/// Dead: at or below this, with the neighbours' median above `DEAD_NEIGHBOURS`.
const DEAD: f32 = 0.002;
const DEAD_NEIGHBOURS: f32 = 0.02;
/// The most same-colour neighbours a site can have within `RADIUS`.
const MAX_NEIGHBOURS: usize = ((2 * RADIUS + 1) * (2 * RADIUS + 1)) as usize - 1;

/// For each of `sites` `(y, x)` inside the frame that has same-colour
/// neighbours, `(index, median of them)`: what the user's picked pixels are
/// replaced by, whatever their value.
pub fn replacements(
    data: &[f32],
    w: usize,
    h: usize,
    cfa: &CfaPattern,
    sites: &[(usize, usize)],
) -> Vec<(usize, f32)> {
    let offsets = phase_offsets(cfa);
    let mut near = [0f32; MAX_NEIGHBOURS];
    sites
        .iter()
        .filter(|&&(y, x)| y < h && x < w)
        .filter_map(|&(y, x)| {
            let mut n = 0;
            for &(dy, dx) in &offsets[(y % cfa.height) * cfa.width + x % cfa.width] {
                let (yy, xx) = (y as isize + dy, x as isize + dx);
                if yy >= 0 && xx >= 0 && yy < h as isize && xx < w as isize {
                    near[n] = data[yy as usize * w + xx as usize];
                    n += 1;
                }
            }
            (n > 0).then(|| (y * w + x, median(&mut near[..n])))
        })
        .collect()
}

/// The median of `near`, which is sorted by it.
fn median(near: &mut [f32]) -> f32 {
    near.sort_unstable_by(f32::total_cmp);
    let n = near.len();
    if n % 2 == 1 {
        near[n / 2]
    } else {
        0.5 * (near[n / 2 - 1] + near[n / 2])
    }
}

/// Per CFA phase, the offsets of the same-colour sites around it.
fn phase_offsets(cfa: &CfaPattern) -> Vec<Vec<(isize, isize)>> {
    let (cw, ch) = (cfa.width, cfa.height);
    (0..ch * cw)
        .map(|phase| {
            let colour = cfa.colors[phase];
            let (py, px) = ((phase / cw) as isize, (phase % cw) as isize);
            let mut list = Vec::new();
            for dy in -RADIUS..=RADIUS {
                for dx in -RADIUS..=RADIUS {
                    let (y, x) = (
                        (py + dy).rem_euclid(ch as isize),
                        (px + dx).rem_euclid(cw as isize),
                    );
                    if (dy, dx) != (0, 0) && cfa.colors[y as usize * cw + x as usize] == colour {
                        list.push((dy, dx));
                    }
                }
            }
            list
        })
        .collect()
}

/// The defective sites of `data` (`w` x `h`, as decoded) and what each should
/// be: `(index, replacement)`, in index order.
pub fn find(data: &[f32], w: usize, h: usize, cfa: &CfaPattern) -> Vec<(usize, f32)> {
    let (cw, ch) = (cfa.width, cfa.height);
    let offsets = phase_offsets(cfa);

    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let band = h.div_ceil(threads).max(1);
    let scan = |rows: std::ops::Range<usize>| {
        let mut found = Vec::new();
        let mut near = [0f32; MAX_NEIGHBOURS];
        for y in rows {
            for x in 0..w {
                let v = data[y * w + x];
                let mut n = 0;
                let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                for &(dy, dx) in &offsets[(y % ch) * cw + x % cw] {
                    let (yy, xx) = (y as isize + dy, x as isize + dx);
                    if yy < 0 || xx < 0 || yy >= h as isize || xx >= w as isize {
                        continue;
                    }
                    let s = data[yy as usize * w + xx as usize];
                    near[n] = s;
                    n += 1;
                    lo = lo.min(s);
                    hi = hi.max(s);
                }
                if n < MIN_NEIGHBOURS {
                    continue;
                }
                let margin = K * (hi - lo) + FLOOR;
                if v - hi > margin || lo - v > margin {
                    let near = &mut near[..n];
                    let median = median(near);
                    let extreme = (v >= STUCK_ON && median < STUCK_ON)
                        || (v <= DEAD && median > DEAD_NEIGHBOURS);
                    if EXTREME_ONLY && !extreme {
                        continue;
                    }
                    found.push((y * w + x, median));
                }
            }
        }
        found
    };
    std::thread::scope(|scope| {
        let scan = &scan;
        let parts: Vec<_> = (0..h)
            .step_by(band)
            .map(|y0| scope.spawn(move || scan(y0..(y0 + band).min(h))))
            .collect();
        parts.into_iter().flat_map(|p| p.join().unwrap()).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(bayer: bool) -> CfaPattern {
        if bayer {
            CfaPattern {
                width: 2,
                height: 2,
                colors: vec![0, 1, 1, 2],
            }
        } else {
            // a 6x6 X-Trans tile
            let rows = ["GBGGRG", "RGRBGB", "GBGGRG", "GRGGBG", "BGBRGR", "GRGGBG"];
            let colors = rows
                .iter()
                .flat_map(|r| {
                    r.chars().map(|c| match c {
                        'R' => 0,
                        'G' => 1,
                        _ => 2,
                    })
                })
                .collect();
            CfaPattern {
                width: 6,
                height: 6,
                colors,
            }
        }
    }

    /// A smooth scene with a little noise (a deterministic hash).
    fn scene(w: usize, h: usize) -> Vec<f32> {
        (0..w * h)
            .map(|i| {
                let (y, x) = ((i / w) as f32, (i % w) as f32);
                let noise = ((i as u32).wrapping_mul(2654435761) >> 16) as f32 / 65536.0 - 0.5;
                0.3 + 0.2 * (x / 17.0).sin() * (y / 23.0).cos() + 0.01 * noise
            })
            .collect()
    }

    #[test]
    fn finds_planted_defects_and_nothing_else() {
        for bayer in [true, false] {
            let cfa = pattern(bayer);
            let (w, h) = (96, 84);
            let clean = scene(w, h);
            assert!(
                find(&clean, w, h, &cfa).is_empty(),
                "false positives (bayer {bayer})"
            );
            let mut data = clean.clone();
            // hot, dead, one in a corner, one in the dark with a small bump
            let planted = [(40, 30, 1.0), (41, 50, 0.0), (0, 0, 1.0), (83, 95, 0.0)];
            for &(y, x, v) in &planted {
                data[y * w + x] = v;
            }
            let found = find(&data, w, h, &cfa);
            let indices: Vec<usize> = found.iter().map(|f| f.0).collect();
            let wanted: Vec<usize> = {
                let mut v: Vec<usize> = planted.iter().map(|&(y, x, _)| y * w + x).collect();
                v.sort();
                v
            };
            assert_eq!(indices, wanted, "bayer {bayer}");
            for (i, fixed) in found {
                assert!(
                    (fixed - clean[i]).abs() < 0.05,
                    "replacement {fixed} vs {}",
                    clean[i]
                );
            }
        }
    }

    #[test]
    fn replaces_picked_sites_whatever_their_value() {
        let cfa = pattern(true);
        let (w, h) = (64, 64);
        let clean = scene(w, h);
        let mut data = clean.clone();
        data[20 * w + 20] += 0.1; // not an outlier
        let found = replacements(&data, w, h, &cfa, &[(20, 20), (0, 0), (99, 99)]);
        assert_eq!(found.len(), 2); // the third is outside the frame
        assert_eq!(found[0].0, 20 * w + 20);
        assert!((found[0].1 - clean[20 * w + 20]).abs() < 0.05);
    }

    #[test]
    fn leaves_partial_outliers() {
        let cfa = pattern(true);
        let (w, h) = (64, 64);
        let mut data = scene(w, h);
        data[20 * w + 20] += 0.4; // warm, not stuck
        data[30 * w + 31] = 0.01; // low, not dead
        assert!(find(&data, w, h, &cfa).is_empty());
    }

    #[test]
    fn keeps_lines_and_edges() {
        let cfa = pattern(true);
        let (w, h) = (64, 64);
        let mut data = vec![0.05f32; w * h];
        for x in 0..w {
            data[32 * w + x] = 0.9; // a one pixel line
        }
        for y in 0..h {
            for x in 40..w {
                data[y * w + x] = 0.7; // an edge
            }
        }
        assert!(find(&data, w, h, &cfa).is_empty());
    }
}
