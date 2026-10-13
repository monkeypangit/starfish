//! Fitting a raw frame to a network: the CFA phase, and the padding.
//!
//! Each network is built for one CFA phase (`Model::cfa`) and sizes that are a
//! multiple of its period, and pads circularly. So the frame is padded: on
//! every side by at least the network's reach (`Model::halo`), so no frame
//! pixel sees the opposite edge, and by up to one period more to land on the
//! model's phase and a whole number of periods. The padding mirrors the frame
//! across its edges in whole periods, which keeps every padded site's CFA
//! colour right. `Padding::source` is what `RawInput::rows` / `cols` take.

/// One axis of the padded mosaic.
pub struct Padding {
    /// Padded pixels before the frame.
    pub before: usize,
    /// The frame pixel each padded pixel is copied from.
    pub source: Vec<u32>,
}

impl Padding {
    /// Pads `len` px, whose CFA starts `phase` px into the model's, by at
    /// least `margin` on both sides: onto the model's phase, to a whole number
    /// of periods.
    pub fn new(len: usize, phase: usize, margin: usize, period: usize) -> Self {
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

/// The (row, column) offset that puts a frame's CFA (`colors`, `cw` x `ch`,
/// row-major, 0 = R, 1 = G, 2 = B) onto the model's `cfa`.
pub fn phase(
    colors: &[u32],
    (cw, ch): (usize, usize),
    cfa: &[u32],
    period: usize,
) -> Result<(usize, usize), String> {
    if period % cw != 0 || period % ch != 0 {
        return Err(format!("A {cw}×{ch} CFA does not repeat every {period} px"));
    }
    let color = |y: usize, x: usize| colors[(y % ch) * cw + x % cw];
    (0..period * period)
        .map(|i| (i / period, i % period))
        .find(|&(dy, dx)| {
            (0..period).all(|y| (0..period).all(|x| color(dy + y, dx + x) == cfa[y * period + x]))
        })
        .ok_or_else(|| "The sensor's CFA does not match the model's in any phase".into())
}

/// The h x w frame cropped to the model's phase (`dy`, `dx`) and whole
/// periods, with its size: what `noise::estimate_noise` takes.
pub fn phased(
    data: &[f32],
    h: usize,
    w: usize,
    (dy, dx): (usize, usize),
    period: usize,
) -> (Vec<f32>, usize, usize) {
    let height = (h - dy) / period * period;
    let width = (w - dx) / period * period;
    let mut out = Vec::with_capacity(width * height);
    for y in dy..dy + height {
        let start = y * w + dx;
        out.extend_from_slice(&data[start..start + width]);
    }
    (out, height, width)
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
