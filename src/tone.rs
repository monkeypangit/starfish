//! The tone curve: a contrast and midpoint sigmoid in display-encoded space,
//! after the sigmoid of Troy Sobotka's AgX (as in Blender and "minimal AgX"),
//! without its inset/outset matrices and with a fixed white point.
//!
//! linear sRGB → black point → exposure →
//! (x / WHITE)^(1/2.4) → sigmoid → saturation → ^2.4 (back to linear) → colour mixer and grading (color_edit.rs) → display.
//!
//! Blender's AgX runs the curve over log2(x / 0.18) from -10 to +6.5 stops,
//! so values far above the sensor's clipping point still roll off. Here,
//! after exposure, [`WHITE`] and everything above it is white, black is 0,
//! and the curve only reshapes the tones in between. A log-domain sigmoid
//! can't do that well: with white only 2.5 stops above mid grey it must be
//! steeper at mid grey than no curve at all and crushes the shadows. In the
//! display-encoded domain no curve is the straight line, so contrast 1 is
//! the identity and more bends it into an S.
//!
//! `develop.wgsl` runs these steps on the GPU, for the preview, the export
//! and the histogram alike, with the numbers from [`Curve`]; this module
//! keeps the curve's maths and the encoding the controls work in.

use crate::preview::Adjustments;

/// Linear value, after exposure, shown as white. 1.0 is the sensor's
/// clipping point (before white balance).
pub const WHITE: f32 = 1.0;
/// Contrast is the curve's slope where it crosses the diagonal: 1 is no
/// curve, above 1 an S, below 1 an inverted S that flattens the midtones.
pub const DEFAULT_CONTRAST: f32 = 1.4;
/// The display-encoded tone whose lift sets what "contrast" means (see
/// [`Curve::new`]): a well-lit highlight.
const HIGHLIGHT: f32 = 0.75;
/// Default sharpness of the sigmoid's toe and shoulder knees: low is a soft
/// roll-off, high a hard turn out of the straight middle.
pub const POWER: f32 = 2.0;
/// The knee sliders' range.
pub const POWER_RANGE: std::ops::RangeInclusive<f32> = 0.5..=6.0;
/// The curve's output is display-encoded with this power.
const DISPLAY_GAMMA: f32 = 2.4;

/// The tone curve over the display-encoded value `x` in 0..1. It crosses
/// the diagonal at the midpoint m, and its two sides are one shape `g` (with
/// `g(0) = 0`, `g(1) = 1`, slope `contrast` at 0) scaled to their spans and
/// turned about the crossover:
///
/// `y = m + (1 - m) g((x - m) / (1 - m))` above m,
/// `y = m - m g((m - x) / m)` below,
/// `g(u) = c u / (1 ± (c u / L)^power)^(1/power)`,
///
/// with L solved so g(1) = 1: + bends an S (c > 1), − its inverse (c < 1).
/// Folding the spans in, each side is `pivot + z / (1 ± (|z| / |limit|)^p)^(1/p)`
/// with `z = slope * (x - pivot)` and `limit = ±L * span`, a negative limit
/// meaning −; that is what [`Curve::apply`] and `develop.wgsl` evaluate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Curve {
    pub pivot_x: f32,
    pub pivot_y: f32,
    pub slope: f32,
    /// Sharpness of the knee below the pivot (the toe) and above it.
    pub toe_power: f32,
    pub shoulder_power: f32,
    pub toe: f32,
    pub shoulder: f32,
}

impl Curve {
    /// The curve for the controls. `Adjustments::contrast` is the highlight
    /// strength, not the slope at the pivot: the curve's slope is solved so
    /// that a reference highlight tone is lifted as much as contrast `contrast`
    /// lifts it with the midpoint at mid grey. Moving the midpoint (the shade
    /// level) then leaves the highlights as they were; without it the lit span
    /// above the pivot shrinks or grows and the highlights flatten or open up.
    pub fn new(adjustments: &Adjustments) -> Self {
        let pivot = encode(adjustments.midpoint);
        let powers = (adjustments.toe_power, adjustments.shoulder_power);
        let slope = Self::solve_slope(pivot, adjustments.contrast, powers);
        Self::with_slope(pivot, slope, powers)
    }

    /// The slope at `pivot` that lifts the reference highlight as `contrast`
    /// does at the default midpoint (where the two are the same).
    fn solve_slope(pivot: f32, contrast: f32, powers: (f32, f32)) -> f32 {
        let home = encode(crate::preview::MID_GRAY);
        // The reference sits above the pivot: at HIGHLIGHT, or halfway up the
        // lit span when the pivot is further up.
        let reference = reference(pivot);
        if (contrast - 1.0).abs() < 1e-6 || reference <= pivot + 1e-3 || pivot <= 1e-3 {
            return contrast;
        }
        let target = Self::with_slope(home, contrast, powers).apply(reference);
        // The lift rises with the slope, so bisect (in log); out of reach
        // clamps to the ends.
        let (mut lo, mut hi) = (0.05_f32.ln(), 64.0_f32.ln());
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if Self::with_slope(pivot, mid.exp(), powers).apply(reference) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (0.5 * (lo + hi)).exp()
    }

    /// The contrast (within `range`) that puts the curve's output at the
    /// contrast handle (`reference(pivot)`) at `target` (display-encoded),
    /// for the other controls as `adjustments` has them. The output rises
    /// with contrast, so it bisects; a target out of reach gives an end.
    pub fn contrast_for(
        adjustments: &Adjustments,
        target: f32,
        range: std::ops::RangeInclusive<f32>,
    ) -> f32 {
        let at = |contrast: f32| {
            let adjustments = Adjustments { contrast, ..*adjustments };
            let curve = Self::new(&adjustments);
            curve.apply(reference(curve.pivot_x))
        };
        let (mut lo, mut hi) = (*range.start(), *range.end());
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if at(mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    fn with_slope(pivot: f32, slope: f32, (toe_power, shoulder_power): (f32, f32)) -> Self {
        // g(1) = 1: c / (1 ± (c / L)^p)^(1/p) = 1 gives ±(c / L)^p = c^p - 1,
        // for each side's own p.
        let limit = |power: f32| {
            let excess = slope.powf(power) - 1.0;
            if excess.abs() < 1e-6 {
                f32::MAX
            } else {
                slope / excess.abs().powf(1.0 / power) * excess.signum()
            }
        };
        Self {
            pivot_x: pivot,
            pivot_y: pivot,
            slope,
            toe_power,
            shoulder_power,
            // A side with no span (midpoint at 0 or 100%) only ever sees
            // z = 0; the floor keeps |z| / |limit| from being 0 / 0.
            toe: limit(toe_power) * pivot.max(1e-6),
            shoulder: limit(shoulder_power) * (1.0 - pivot).max(1e-6),
        }
    }

    pub fn apply(&self, x: f32) -> f32 {
        let z = self.slope * (x - self.pivot_x);
        let (limit, power) = if z < 0.0 {
            (self.toe, self.toe_power)
        } else {
            (self.shoulder, self.shoulder_power)
        };
        // The floor keeps the shader's pow() away from 0, where it is undefined.
        let q = (z.abs() / limit.abs()).max(1e-20).powf(power);
        let bend = (1.0 + limit.signum() * q).max(1e-20);
        (self.pivot_y + z / bend.powf(1.0 / power)).clamp(0.0, 1.0)
    }
}

/// The display-encoded tone the contrast is measured at (see [`Curve::new`]),
/// above the pivot: where the contrast handle sits on the curve.
pub fn reference(pivot: f32) -> f32 {
    HIGHLIGHT.max(pivot + (1.0 - pivot) / 2.0).min(0.995)
}

/// Linear value → display-encoded, 0 at black and 1 at [`WHITE`] and above.
pub fn encode(linear: f32) -> f32 {
    (linear / WHITE).clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA)
}

/// Display-encoded → linear, the inverse of [`encode`] over 0..1.
pub fn decode(encoded: f32) -> f32 {
    encoded.clamp(0.0, 1.0).powf(DISPLAY_GAMMA) * WHITE
}

/// The midpoint slider's left end (linear); the slider is logarithmic from
/// here to 1, so every stop of shadow gets the same travel.
pub const MIDPOINT_MIN: f32 = 0.0001;

/// Slider position (0..1) of a linear midpoint.
pub fn midpoint_position(midpoint: f32) -> f32 {
    (midpoint.max(MIDPOINT_MIN).ln() / MIDPOINT_MIN.ln() * -1.0 + 1.0).clamp(0.0, 1.0)
}

/// The linear midpoint at a slider position, the inverse of [`midpoint_position`].
pub fn midpoint_from_position(position: f32) -> f32 {
    MIDPOINT_MIN.powf(1.0 - position.clamp(0.0, 1.0))
}

#[cfg(test)]
mod contrast_tests {
    use super::*;

    fn adj(midpoint: f32, contrast: f32) -> Adjustments {
        Adjustments {
            midpoint,
            contrast,
            ..Adjustments::default()
        }
    }

    #[test]
    fn highlights_stay_put_as_the_midpoint_moves() {
        let home = Curve::new(&adj(crate::preview::MID_GRAY, 1.6));
        for midpoint in [0.05, 0.1, 0.3, 0.45] {
            let curve = Curve::new(&adj(midpoint, 1.6));
            for x in [0.85, 0.9, 0.95] {
                let (a, b) = (home.apply(x), curve.apply(x));
                println!("m {midpoint} x {x}: {a:.4} {b:.4} slope {:.3}", curve.slope);
                // The slope is solved so one reference highlight tone matches
                // the home curve exactly (`Curve::solve_slope`); the other
                // highlight tones drift a little (0.0065 at most here, under
                // 2/255), which is the model's limit, not solver precision.
                assert!((a - b).abs() < 0.01, "m {midpoint} x {x}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn knees_are_independent_and_keep_the_ends() {
        let base = Curve::new(&adj(0.18, 2.0));
        let soft_toe = Curve::new(&Adjustments {
            toe_power: 0.8,
            ..adj(0.18, 2.0)
        });
        let hard_shoulder = Curve::new(&Adjustments {
            shoulder_power: 5.0,
            ..adj(0.18, 2.0)
        });
        for c in [soft_toe, hard_shoulder] {
            assert!(c.apply(0.0).abs() < 1e-5 && (c.apply(1.0) - 1.0).abs() < 1e-5);
        }
        let (hi, lo) = (0.9, 0.1);
        // Above the pivot only the shoulder power matters, below only the toe's.
        assert!((soft_toe.apply(hi) - base.apply(hi)).abs() < 1e-4);
        assert!((hard_shoulder.apply(lo) - base.apply(lo)).abs() < 1e-4);
        assert!(soft_toe.apply(lo) != base.apply(lo) && hard_shoulder.apply(hi) != base.apply(hi));
    }

    #[test]
    fn midpoint_slider_round_trips() {
        for m in [0.005, 0.02, 0.18, 0.5, 1.0] {
            assert!((midpoint_from_position(midpoint_position(m)) - m).abs() < 1e-5 * m.max(1.0));
        }
    }

    #[test]
    fn default_midpoint_and_no_contrast_are_unchanged() {
        assert!((Curve::new(&adj(crate::preview::MID_GRAY, 1.6)).slope - 1.6).abs() < 1e-3);
        assert_eq!(Curve::new(&adj(0.05, 1.0)).slope, 1.0);
    }
}

#[cfg(test)]
mod handle_tests {
    use super::*;

    #[test]
    fn contrast_for_inverts_the_handle() {
        for midpoint in [0.02, 0.18, 0.5] {
            for contrast in [1.0, 1.4, 2.5, 6.0] {
                let adjustments = Adjustments { midpoint, contrast, ..Adjustments::default() };
                let curve = Curve::new(&adjustments);
                let y = curve.apply(reference(curve.pivot_x));
                let found = Curve::contrast_for(&adjustments, y, 1.0..=10.0);
                assert!((found - contrast).abs() < 1e-2, "{midpoint} {contrast}: {found}");
            }
        }
    }
}
