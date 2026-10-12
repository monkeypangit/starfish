//! Tone mapping after AgX (Troy Sobotka's, as in Blender and "minimal AgX"):
//! its inset/outset matrices around a sigmoid, but with a fixed white point.
//!
//! linear sRGB → black point → exposure → shadows/midtones/highlights levels → inset matrix →
//! (x / WHITE)^(1/2.4) → sigmoid → saturation → colour grading → ^2.4 (back to linear) → outset matrix → display.
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
/// Toe and shoulder sharpness of the sigmoid.
const POWER: f32 = 2.0;
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
    pub power: f32,
    pub toe: f32,
    pub shoulder: f32,
}

impl Curve {
    pub fn new(adjustments: &Adjustments) -> Self {
        let pivot = encode(adjustments.midpoint);
        let slope = adjustments.contrast;
        // g(1) = 1: c / (1 ± (c / L)^p)^(1/p) = 1 gives ±(c / L)^p = c^p - 1.
        let excess = slope.powf(POWER) - 1.0;
        let limit = if excess.abs() < 1e-6 {
            f32::MAX
        } else {
            slope / excess.abs().powf(1.0 / POWER) * excess.signum()
        };
        Self {
            pivot_x: pivot,
            pivot_y: pivot,
            slope,
            power: POWER,
            // A side with no span (midpoint at 0 or 100%) only ever sees
            // z = 0; the floor keeps |z| / |limit| from being 0 / 0.
            toe: limit * pivot.max(1e-6),
            shoulder: limit * (1.0 - pivot).max(1e-6),
        }
    }

    pub fn apply(&self, x: f32) -> f32 {
        let z = self.slope * (x - self.pivot_x);
        let limit = if z < 0.0 { self.toe } else { self.shoulder };
        // The floor keeps the shader's pow() away from 0, where it is undefined.
        let q = (z.abs() / limit.abs()).max(1e-20).powf(self.power);
        let bend = (1.0 + limit.signum() * q).max(1e-20);
        (self.pivot_y + z / bend.powf(1.0 / self.power)).clamp(0.0, 1.0)
    }
}

/// Linear value → display-encoded, 0 at black and 1 at [`WHITE`] and above.
pub fn encode(linear: f32) -> f32 {
    (linear / WHITE).clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA)
}

/// Display-encoded → linear, the inverse of [`encode`] over 0..1.
pub fn decode(encoded: f32) -> f32 {
    encoded.clamp(0.0, 1.0).powf(DISPLAY_GAMMA) * WHITE
}
