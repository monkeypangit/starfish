//! The colour edits: a colour mixer and colour grading. Both work on the
//! display's colour in Oklab (whose L is close to perceived lightness and
//! whose hue is stable under a change of chroma), after the tone curve and
//! saturation, and `develop.wgsl`'s `apply_color_edits` is this module's maths
//! on the GPU; `pack` is each one's uniform, and `apply` (tests only) a CPU
//! copy of the whole.
//!
//! **Colour mixer.** Eight colours (red … magenta, at fixed OkLCh hues) each
//! have a hue shift, a saturation and a brightness. A pixel's own hue sits
//! between two neighbouring colours and takes a smooth blend of their
//! settings, so the sliders have no seams, and greys (which have no hue) are
//! left alone.
//!
//! **Colour grading.** Shadows, midtones and highlights (three overlapping
//! ranges of the pixel's lightness that always add up to one) each push the
//! colour towards a target hue by an amount, in Oklab a / b, which also works
//! on greys.

/// The colours of the mixer.
pub const MIX_COLORS: usize = 8;
/// Their names.
pub const MIX_NAMES: [&str; MIX_COLORS] = [
    "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta",
];
/// Where each colour is, in OkLCh degrees, ascending (sRGB's red is 29°,
/// yellow 110°, green 142°, cyan 195°, blue 264°).
pub const MIX_HUES: [f32; MIX_COLORS] = [29.0, 62.0, 105.0, 145.0, 195.0, 262.0, 300.0, 340.0];
/// The hue a shift of ±1 turns a colour by, degrees.
pub const HUE_SHIFT_MAX: f32 = 30.0;
/// Lightness moved by a brightness of ±1.
const BRIGHTNESS_SPAN: f32 = 0.25;
/// Saturation of +1 scales chroma by 3; of -1, by 0.
const SATURATION_MAX: f32 = 2.0;
/// Below this chroma the mixer fades out: greys have no hue (the shader's
/// `HUE_FADE_CHROMA`).
#[cfg(test)]
const HUE_FADE_CHROMA: f32 = 0.02;

/// The grading ranges.
pub const RANGES: usize = 3;
pub const RANGE_NAMES: [&str; RANGES] = ["Shadows", "Midtones", "Highlights"];
/// Oklab lightness where the shadows end (their weight is 1 below the first
/// number, 0 above the second) and the highlights begin (the shader's
/// `SHADOWS` and `HIGHLIGHTS`). The midtones are what is left.
#[cfg(test)]
const SHADOWS: [f32; 2] = [0.2, 0.6];
#[cfg(test)]
const HIGHLIGHTS: [f32; 2] = [0.6, 0.9];
/// Oklab chroma added by a full-strength tint.
const TINT_STRENGTH: f32 = 0.1;

/// What the mixer sets for a colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixParam {
    /// −1 … 1: ± [`HUE_SHIFT_MAX`] degrees.
    Hue,
    /// −1 … 1: −1 greys the colour, +1 triples its chroma.
    Saturation,
    /// −1 … 1: ± a quarter of the lightness range.
    Brightness,
}

/// The mixer's settings, indexed by colour, each −1 … 1.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ColorMix {
    pub hue: [f32; MIX_COLORS],
    pub saturation: [f32; MIX_COLORS],
    pub brightness: [f32; MIX_COLORS],
}

impl ColorMix {
    pub fn get(&self, color: usize, param: MixParam) -> f32 {
        match param {
            MixParam::Hue => self.hue[color],
            MixParam::Saturation => self.saturation[color],
            MixParam::Brightness => self.brightness[color],
        }
    }

    /// Sets a setting; a colour out of range is ignored.
    pub fn set(&mut self, color: usize, param: MixParam, value: f32) {
        if color >= MIX_COLORS {
            return;
        }
        match param {
            MixParam::Hue => self.hue[color] = value,
            MixParam::Saturation => self.saturation[color] = value,
            MixParam::Brightness => self.brightness[color] = value,
        }
    }

    /// Whether the mixer moves anything.
    pub fn has_effect(&self) -> bool {
        [&self.hue, &self.saturation, &self.brightness]
            .iter()
            .any(|v| v.iter().any(|&x| x != 0.0))
    }

    /// The uniform: one `vec4` per colour, as `apply_color_edits` reads them:
    /// hue shift (radians), chroma scale − 1, lightness shift, and the
    /// colour's hue (radians).
    pub fn pack(&self) -> [[f32; 4]; MIX_COLORS] {
        std::array::from_fn(|i| {
            let s = self.saturation[i];
            let saturation = if s >= 0.0 { s * SATURATION_MAX } else { s };
            [
                (self.hue[i] * HUE_SHIFT_MAX).to_radians(),
                saturation,
                self.brightness[i] * BRIGHTNESS_SPAN,
                MIX_HUES[i].to_radians(),
            ]
        })
    }
}

/// What grading sets for a range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradeParam {
    /// The hue colours are pushed towards, OkLCh degrees, 0 … 360.
    Hue,
    /// How hard, 0 … 1 (the pushed chroma grows with its square).
    Amount,
}

/// The grading's settings, indexed by range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grading {
    pub hue: [f32; RANGES],
    pub amount: [f32; RANGES],
}

impl Default for Grading {
    fn default() -> Self {
        // The middle of the hue slider, for every range.
        Self {
            hue: [180.0; RANGES],
            amount: [0.0; RANGES],
        }
    }
}

impl Grading {
    pub fn get(&self, range: usize, param: GradeParam) -> f32 {
        match param {
            GradeParam::Hue => self.hue[range],
            GradeParam::Amount => self.amount[range],
        }
    }

    /// Sets a setting; a range out of range is ignored.
    pub fn set(&mut self, range: usize, param: GradeParam, value: f32) {
        if range >= RANGES {
            return;
        }
        match param {
            GradeParam::Hue => self.hue[range] = value,
            GradeParam::Amount => self.amount[range] = value,
        }
    }

    /// Whether grading moves anything.
    pub fn has_effect(&self) -> bool {
        self.amount.iter().any(|&a| a != 0.0)
    }

    /// The uniform: one `vec4` per range, the Oklab (a, b) offset of its
    /// tint in x and y.
    pub fn pack(&self) -> [[f32; 4]; RANGES] {
        std::array::from_fn(|i| {
            let k = self.amount[i] * self.amount[i] * TINT_STRENGTH;
            let h = self.hue[i].to_radians();
            [k * h.cos(), k * h.sin(), 0.0, 0.0]
        })
    }
}

/// The uniform's `edit_flags`: which of the two do anything.
pub fn flags(mix: &ColorMix, grading: &Grading) -> u32 {
    u32::from(mix.has_effect()) | (u32::from(grading.has_effect()) << 1)
}

/// `x` brought into 0 … 2π.
#[cfg(test)]
fn wrap_turn(x: f32) -> f32 {
    x - std::f32::consts::TAU * (x / std::f32::consts::TAU).floor()
}

#[cfg(test)]
fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The mixer's `[hue shift, chroma scale − 1, lightness shift]` at `hue`
/// radians: a smooth blend between the colours either side of it.
#[cfg(test)]
fn mix_at(packed: &[[f32; 4]; MIX_COLORS], hue: f32) -> [f32; 3] {
    let h = wrap_turn(hue);
    let k = packed.iter().filter(|c| h >= c[3]).count();
    let (a, b) = (
        packed[(k + MIX_COLORS - 1) % MIX_COLORS],
        packed[k % MIX_COLORS],
    );
    let span = wrap_turn(b[3] - a[3]);
    let s = smoothstep(0.0, 1.0, wrap_turn(h - a[3]) / span);
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * s)
}

/// The weights of the shadows, midtones and highlights at Oklab lightness
/// `l`; they add up to one.
#[cfg(test)]
fn range_weights(l: f32) -> [f32; RANGES] {
    let shadows = 1.0 - smoothstep(SHADOWS[0], SHADOWS[1], l);
    let highlights = smoothstep(HIGHLIGHTS[0], HIGHLIGHTS[1], l);
    [shadows, 1.0 - shadows - highlights, highlights]
}

/// The two edits applied to a linear sRGB colour, what `apply_color_edits`
/// does.
#[cfg(test)]
pub fn apply(mix: &ColorMix, grading: &Grading, linear: [f32; 3]) -> [f32; 3] {
    let [l, a, b] = linear_to_oklab(linear);
    let chroma = a.hypot(b);
    let mut hue = b.atan2(a);
    let (mut c, mut lightness) = (chroma, l);
    let mut offset = [0.0, 0.0];
    if mix.has_effect() {
        let fade = smoothstep(0.0, HUE_FADE_CHROMA, chroma);
        let [dh, ds, dl] = mix_at(&mix.pack(), hue);
        hue += dh * fade;
        c = (c * (1.0 + ds * fade)).max(0.0);
        lightness += dl * fade;
    }
    if grading.has_effect() {
        let weights = range_weights(l);
        for (w, g) in weights.iter().zip(grading.pack()) {
            offset[0] += w * g[0];
            offset[1] += w * g[1];
        }
    }
    oklab_to_linear([
        lightness,
        c * hue.cos() + offset[0],
        c * hue.sin() + offset[1],
    ])
}

/// Linear sRGB → Oklab (Ottosson).
#[cfg(test)]
pub fn linear_to_oklab([r, g, b]: [f32; 3]) -> [f32; 3] {
    let l = (0.412_221_47 * r + 0.536_332_55 * g + 0.051_445_995 * b)
        .max(0.0)
        .cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b)
        .max(0.0)
        .cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b)
        .max(0.0)
        .cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

/// Oklab → linear sRGB.
pub fn oklab_to_linear([l, a, b]: [f32; 3]) -> [f32; 3] {
    let l_ = l + 0.396_337_78 * a + 0.215_803_76 * b;
    let m_ = l - 0.105_561_35 * a - 0.063_854_17 * b;
    let s_ = l - 0.089_484_18 * a - 1.291_485_5 * b;
    let (l, m, s) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_93 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_4 * s,
        -0.004_196_086 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

/// An OkLCh colour (hue in degrees) as display sRGB in 0 … 1, clipped to
/// the gamut, for painting the sliders and swatches.
pub fn oklch_to_display(l: f32, c: f32, hue: f32) -> [f32; 3] {
    let h = hue.to_radians();
    oklab_to_linear([l, c * h.cos(), c * h.sin()]).map(|v| {
        let v = v.clamp(0.0, 1.0);
        if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3], tolerance: f32) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance)
    }

    /// Chroma and hue (degrees) of a linear sRGB colour.
    fn lch(linear: [f32; 3]) -> (f32, f32, f32) {
        let [l, a, b] = linear_to_oklab(linear);
        (l, a.hypot(b), b.atan2(a).to_degrees().rem_euclid(360.0))
    }

    #[test]
    fn oklab_round_trips() {
        for rgb in [
            [0.2, 0.5, 0.8],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.9, 0.9, 0.1],
        ] {
            assert!(
                close(oklab_to_linear(linear_to_oklab(rgb)), rgb, 1e-4),
                "{rgb:?}"
            );
        }
        // White is neutral at L = 1.
        let [l, a, b] = linear_to_oklab([1.0; 3]);
        assert!((l - 1.0).abs() < 1e-3 && a.abs() < 1e-3 && b.abs() < 1e-3);
    }

    #[test]
    fn the_colours_are_ascending_and_sit_on_their_names() {
        assert!(MIX_HUES.windows(2).all(|w| w[0] < w[1]) && MIX_HUES[MIX_COLORS - 1] < 360.0);
        // sRGB's primaries are near the mixer's colours of the same name.
        for (rgb, name) in [
            ([1.0, 0.0, 0.0], "Red"),
            ([1.0, 1.0, 0.0], "Yellow"),
            ([0.0, 1.0, 0.0], "Green"),
            ([0.0, 1.0, 1.0], "Aqua"),
            ([0.0, 0.0, 1.0], "Blue"),
        ] {
            let (_, _, hue) = lch(rgb);
            let at = MIX_NAMES.iter().position(|n| *n == name).unwrap();
            assert!((hue - MIX_HUES[at]).abs() < 8.0, "{name}: {hue}");
        }
    }

    #[test]
    fn nothing_set_changes_nothing_and_is_not_flagged() {
        let (mix, grading) = (ColorMix::default(), Grading::default());
        assert_eq!(flags(&mix, &grading), 0);
        for rgb in [[0.3, 0.2, 0.6], [0.3; 3], [0.0; 3], [0.9, 0.5, 0.1]] {
            assert!(close(apply(&mix, &grading, rgb), rgb, 1e-4), "{rgb:?}");
        }
        let mut mix = mix;
        mix.set(2, MixParam::Hue, 0.5);
        let mut grading = grading;
        grading.set(1, GradeParam::Amount, 0.5);
        assert_eq!(flags(&mix, &grading), 3);
        // Moving only a hue leaves the range as it is.
        grading.set(1, GradeParam::Hue, 10.0);
        grading.set(1, GradeParam::Amount, 0.0);
        assert_eq!(flags(&ColorMix::default(), &grading), 0);
    }

    #[test]
    fn a_colours_settings_reach_full_strength_at_its_hue() {
        let packed = {
            let mut mix = ColorMix::default();
            mix.set(0, MixParam::Hue, 1.0);
            mix.set(0, MixParam::Saturation, 1.0);
            mix.set(0, MixParam::Brightness, -1.0);
            mix.pack()
        };
        let at_red = mix_at(&packed, MIX_HUES[0].to_radians());
        assert!((at_red[0].to_degrees() - HUE_SHIFT_MAX).abs() < 1e-3);
        assert!((at_red[1] - SATURATION_MAX).abs() < 1e-5);
        assert!((at_red[2] + BRIGHTNESS_SPAN).abs() < 1e-5);
        // Half way to orange, a smoothstep's half; at orange, nothing.
        let mid = (MIX_HUES[0] + MIX_HUES[1]) / 2.0;
        assert!((mix_at(&packed, mid.to_radians())[1] - SATURATION_MAX / 2.0).abs() < 1e-3);
        assert!(mix_at(&packed, MIX_HUES[1].to_radians())[1].abs() < 1e-5);
        // Between magenta and red the blend wraps round zero.
        let wrap = mix_at(&packed, 5.0_f32.to_radians());
        assert!(wrap[1] > 0.0 && wrap[1] < SATURATION_MAX);
    }

    #[test]
    fn the_blend_is_continuous_all_round_the_circle() {
        let mut mix = ColorMix::default();
        for i in 0..MIX_COLORS {
            mix.set(i, MixParam::Saturation, if i % 2 == 0 { 1.0 } else { -0.5 });
            mix.set(i, MixParam::Hue, (i as f32 / 4.0) - 1.0);
        }
        let packed = mix.pack();
        let mut last = mix_at(&packed, 0.0);
        for step in 1..=3600 {
            let now = mix_at(&packed, (step as f32 * 0.1).to_radians());
            assert!(
                (0..3).all(|i| (now[i] - last[i]).abs() < 0.05),
                "jump at {}°",
                step as f32 * 0.1
            );
            last = now;
        }
    }

    #[test]
    fn desaturating_every_colour_greys_a_colour_and_leaves_greys() {
        let mut mix = ColorMix::default();
        for i in 0..MIX_COLORS {
            mix.set(i, MixParam::Saturation, -1.0);
        }
        let none = Grading::default();
        let [r, g, b] = apply(&mix, &none, [0.6, 0.2, 0.1]);
        assert!((r - g).abs() < 1e-3 && (g - b).abs() < 1e-3);
        let grey = [0.3; 3];
        assert!(close(apply(&mix, &none, grey), grey, 1e-4));
    }

    #[test]
    fn a_hue_shift_turns_the_hue_and_keeps_lightness_and_chroma() {
        let mut mix = ColorMix::default();
        mix.set(0, MixParam::Hue, 1.0);
        let red = oklab_to_linear([
            0.6,
            0.15 * 29f32.to_radians().cos(),
            0.15 * 29f32.to_radians().sin(),
        ]);
        let (l0, c0, h0) = lch(red);
        let (l1, c1, h1) = lch(apply(&mix, &Grading::default(), red));
        assert!((h1 - h0 - HUE_SHIFT_MAX).abs() < 0.5, "{h0} → {h1}");
        assert!((l1 - l0).abs() < 1e-3 && (c1 - c0).abs() < 1e-3);
    }

    #[test]
    fn a_colour_only_moves_its_own_hues() {
        let mut mix = ColorMix::default();
        let blue_at = MIX_NAMES.iter().position(|n| *n == "Blue").unwrap();
        mix.set(blue_at, MixParam::Brightness, 1.0);
        let none = Grading::default();
        let green = oklab_to_linear([
            0.7,
            0.1 * 145f32.to_radians().cos(),
            0.1 * 145f32.to_radians().sin(),
        ]);
        assert!(close(apply(&mix, &none, green), green, 1e-4));
        let blue = oklab_to_linear([
            0.5,
            0.1 * 262f32.to_radians().cos(),
            0.1 * 262f32.to_radians().sin(),
        ]);
        assert!(lch(apply(&mix, &none, blue)).0 > 0.5 + 0.2);
    }

    #[test]
    fn the_ranges_always_add_up_to_one() {
        for i in 0..=200 {
            let w = range_weights(i as f32 / 200.0);
            assert!(w.iter().all(|&x| (0.0..=1.0).contains(&x)));
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        assert_eq!(range_weights(0.0), [1.0, 0.0, 0.0]);
        assert_eq!(range_weights(SHADOWS[1]), [0.0, 1.0, 0.0]);
        assert_eq!(range_weights(1.0), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_range_tints_its_own_lightness_and_leaves_the_others() {
        let mut grading = Grading::default();
        grading.set(0, GradeParam::Hue, 264.0); // blue shadows
        grading.set(0, GradeParam::Amount, 1.0);
        let none = ColorMix::default();
        let dark = [0.0005; 3];
        let mid = oklab_to_linear([SHADOWS[1], 0.0, 0.0]);
        let bright = [0.8, 0.8, 0.8];
        let tinted = apply(&none, &grading, dark);
        assert!(!close(tinted, dark, 1e-4) && tinted[2] > tinted[0]);
        assert!(close(apply(&none, &grading, mid), mid, 1e-4));
        assert!(close(apply(&none, &grading, bright), bright, 1e-4));
    }

    #[test]
    fn grading_pushes_towards_its_hue_by_the_square_of_the_amount() {
        let mut grading = Grading::default();
        grading.set(1, GradeParam::Hue, 0.0);
        grading.set(1, GradeParam::Amount, 0.5);
        let [a, b, ..] = grading.pack()[1];
        assert!((a - 0.25 * TINT_STRENGTH).abs() < 1e-6 && b.abs() < 1e-6);
    }
}
