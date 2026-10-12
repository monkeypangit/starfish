//! Lens corrections from the profile a camera writes into its raw files.
//!
//! rawler does not decode these, so the formats are read here. None of them
//! is documented; the meaning of the numbers was worked out by fitting them
//! against each camera's own corrected JPEG preview. A format only has to
//! produce a [`LensCorrection`], which the rest of the program (the table
//! below, the GPU pass, the checkboxes) treats alike. Radii are normalised
//! to the half-diagonal of the image.
//!
//! **Fujifilm RAF** (`fuji`): the file's second TIFF structure (pointer at
//! byte 100) and its Fuji sub-IFD (tag 0xf000) hold, for the lens, focal
//! length and aperture of the shot, signed rationals laid out as
//! `[scale, knots.., values..]`:
//!
//! - 0xf00b distortion, percent: `r_distorted = r · (1 + D(r)/100)`
//! - 0xf010 vignetting, the brightness percent left at that radius
//! - 0xf00f lateral chromatic aberration: the radial scale of the red and
//!   the blue channel against green, `[scale, knots, red.., blue..]`
//!
//! The half-diagonal is `scale × knot count` pixels. Distortion matched the
//! camera's JPEG to NCC 0.86 on an XF16mm and 0.95 on an XF18-135mm; the CA
//! sign rests on weaker evidence.
//!
//! **Canon CR2 and CR3** (`canon`): tag 0x4015 of the maker note, in the CR3
//! found in its timed-metadata track, see `canon.rs` for the layout. Radii are
//! in 6 µm units of sensor, `pitch / 6000` per pixel, at tables of 6 to 22
//! knots. Vignetting is a gain of `8191 / value`: it fits the JPEG of an M50
//! to 3% at every radius and the RF-S 55-210 and RF 24-50 at f/22 as well.
//! Distortion (CR3 only; the EF lenses of a CR2 have none) is the source
//! radius over the output radius at the output radius: with no fitted
//! number it matches the profile measured in the JPEG of an RF 24-50 at 35
//! and 50mm (rms 0.1 to 0.2%) and an RF-S 18-45 at 18mm (0.26%). The camera
//! applies the table as is, with no zoom, which crops in more than needed
//! (2 to 4% at the edge); the zoom here is the one of Fuji and Sony, so the
//! framing is wider than Canon's JPEG. Word 5 and 7 of tag 0x4016 say whether the
//! camera applied vignetting and distortion to its JPEG: the numbers are
//! only verified where it did (the table of a lens it leaves alone, such as
//! the RF 85mm macro, is read the same way but nothing confirms it), and
//! the others start unchecked. Two more tables near 1.0, which are not
//! chromatic aberration (a radial scale that large would show as fringes),
//! are left unread, so Canon has no CA correction.
//!
//! **Sony ARW** (`sony`): the raw SubIFD (unencrypted, unlike the private
//! block) holds tags 0x7031 / 0x7034 / 0x7036, whether the camera applied
//! vignetting / CA / distortion to its JPEG, and tables 0x7032 / 0x7035 /
//! 0x7037 as `i16` `[n, n values.., padding]`, `n` 11 (A6700) or 16 (A7 III),
//! at radii `i/(n-1)` of the half-diagonal. Distortion is `value / 16384` of
//! the radius ratio (negative = barrel), indexed by the output radius, zoomed
//! as for Fuji. It matched the JPEG on the A7 III with the 55mm f/1.8 at NCC
//! 0.95 to 0.98, and within 0.01 of the best fit on most A6700 files, but the
//! amplitude of the A6700 tables is only right to about 8% (best-fit units
//! 15 000 to 17 700, differing by lens): the strong 15mm barrel reaches NCC
//! 0.76 against 0.97 for its best fit.
//!
//! The vignetting table is a gain of `2^(value / 8192)` at the radius. The
//! A6700 JPEGs do not show it (their flag is on but the corners are not
//! brighter), so no measurement confirmed the unit; it was judged by eye on
//! corrected renders, where the corners come out flat, and is treated as
//! verified. The CA table (`[2n, n red.., n blue..]`) is read as a radius
//! scale offset of `value / 2^20` for red and blue. **That unit is a guess**
//! (a per-channel CA fit against the JPEGs was inconsistent), so the profile
//! marks CA [`LensCorrection::unverified`] and the program leaves it unchecked.
//!
//! **Panasonic RW2** (`panasonic`): IFD0 tag 0x119 is 32 bytes, sixteen
//! little-endian `i16` words. Only distortion is known. With `p = word/32768`:
//!
//! - word 12 is the half-diagonal in pixels (3606 on a 6000×4000 frame)
//! - word 5 is the zoom, word 8 / 4 / 11 the coefficients of the distorted
//!   radius `s`: `r_corrected = s · (1 + p8·s² + p4·s⁴ + p11·s⁶)`
//! - words 2, 3, 6, 7, 9, 10, 13 are unexplained (13 = 9 + 500 always);
//!   no further radial term, centre offset or aperture fit used them.
//!
//! Nothing is fitted: these readings reach the correlation a free fit does
//! on four DC-S5M2 shots (0.83 to 0.97), and every other assignment of the
//! words was at least 0.028 worse. All four were 25 to 98 mm, so lenses and
//! cameras beyond that are untested.

mod canon;

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Highest normalised radius the lookup table covers; the corner is 1.
pub const LUT_RHO_MAX: f32 = 1.5;
/// Entries in [`LensCorrection::lut`].
pub const LUT_SIZE: usize = 257;
/// Sony distortion words per unit of radius ratio.
const SONY_UNIT: f64 = 16384.0;
/// Sony vignetting words per stop of gain. Chosen from plausible stops for the
/// lenses (1.4 to 1.6 for the 55mm f/1.8 and the 12mm at f/8) and the weak raw
/// falloff trend, then confirmed by eye: the corrected corners come out flat.
const SONY_VIGNETTING_UNIT: f64 = 8192.0;
/// GUESS: Sony CA words per unit of radius, as a scale of the red / blue
/// sample radius against green (up to 0.2%: 3 to 6 pixels at the corner).
const SONY_CA_UNIT: f64 = 1_048_576.0;
/// Vignetting gains beyond this are taken for bad data.
const MAX_GAIN: f32 = 4.0;

/// One kind of correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Distortion,
    Vignetting,
    Chromatic,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Distortion, Kind::Vignetting, Kind::Chromatic];

    pub fn label(self) -> &'static str {
        match self {
            Kind::Distortion => "Distortion",
            Kind::Vignetting => "Vignetting",
            Kind::Chromatic => "Chromatic aberration",
        }
    }
}

/// Which corrections are on (or, for [`LensCorrection::available`], which
/// the profile has).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Corrections {
    pub distortion: bool,
    pub vignetting: bool,
    pub chromatic: bool,
}

impl Corrections {
    pub const ALL: Self = Self {
        distortion: true,
        vignetting: true,
        chromatic: true,
    };

    pub const NONE: Self = Self {
        distortion: false,
        vignetting: false,
        chromatic: false,
    };

    /// What the demosaic stage applies (`LensPass`); distortion and
    /// vignetting are applied where the image is drawn (`develop.wgsl`).
    pub const CHROMATIC: Self = Self {
        chromatic: true,
        ..Self::NONE
    };

    pub fn get(self, kind: Kind) -> bool {
        match kind {
            Kind::Distortion => self.distortion,
            Kind::Vignetting => self.vignetting,
            Kind::Chromatic => self.chromatic,
        }
    }

    pub fn set(&mut self, kind: Kind, on: bool) {
        match kind {
            Kind::Distortion => self.distortion = on,
            Kind::Vignetting => self.vignetting = on,
            Kind::Chromatic => self.chromatic = on,
        }
    }

    /// The corrections in both.
    pub fn and(self, other: Self) -> Self {
        Self {
            distortion: self.distortion && other.distortion,
            vignetting: self.vignetting && other.vignetting,
            chromatic: self.chromatic && other.chromatic,
        }
    }

    /// The corrections in `self` and not in `other`.
    pub fn without(self, other: Self) -> Self {
        Self {
            distortion: self.distortion && !other.distortion,
            vignetting: self.vignetting && !other.vignetting,
            chromatic: self.chromatic && !other.chromatic,
        }
    }

    pub fn any(self) -> bool {
        self.distortion || self.vignetting || self.chromatic
    }
}

/// Piecewise linear function of the normalised radius, constant past its
/// last knot.
#[derive(Debug, Clone, PartialEq)]
struct Curve {
    knots: Vec<f32>,
    values: Vec<f32>,
}

impl Curve {
    /// The curve through `(knots, values)`, anchored at `(0, origin)` when
    /// the first knot is not at 0. `None` if the data is unusable.
    fn new(knots: &[f64], values: &[f64], origin: f32) -> Option<Self> {
        let valid = !knots.is_empty()
            && knots.len() == values.len()
            && knots.iter().chain(values).all(|v| v.is_finite())
            && knots.windows(2).all(|w| w[0] < w[1])
            && knots[0] >= 0.0;
        if !valid {
            return None;
        }
        let mut curve = Self {
            knots: knots.iter().map(|&k| k as f32).collect(),
            values: values.iter().map(|&v| v as f32).collect(),
        };
        if curve.knots[0] > 0.0 {
            curve.knots.insert(0, 0.0);
            curve.values.insert(0, origin);
        }
        Some(curve)
    }

    /// The largest value on `lo..=hi`: at an end or a knot.
    fn max_over(&self, lo: f32, hi: f32) -> f32 {
        self.knots
            .iter()
            .filter(|&&k| k > lo && k < hi)
            .map(|&k| self.at(k))
            .chain([self.at(lo), self.at(hi)])
            .fold(f32::MIN, f32::max)
    }

    fn at(&self, rho: f32) -> f32 {
        let i = self.knots.partition_point(|&k| k <= rho);
        match i {
            0 => self.values[0],
            i if i == self.knots.len() => self.values[i - 1],
            i => {
                let t = (rho - self.knots[i - 1]) / (self.knots[i] - self.knots[i - 1]);
                self.values[i - 1] + t * (self.values[i] - self.values[i - 1])
            }
        }
    }
}

/// How a profile maps the corrected image onto the source.
#[derive(Debug, Clone, PartialEq)]
enum Distortion {
    /// Fujifilm and Sony: percent, `r_source = r_out · (1 + D(r_out)/100)`.
    /// The zoom is the largest that keeps every output pixel inside the
    /// source, `1 / (1 + max D/100)` over the frame's edge, from the middle
    /// of the short edge (barrel) out to the corners (pincushion): what the
    /// cameras' own JPEGs do.
    Percent(Curve),
    /// Canon: the source radius over the output radius at the output radius,
    /// zoomed like [`Distortion::Percent`], by `1 / max ratio` over the frame
    /// edge. Canon's own JPEGs apply the table as is, which crops in more than
    /// the correction needs (2 to 4% at the edge of an RF 24-50 or RF-S 18-45).
    Ratio(Curve),
    /// Panasonic: the corrected radius is a polynomial of the source radius,
    /// `r_out = s · (1 + c0·s² + c1·s⁴ + c2·s⁶)`, and the source is sampled
    /// at `zoom · s`.
    Polynomial { zoom: f32, coefficients: [f32; 3] },
}

impl Distortion {
    /// Source radius over output radius at output radius `rho`, for an image
    /// whose short edge midpoint is at `short_edge`.
    fn ratio(&self, rho: f32, short_edge: f32) -> f32 {
        match self {
            Distortion::Percent(curve) => {
                let zoom = 1.0 / (1.0 + curve.max_over(short_edge, 1.0) / 100.0);
                zoom * (1.0 + curve.at(rho) / 100.0)
            }
            Distortion::Ratio(curve) => curve.at(rho) / curve.max_over(short_edge, 1.0),
            Distortion::Polynomial { zoom, coefficients } => {
                if rho < 1e-6 {
                    return *zoom;
                }
                let [a, b, c] = *coefficients;
                let forward = |s: f32| s * (1.0 + s * s * (a + s * s * (b + s * s * c)));
                // `forward` is increasing (checked on construction), so bisect.
                let (mut low, mut high) = (0.0, 2.5 * rho.max(1.0));
                for _ in 0..40 {
                    let mid = (low + high) / 2.0;
                    if forward(mid) < rho {
                        low = mid;
                    } else {
                        high = mid;
                    }
                }
                zoom * (low + high) / 2.0 / rho
            }
        }
    }

    /// Whether it changes the image at all.
    fn does_something(&self) -> bool {
        match self {
            Distortion::Percent(curve) => curve.values.iter().any(|&d| d != 0.0),
            Distortion::Ratio(curve) => curve.values.iter().any(|&r| (r - 1.0).abs() > 1e-4),
            Distortion::Polynomial { zoom, coefficients } => {
                *zoom != 1.0 || coefficients.iter().any(|&c| c != 0.0)
            }
        }
    }
}

/// The correction profile embedded in one raw file.
#[derive(Debug, Clone, PartialEq)]
pub struct LensCorrection {
    /// Half-diagonal of the image in pixels, the unit radius, if the file
    /// says; else the frame's.
    half_diagonal: Option<f32>,
    distortion: Option<Distortion>,
    /// Brightness percent.
    vignetting: Option<Curve>,
    /// Radial scale offsets of red and blue against green.
    chromatic: Option<(Curve, Curve)>,
    /// Corrections whose numbers are a guess (see the module docs), which the
    /// program leaves off until they are confirmed.
    unverified: Corrections,
}

impl LensCorrection {
    /// The correction in `path`, if it is a raw file of a supported maker
    /// that carries one.
    pub fn read(path: &Path) -> Option<Self> {
        let mut magic = [0; 8];
        File::open(path).ok()?.read_exact(&mut magic).ok()?;
        match &magic {
            b"FUJIFILM" => Self::fuji(path),
            [b'I', b'I', b'U', 0, ..] => Self::panasonic(path),
            [b'I', b'I', b'*', 0, ..] => Self::sony(path).or_else(|| Self::canon(path)),
            [_, _, _, _, b'f', b't', b'y', b'p'] => Self::canon(path),
            _ => None,
        }
    }

    /// The correction in a Fujifilm RAF.
    fn fuji(path: &Path) -> Option<Self> {
        let mut file = File::open(path).ok()?;
        let mut magic = [0; 8];
        file.read_exact(&mut magic).ok()?;
        if &magic != b"FUJIFILM" {
            return None;
        }
        let mut pointer = [0; 4];
        file.seek(SeekFrom::Start(100)).ok()?;
        file.read_exact(&mut pointer).ok()?;
        let base = u32::from_be_bytes(pointer) as u64;

        let tiff = Tiff { base };
        let header = tiff.read(&mut file, 0, 8)?;
        if &header[..4] != b"II*\0" {
            return None;
        }
        let root = tiff.ifd(&mut file, u32_le(&header[4..])? as u64)?;
        let fuji_ifd = root.iter().find(|e| e.tag == 0xf000)?;
        let ifd = tiff.ifd(&mut file, u32_le(&fuji_ifd.value)? as u64)?;

        let mut table = |tag| {
            let entry = ifd.iter().find(|e| e.tag == tag)?;
            tiff.rationals(&mut file, entry)
        };
        let distortion = table(0xf00b);
        let vignetting = table(0xf010);
        let chromatic = table(0xf00f);

        // Distortion and vignetting: [scale, knots (n), values (n)].
        let split = |t: &[f64]| -> Option<(usize, f64, Vec<f64>, Vec<f64>)> {
            let n = t.len().checked_sub(1)? / 2;
            (n > 0 && t.len() == 1 + 2 * n)
                .then(|| (n, t[0], t[1..=n].to_vec(), t[n + 1..].to_vec()))
        };
        let distortion = distortion.as_deref().and_then(split);
        let vignetting = vignetting.as_deref().and_then(split);

        // The knot count times the scale is the half-diagonal in pixels.
        let half_diagonal = distortion
            .iter()
            .chain(&vignetting)
            .map(|(n, scale, ..)| (*scale * *n as f64) as f32)
            .find(|d| d.is_finite() && *d > 0.0);
        half_diagonal?;

        let distortion = distortion
            .and_then(|(_, _, k, v)| Curve::new(&k, &v, 0.0))
            .map(Distortion::Percent);
        let vignetting = vignetting.and_then(|(_, _, k, v)| Curve::new(&k, &v, 100.0));
        // CA: [scale, knots (n), red (n), blue (n)], maybe one more value.
        let chromatic = chromatic.and_then(|t| {
            let n = t.len().checked_sub(1)? / 3;
            if n == 0 {
                return None;
            }
            let knots = &t[1..=n];
            Some((
                Curve::new(knots, &t[n + 1..=2 * n], 0.0)?,
                Curve::new(knots, &t[2 * n + 1..=3 * n], 0.0)?,
            ))
        });

        // Tables that do nothing are not worth a pass over the image.
        let distortion = distortion.filter(Distortion::does_something);
        let vignetting = vignetting.filter(|c| c.values.iter().any(|&v| v < 100.0));
        let chromatic = chromatic.filter(|(r, b)| r.values.iter().chain(&b.values).any(|&c| c != 0.0));
        if distortion.is_none() && vignetting.is_none() && chromatic.is_none() {
            return None;
        }
        Some(Self {
            half_diagonal,
            distortion,
            vignetting,
            chromatic,
            unverified: Corrections::NONE,
        })
    }

    /// The correction in a Canon CR2 or CR3 (see [`canon`]).
    fn canon(path: &Path) -> Option<Self> {
        Self::from_canon(&canon::read(path)?)
    }

    /// The profile in Canon's tables. A correction is verified where the
    /// camera applied it to its JPEG, which is what the numbers were checked
    /// against; the rest start off.
    fn from_canon(tables: &canon::Tables) -> Option<Self> {
        let half_diagonal = (tables.width as f64).hypot(tables.height as f64) / 2.0;
        // Knots are radii in units of 6 µm of sensor, `pitch / 6000` per pixel.
        let radii: Vec<f64> = tables
            .knots
            .iter()
            .map(|k| k * 6000.0 / (tables.pitch_nm * half_diagonal))
            .collect();
        // 100% is the first value (8191, or 8199 for some).
        let unity = *tables.vignetting.first()?;
        let percent: Vec<f64> = tables.vignetting.iter().map(|v| 100.0 * v / unity).collect();
        let vignetting = Curve::new(&radii, &percent, 100.0)
            .filter(|c| c.values.iter().any(|&v| v < 100.0));
        let distortion = tables.distortion.as_ref().and_then(|d| {
            let ratios: Vec<f64> = d.iter().map(|v| v / 16384.0).collect();
            Some(Distortion::Ratio(Curve::new(&radii, &ratios, 1.0)?))
        });
        let distortion = distortion.filter(Distortion::does_something);
        let applied = tables.applied;
        let profile = Self {
            half_diagonal: Some(half_diagonal as f32),
            unverified: Corrections {
                distortion: !applied.is_some_and(|a| a.distortion),
                vignetting: !applied.is_some_and(|a| a.vignetting),
                chromatic: false,
            },
            distortion,
            vignetting,
            chromatic: None,
        };
        profile.available().any().then_some(profile)
    }

    /// The correction in a Sony ARW: tables in the raw SubIFD.
    fn sony(path: &Path) -> Option<Self> {
        let mut file = File::open(path).ok()?;
        let tiff = Tiff { base: 0 };
        let header = tiff.read(&mut file, 0, 8)?;
        let root = tiff.ifd(&mut file, u32_le(&header[4..])? as u64)?;
        let make = root.iter().find(|e| e.tag == 0x10f)?;
        let make = tiff.read(&mut file, u32_le(&make.value)? as u64, 4)?;
        if &make != b"SONY" {
            return None;
        }
        let subifds = root.iter().find(|e| e.tag == 0x14a)?;
        let offsets = if subifds.count == 1 {
            vec![u32_le(&subifds.value)?]
        } else {
            let raw = tiff.read(&mut file, u32_le(&subifds.value)? as u64, subifds.count as usize * 4)?;
            raw.chunks_exact(4).filter_map(u32_le).collect()
        };
        for offset in offsets {
            let Some(ifd) = tiff.ifd(&mut file, offset as u64) else { continue };
            if !ifd.iter().any(|e| e.tag == 0x7037) {
                continue;
            }
            // Signed 16-bit tables: [n, n values.., padding].
            let mut table = |tag| -> Option<Vec<i16>> {
                let entry = ifd.iter().find(|e| e.tag == tag)?;
                if entry.kind != 8 || entry.count < 3 || entry.count > 64 {
                    return None;
                }
                let data = tiff.read(&mut file, u32_le(&entry.value)? as u64, entry.count as usize * 2)?;
                Some(data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect())
            };
            let (distortion, vignetting, chromatic) = (table(0x7037), table(0x7032), table(0x7035));
            return Self::from_sony_tables(distortion.as_deref(), vignetting.as_deref(), chromatic.as_deref());
        }
        None
    }

    /// The profile in the words of tags 0x7037 (distortion), 0x7032
    /// (vignetting) and 0x7035 (CA); see the module docs. CA uses a guessed
    /// unit and is marked unverified.
    fn from_sony_tables(
        distortion: Option<&[i16]>,
        vignetting: Option<&[i16]>,
        chromatic: Option<&[i16]>,
    ) -> Option<Self> {
        // `count` values at radii i/(count-1) of the half-diagonal, from the
        // `skip`th word on.
        let values = |words: &[i16], count: usize, skip: usize| -> Option<(Vec<f64>, Vec<f64>)> {
            let end = skip.checked_add(count)?;
            (count >= 2 && end <= words.len()).then(|| {
                let knots = (0..count).map(|i| i as f64 / (count - 1) as f64).collect();
                (knots, words[skip..end].iter().map(|&w| w as f64).collect())
            })
        };
        // The leading word is the number of values.
        let counted = |words: &[i16]| usize::try_from(*words.first()?).ok();

        let distortion = distortion.and_then(|w| {
            let (knots, v) = values(w, counted(w)?, 1)?;
            let percent: Vec<f64> = v.iter().map(|d| d / SONY_UNIT * 100.0).collect();
            Some(Distortion::Percent(Curve::new(&knots, &percent, 0.0)?))
        });
        // Brightness percent left after an assumed `value / SONY_VIGNETTING_UNIT` stops.
        let vignetting = vignetting.and_then(|w| {
            let (knots, v) = values(w, counted(w)?, 1)?;
            let percent: Vec<f64> = v.iter().map(|s| 100.0 * (-s / SONY_VIGNETTING_UNIT).exp2()).collect();
            Curve::new(&knots, &percent, 100.0)
        });
        // Red then blue, each half of the leading count.
        let chromatic = chromatic.and_then(|w| {
            let n = counted(w)? / 2;
            let scale = |skip| {
                let (knots, v) = values(w, n, skip)?;
                let offsets: Vec<f64> = v.iter().map(|c| c / SONY_CA_UNIT).collect();
                Curve::new(&knots, &offsets, 0.0)
            };
            Some((scale(1)?, scale(1 + n)?))
        });

        let distortion = distortion.filter(Distortion::does_something);
        let vignetting = vignetting.filter(|c| c.values.iter().any(|&v| v < 100.0));
        let chromatic = chromatic.filter(|(r, b)| r.values.iter().chain(&b.values).any(|&c| c != 0.0));
        let profile = Self {
            half_diagonal: None,
            unverified: Corrections {
                distortion: false,
                vignetting: false,
                chromatic: chromatic.is_some(),
            },
            distortion,
            vignetting,
            chromatic,
        };
        profile.available().any().then_some(profile)
    }

    /// The correction in a Panasonic RW2: IFD0 tag 0x119.
    fn panasonic(path: &Path) -> Option<Self> {
        let mut file = File::open(path).ok()?;
        let tiff = Tiff { base: 0 };
        let header = tiff.read(&mut file, 0, 8)?;
        let ifd = tiff.ifd(&mut file, u32_le(&header[4..])? as u64)?;
        let entry = ifd.iter().find(|e| e.tag == 0x119 && e.kind == 7 && e.count == 32)?;
        let block = tiff.read(&mut file, u32_le(&entry.value)? as u64, 32)?;
        let words: Vec<i16> = block
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        Self::from_panasonic_block(words.try_into().ok()?)
    }

    /// The profile in the sixteen words of tag 0x119 (see the module docs).
    fn from_panasonic_block(words: [i16; 16]) -> Option<Self> {
        let p = |i: usize| words[i] as f32 / 32768.0;
        let half_diagonal = Some(words[12] as f32).filter(|d| *d > 0.0);
        let coefficients = [p(8), p(4), p(11)];
        // The correction must be monotonic to invert, well past the corner.
        let monotonic = (0..=160).all(|i| {
            let s2 = (i as f32 / 100.0).powi(2);
            let [a, b, c] = coefficients;
            1.0 + s2 * (3.0 * a + s2 * (5.0 * b + s2 * 7.0 * c)) > 0.0
        });
        let distortion = Distortion::Polynomial {
            zoom: 1.0 + p(5),
            coefficients,
        };
        if half_diagonal.is_none() || !monotonic || !distortion.does_something() {
            return None;
        }
        Some(Self {
            half_diagonal,
            distortion: Some(distortion),
            vignetting: None,
            chromatic: None,
            unverified: Corrections::NONE,
        })
    }

    /// The corrections whose numbers are a guess: off unless the user turns
    /// them on.
    pub fn unverified(&self) -> Corrections {
        self.unverified
    }

    /// The corrections the profile has data for.
    pub fn available(&self) -> Corrections {
        Corrections {
            distortion: self.distortion.is_some(),
            vignetting: self.vignetting.is_some(),
            chromatic: self.chromatic.is_some(),
        }
    }

    /// What the profile does, for the metadata panel.
    pub fn description(&self) -> String {
        let available = self.available();
        let names: Vec<_> = Kind::ALL
            .into_iter()
            .filter(|&k| available.get(k))
            .map(|k| k.label().to_lowercase())
            .collect();
        names.join(", ")
    }

    /// Half-diagonal in pixels of an image `width` × `height`: the profile's
    /// own unless that is far from the frame (a crop mode or another sensor
    /// readout than the profile assumes), then the frame's.
    pub fn half_diagonal(&self, width: u32, height: u32) -> f32 {
        let frame = (width as f32).hypot(height as f32) / 2.0;
        match self.half_diagonal {
            Some(d) if (0.8..1.25).contains(&(d / frame)) => d,
            _ => frame,
        }
    }

    /// Everything the shader needs for an image of `width` × `height`, with
    /// only the corrections in `on`:
    /// the unit radius in pixels and a table over the normalised radius of
    /// [`LUT_SIZE`] entries up to [`LUT_RHO_MAX`], each
    /// `[source radius / output radius, red CA, blue CA, vignetting gain]`.
    ///
    /// The correction samples the source at `r_out · scale(r_out)`, which
    /// each distortion model works out for itself (Fuji's keeps the picture
    /// its height, so the corners come in from outside the frame). The CA
    /// scales the red and blue sample radius on top, and the gain is looked
    /// up at the source radius.
    pub fn lut(&self, width: u32, height: u32, on: Corrections) -> (f32, Vec<[f32; 4]>) {
        let half_diagonal = self.half_diagonal(width, height);
        let short_edge = width.min(height) as f32 / 2.0 / half_diagonal;
        let distortion = self.distortion.as_ref().filter(|_| on.distortion);
        let vignetting = self.vignetting.as_ref().filter(|_| on.vignetting);
        let chromatic = self.chromatic.as_ref().filter(|_| on.chromatic);

        let lut = (0..LUT_SIZE)
            .map(|i| {
                let rho = i as f32 / (LUT_SIZE - 1) as f32 * LUT_RHO_MAX;
                let scale = distortion.map_or(1.0, |d| d.ratio(rho, short_edge));
                let (red, blue) =
                    chromatic.map_or((0.0, 0.0), |(r, b)| (r.at(rho), b.at(rho)));
                let gain = vignetting
                    .map_or(1.0, |v| (100.0 / v.at(rho).max(1.0)).min(MAX_GAIN));
                [scale, red, blue, gain]
            })
            .collect();
        (half_diagonal, lut)
    }
}

/// The TIFF structure inside the RAF; offsets are from `base`.
struct Tiff {
    base: u64,
}

struct Entry {
    tag: u16,
    kind: u16,
    count: u32,
    /// The inline value field, or the offset of the data.
    value: [u8; 4],
}

impl Tiff {
    fn read(&self, file: &mut File, offset: u64, len: usize) -> Option<Vec<u8>> {
        let mut buf = vec![0; len];
        file.seek(SeekFrom::Start(self.base + offset)).ok()?;
        file.read_exact(&mut buf).ok()?;
        Some(buf)
    }

    fn ifd(&self, file: &mut File, offset: u64) -> Option<Vec<Entry>> {
        let n = u16::from_le_bytes(self.read(file, offset, 2)?.try_into().ok()?) as usize;
        if n > 256 {
            return None;
        }
        let raw = self.read(file, offset + 2, n * 12)?;
        raw.chunks_exact(12)
            .map(|e| {
                Some(Entry {
                    tag: u16::from_le_bytes(e[0..2].try_into().ok()?),
                    kind: u16::from_le_bytes(e[2..4].try_into().ok()?),
                    count: u32_le(&e[4..8])?,
                    value: e[8..12].try_into().ok()?,
                })
            })
            .collect()
    }

    /// An `SRATIONAL` entry (type 10) as numbers.
    fn rationals(&self, file: &mut File, entry: &Entry) -> Option<Vec<f64>> {
        if entry.kind != 10 || entry.count == 0 || entry.count > 256 {
            return None;
        }
        let data = self.read(file, u32_le(&entry.value)? as u64, entry.count as usize * 8)?;
        data.chunks_exact(8)
            .map(|c| {
                let numerator = i32::from_le_bytes(c[0..4].try_into().ok()?);
                let denominator = i32::from_le_bytes(c[4..8].try_into().ok()?);
                (denominator != 0).then(|| numerator as f64 / denominator as f64)
            })
            .collect()
    }
}

fn u32_le(bytes: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table's source/output radius ratio at `rho`, interpolated.
    fn ratio_at(lut: &[[f32; 4]], rho: f32) -> f32 {
        column_at(lut, rho, 0)
    }

    fn column_at(lut: &[[f32; 4]], rho: f32, column: usize) -> f32 {
        let t = rho / LUT_RHO_MAX * (LUT_SIZE - 1) as f32;
        let i = t as usize;
        lut[i][column] + (t - i as f32) * (lut[i + 1][column] - lut[i][column])
    }

    fn curve(knots: &[f64], values: &[f64], origin: f32) -> Curve {
        Curve::new(knots, values, origin).unwrap()
    }

    #[test]
    fn curve_interpolates_and_holds() {
        let c = curve(&[0.5, 1.0], &[2.0, 4.0], 0.0);
        assert_eq!(c.at(0.0), 0.0);
        assert_eq!(c.at(0.25), 1.0);
        assert_eq!(c.at(0.75), 3.0);
        assert_eq!(c.at(1.25), 4.0);
        assert!(Curve::new(&[0.5, 0.4], &[1.0, 2.0], 0.0).is_none());
    }

    /// The XF16mm table from a real file: barrel distortion that the zoom
    /// leaves centred on the middle of the short edge.
    #[test]
    fn zoom_keeps_the_short_edge() {
        let knots = [0.35352, 0.5, 0.61268, 0.70704, 0.79085, 0.8662, 0.93521, 1.0, 1.0606];
        let values = [-1.5435, -3.4292, -5.1575, -6.7492, -8.312, -9.6693, -10.782, -11.68, -12.578];
        let lens = LensCorrection {
            half_diagonal: Some(3749.0),
            distortion: Some(Distortion::Percent(curve(&knots, &values, 0.0))),
            vignetting: None,
            chromatic: None,
            unverified: Corrections::NONE,
        };
        let (half_diagonal, lut) = lens.lut(6240, 4160, Corrections::ALL);
        let rho = 2080.0 / half_diagonal;
        let at = |rho: f32| {
            let t = rho / LUT_RHO_MAX * (LUT_SIZE - 1) as f32;
            let i = t as usize;
            lut[i][0] + (t - i as f32) * (lut[i + 1][0] - lut[i][0])
        };
        assert!((at(rho) - 1.0).abs() < 1e-3, "{}", at(rho));
        // Barrel: the corner samples inside the frame, the centre outside.
        assert!(at(1.0) < 0.95 && at(0.05) > 1.03);
    }

    #[test]
    fn each_correction_only_touches_its_own_column() {
        let knots = [0.5, 1.0];
        let lens = LensCorrection {
            half_diagonal: Some(1000.0),
            distortion: Some(Distortion::Percent(curve(&knots, &[-5.0, -10.0], 0.0))),
            vignetting: Some(curve(&knots, &[80.0, 60.0], 100.0)),
            chromatic: Some((curve(&knots, &[0.001, 0.002], 0.0), curve(&knots, &[-0.001, -0.002], 0.0))),
            unverified: Corrections::NONE,
        };
        let only = |kind: Kind| {
            let mut on = Corrections::NONE;
            on.set(kind, true);
            lens.lut(1200, 800, on).1[LUT_SIZE - 1]
        };
        let [scale, red, blue, gain] = only(Kind::Distortion);
        assert!(scale < 1.0 && red == 0.0 && blue == 0.0 && gain == 1.0);
        let [scale, red, blue, gain] = only(Kind::Vignetting);
        assert!(scale == 1.0 && red == 0.0 && blue == 0.0 && gain > 1.5);
        let [scale, red, blue, gain] = only(Kind::Chromatic);
        assert!(scale == 1.0 && red > 0.0 && blue < 0.0 && gain == 1.0);
        assert_eq!(lens.available(), Corrections::ALL);
    }

    /// The XF200mm table of a real X-T30 shot: pincushion, so the zoom must
    /// pull the corners in (the short edge rule would leave them blank).
    #[test]
    fn fuji_pincushion_zoom_keeps_the_corners_inside() {
        let knots = [0.35352, 0.5, 0.61268, 0.70704, 0.79085, 0.8662, 0.93521, 1.0, 1.0606];
        let values = [0.106, 0.21362, 0.3116, 0.40958, 0.49791, 0.57501, 0.65372, 0.71153, 0.76936];
        let lens = LensCorrection {
            half_diagonal: Some(3749.0),
            distortion: Some(Distortion::Percent(curve(&knots, &values, 0.0))),
            vignetting: None,
            chromatic: None,
            unverified: Corrections::NONE,
        };
        let (_, lut) = lens.lut(6240, 4160, Corrections::ALL);
        // The corner samples exactly at the corner, no further.
        assert!((ratio_at(&lut, 1.0) - 1.0).abs() < 1e-3, "{}", ratio_at(&lut, 1.0));
    }

    /// Real A6700 words: the 15mm f/1.4 (strong barrel) and, from an A7 III,
    /// the 55mm f/1.8 at f/1.8 (small, S-shaped table).
    const SONY_15MM: [i16; 17] = [
        11, 0, -27, -101, -218, -368, -548, -755, -988, -1246, -1527, -1833, -1833, -1833, -1833,
        -1833, -1833,
    ];
    const SONY_55MM: [i16; 17] = [16, -8, -2, 5, 13, 23, 34, 46, 58, 69, 79, 87, 91, 88, 76, 54, 16];
    /// The 15mm f/1.4 vignetting and CA words of the same shot.
    const SONY_15MM_VIGNETTING: [i16; 17] = [
        11, 0, 48, 224, 512, 912, 1424, 2048, 2768, 3776, 4992, 6208, 0, 0, 0, 0, 0,
    ];
    const SONY_15MM_CA: [i16; 33] = [
        22, 1536, 1408, 1280, 1152, 1024, 1024, 896, 640, 640, 896, 1664, 0, 0, -128, -128, -128,
        0, 128, 384, 384, 0, -1152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    #[test]
    fn sony_ca_is_a_guess_and_marked_unverified() {
        let lens = LensCorrection::from_sony_tables(
            Some(&SONY_15MM),
            Some(&SONY_15MM_VIGNETTING),
            Some(&SONY_15MM_CA),
        )
        .unwrap();
        assert_eq!(lens.available(), Corrections::ALL);
        // Only the CA unit is a guess.
        assert_eq!(
            lens.unverified(),
            Corrections { distortion: false, vignetting: false, chromatic: true }
        );
        let only = |kind: Kind| {
            let mut on = Corrections::NONE;
            on.set(kind, true);
            lens.lut(6192, 4128, on).1
        };
        // Corner gain 2^(6208/8192) from the table's last knot.
        let gain = column_at(&only(Kind::Vignetting), 1.0, 3);
        assert!((gain - (6208.0f32 / 8192.0).exp2()).abs() < 5e-3, "{gain}");
        // Red and blue scale by value / 2^20 at the corner.
        let ca = only(Kind::Chromatic);
        assert!((column_at(&ca, 1.0, 1) - 1664.0 / 1_048_576.0).abs() < 3e-5);
        assert!((column_at(&ca, 1.0, 2) + 1152.0 / 1_048_576.0).abs() < 3e-5);
        // Each correction touches only its own column.
        assert_eq!(column_at(&only(Kind::Vignetting), 1.0, 1), 0.0);
        assert_eq!(column_at(&only(Kind::Distortion), 1.0, 3), 1.0);
        // Missing or empty tables are simply absent; all-empty is no profile.
        let flat = LensCorrection::from_sony_tables(Some(&SONY_15MM), None, None).unwrap();
        assert_eq!(flat.unverified(), Corrections::NONE);
        assert!(LensCorrection::from_sony_tables(None, Some(&[11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), None).is_none());
        // `without` is what the program uses to leave guesses off.
        assert_eq!(
            Corrections::ALL.without(lens.unverified()),
            Corrections { chromatic: false, ..Corrections::ALL }
        );
    }

    /// An A6700 table that is pincushion: the corners map to themselves.
    const SONY_39MM: [i16; 17] = [11, 0, 2, 10, 21, 39, 64, 97, 140, 196, 269, 368, 368, 368, 368, 368, 368];

    #[test]
    fn sony_distortion_uses_the_unit_and_the_zoom_rule() {
        let barrel = LensCorrection::from_sony_tables(Some(&SONY_15MM), None, None).unwrap();
        assert_eq!(barrel.available(), Corrections { distortion: true, ..Corrections::NONE });
        let (half_diagonal, lut) = barrel.lut(6192, 4128, Corrections::ALL);
        assert_eq!(half_diagonal, (6192f32).hypot(4128.0) / 2.0);
        // Middle of the short edge stays put; zoom = 1/(1 - 988/16384...).
        let short_edge = 4128.0 / 2.0 / half_diagonal;
        let at = |rho: f32| ratio_at(&lut, rho);
        assert!((at(short_edge) - 1.0).abs() < 2e-3, "{}", at(short_edge));
        // The corner: -1833/16384 = -11.2% before the zoom.
        let zoom = at(short_edge) / (1.0 + (-661.0 / 16384.0));
        assert!((at(1.0) / zoom - (1.0 - 1833.0 / 16384.0)).abs() < 2e-3);

        let pincushion = LensCorrection::from_sony_tables(Some(&SONY_39MM), None, None).unwrap();
        let lut = pincushion.lut(6192, 4128, Corrections::ALL).1;
        assert!((ratio_at(&lut, 1.0) - 1.0).abs() < 2e-3, "{}", ratio_at(&lut, 1.0));

        // The A7 III table has 16 points and a non-monotonic shape.
        assert!(LensCorrection::from_sony_tables(Some(&SONY_55MM), None, None).is_some());
        // Nothing to do, or nonsense counts.
        assert!(LensCorrection::from_sony_tables(Some(&[11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), None, None).is_none());
        assert!(LensCorrection::from_sony_tables(Some(&[40, 1, 2, 3]), None, None).is_none());
        assert!(LensCorrection::from_sony_tables(Some(&[]), None, None).is_none());
    }

    /// The tag 0x119 words of two real DC-S5M2 shots: an 85mm f/1.8 and a
    /// 25mm zoom setting (pincushion corrected one way, barrel the other).
    const S5M2_85MM: [i16; 16] = [
        9830, 23432, 529, 168, 82, -439, 109, 9217, -640, 220, 346, 153, 3606, 720, 1796, 17863,
    ];
    const S5M2_25MM: [i16; 16] = [
        11705, -13325, 485, 0, -211, 0, 374, 1, 1472, 646, 346, -6, 3606, 1146, 23711, 22531,
    ];

    /// A DC-S1M2 with the Lumix S 24-60mm F2.8 at 24mm, a body and lens the
    /// reading was not derived from: the unfitted model matched the camera's
    /// JPEG at NCC 0.94 where a free fit reaches 0.96.
    const S1M2_24MM: [i16; 16] = [
        2276, -21003, 347, 474, -786, -753, 265, -30975, -252, 489, 346, 340, 3606, 989, 14610,
        16380,
    ];

    /// The same body at 35mm f/4, checked against a full-resolution camera
    /// JPEG: corrected NCC 0.97 at every radius (0.18 and -0.11 in the outer
    /// two bands before).
    const S1M2_35MM: [i16; 16] = [
        -13451, -28095, 649, 0, -59, 0, 576, 1, 532, 478, 346, -123, 3606, 978, -10052, -14523,
    ];

    #[test]
    fn panasonic_polynomial_inverts_and_zooms() {
        let lens = LensCorrection::from_panasonic_block(S5M2_85MM).unwrap();
        assert_eq!(lens.available(), Corrections { distortion: true, ..Corrections::NONE });
        let (half_diagonal, lut) = lens.lut(6000, 4000, Corrections::ALL);
        assert_eq!(half_diagonal, 3606.0);
        // Centre: just the zoom, 1 + word5/32768.
        assert!((lut[0][0] - (1.0 - 439.0 / 32768.0)).abs() < 1e-6);
        // The sampled radius, pushed back through the correction, is the
        // output radius.
        for i in [40, 120, 200] {
            let rho = i as f32 / (LUT_SIZE - 1) as f32 * LUT_RHO_MAX;
            let s = lut[i][0] * rho / (1.0 - 439.0 / 32768.0);
            let back = s * (1.0 + s * s * (-640.0 / 32768.0 + s * s * (82.0 / 32768.0 + s * s * 153.0 / 32768.0)));
            assert!((back - rho).abs() < 1e-4, "{back} vs {rho}");
        }
        // The wide shot is the barrel one: the corner samples inside.
        let wide = LensCorrection::from_panasonic_block(S5M2_25MM).unwrap();
        assert!(wide.lut(6000, 4000, Corrections::ALL).1[170][0] < 1.0);
        let s1 = LensCorrection::from_panasonic_block(S1M2_24MM).unwrap();
        assert!(s1.lut(6000, 4000, Corrections::ALL).1[170][0] < 1.0);
        // No zoom at 35mm (word 5 is 0): the centre samples where it is.
        let mid = LensCorrection::from_panasonic_block(S1M2_35MM).unwrap();
        assert_eq!(mid.lut(6000, 4000, Corrections::ALL).1[0][0], 1.0);
    }

    #[test]
    fn panasonic_rejects_blocks_that_do_nothing_or_cannot_invert() {
        let mut flat = S5M2_85MM;
        flat[5] = 0;
        flat[8] = 0;
        flat[4] = 0;
        flat[11] = 0;
        assert!(LensCorrection::from_panasonic_block(flat).is_none());
        let mut folded = S5M2_85MM;
        folded[8] = -30000; // a radius that folds back on itself
        assert!(LensCorrection::from_panasonic_block(folded).is_none());
        let mut no_diagonal = S5M2_85MM;
        no_diagonal[12] = 0;
        assert!(LensCorrection::from_panasonic_block(no_diagonal).is_none());
    }

    /// The R50 + RF-S 18-45mm at 18mm tables, as `canon::Tables`: the pixel
    /// pitch makes a knot 0.62 pixels.
    fn canon_r50(applied: Option<canon::Applied>) -> canon::Tables {
        let knots: Vec<f64> = [
            0, 189, 379, 568, 757, 946, 1136, 1325, 1514, 1703, 1893, 1987, 2082, 2177, 2271, 2366, 2461,
            2555, 2650, 2745, 2839, 2934,
        ]
        .iter()
        .map(|&k| k as f64)
        .collect();
        let vignetting = [
            8191, 8179, 8140, 8075, 7985, 7867, 7723, 7555, 7363, 7134, 6695, 6361, 6002, 5606, 5198, 4791,
            4382, 3975, 3566, 3159, 2750, 2342,
        ];
        let distortion = [
            16384, 16311, 16266, 16192, 16087, 15956, 15802, 15627, 15438, 15238, 15029, 14922, 14815,
            14706, 14598, 14489, 14380, 14273, 14166, 14059, 13957, 13861,
        ];
        canon::Tables {
            pitch_nm: 3720.0,
            width: 6000,
            height: 4000,
            vignetting: vignetting.iter().map(|&v| v as f64).collect(),
            knots,
            distortion: Some(distortion.iter().map(|&v| v as f64).collect()),
            applied,
        }
    }

    #[test]
    fn canon_tables_become_a_profile_in_pixels_with_the_shared_zoom() {
        let applied = canon::Applied { vignetting: true, distortion: true };
        let lens = LensCorrection::from_canon(&canon_r50(Some(applied))).unwrap();
        assert_eq!(lens.available(), Corrections { chromatic: false, ..Corrections::ALL });
        assert_eq!(lens.unverified(), Corrections::NONE);
        let (half_diagonal, lut) = lens.lut(6000, 4000, Corrections::ALL);
        assert!((half_diagonal - 3605.55).abs() < 0.01);
        // The middle of the short edge (rho 0.5547) is the tightest edge
        // point, so it keeps its place and the zoom is 1 / its table ratio.
        let short_edge = 2000.0 / half_diagonal;
        assert!((ratio_at(&lut, short_edge) - 1.0).abs() < 2e-3, "{}", ratio_at(&lut, short_edge));
        // 600 / 16384-ish at the centre: the zoom pushes it above 1.
        let zoom = ratio_at(&lut, 0.0);
        assert!((1.03..1.06).contains(&zoom), "{zoom}");
        // The corner is knot 3605.55 * 3720 / 6000 = 2235.4: between 2177
        // (14706) and 2271 (14598), 0.62 of the way: 14639 / 16384, zoomed.
        let expected = zoom * 14639.0 / 16384.0;
        assert!((ratio_at(&lut, 1.0) - expected).abs() < 2e-3, "{}", ratio_at(&lut, 1.0));
        // Vignetting: 8191 / 5353 at the same knot.
        let gain = column_at(&lut, 1.0, 3);
        assert!((gain - 8191.0 / 5353.0).abs() < 0.01, "{gain}");
    }

    #[test]
    fn canon_corrections_are_verified_only_where_the_camera_applied_them() {
        // The camera applied distortion but not the peripheral lighting.
        let applied = canon::Applied { vignetting: false, distortion: true };
        let lens = LensCorrection::from_canon(&canon_r50(Some(applied))).unwrap();
        assert_eq!(lens.unverified(), Corrections { vignetting: true, ..Corrections::NONE });
        // Without the camera's settings nothing is confirmed.
        let lens = LensCorrection::from_canon(&canon_r50(None)).unwrap();
        assert_eq!(lens.unverified(), Corrections { chromatic: false, ..Corrections::ALL });
    }

    #[test]
    fn canon_vignetting_alone_is_a_profile() {
        let mut tables = canon_r50(Some(canon::Applied { vignetting: true, distortion: false }));
        tables.distortion = None;
        let lens = LensCorrection::from_canon(&tables).unwrap();
        assert_eq!(lens.available(), Corrections { vignetting: true, ..Corrections::NONE });
        // A flat table does nothing.
        tables.vignetting = vec![8191.0; 22];
        assert!(LensCorrection::from_canon(&tables).is_none());
    }
}
