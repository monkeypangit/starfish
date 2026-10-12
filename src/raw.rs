//! Raw file decoding via `rawler`.
//!
//! Produces a [`RawFrame`]: the cropped, black/white-level normalised sensor
//! mosaic plus the metadata the GPU stages need (CFA layout, white balance,
//! camera → sRGB matrix).

use std::path::{Path, PathBuf};

use rawler::imgop::xyz::Illuminant;
use rawler::rawimage::{RawImage, RawImageData, RawPhotometricInterpretation};

use crate::lens::LensCorrection;

/// File extensions offered in the open dialog.
/// Both cases, because file dialogs match extensions case-sensitively and many
/// cameras write them in uppercase (`.RAF`, `.NEF`, ...).
pub const RAW_EXTENSIONS: &[&str] = &[
    "3fr", "ari", "arw", "cr2", "cr3", "crw", "dcr", "dng", "erf", "iiq", "kdc", "mef", "mos",
    "mrw", "nef", "nrw", "orf", "pef", "raf", "raw", "rw2", "rwl", "srw", "3FR", "ARI", "ARW",
    "CR2", "CR3", "CRW", "DCR", "DNG", "ERF", "IIQ", "KDC", "MEF", "MOS", "MRW", "NEF", "NRW",
    "ORF", "PEF", "RAF", "RAW", "RW2", "RWL", "SRW",
];

/// Largest supported CFA period (X-Trans is 6×6).
pub const MAX_CFA_SIZE: usize = 6;

/// Colour filter array layout, relative to the top-left of the cropped frame.
#[derive(Debug, Clone)]
pub struct CfaPattern {
    pub width: usize,
    pub height: usize,
    /// Row-major colour index per site: 0 = R, 1 = G, 2 = B.
    pub colors: Vec<u32>,
}

impl CfaPattern {
    pub fn kind(&self) -> SensorKind {
        match (self.width, self.height) {
            (2, 2) => SensorKind::Bayer,
            (6, 6) => SensorKind::XTrans,
            _ => SensorKind::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorKind {
    Bayer,
    XTrans,
    Other,
}

impl std::fmt::Display for SensorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SensorKind::Bayer => "Bayer",
            SensorKind::XTrans => "X-Trans",
            SensorKind::Other => "Unknown CFA",
        })
    }
}

/// A decoded, normalised sensor mosaic ready for demosaicing.
#[derive(Debug, Clone)]
pub struct RawFrame {
    pub path: PathBuf,
    pub camera: String,
    pub width: usize,
    pub height: usize,
    /// One sample per pixel, black level subtracted and scaled so white = 1.0.
    /// Not clamped at 0: the noise below black is kept, as the neural
    /// demosaic's noise estimate and training data expect.
    pub data: Vec<f32>,
    pub cfa: CfaPattern,
    /// White balance multipliers (R, G, B), normalised so G = 1.
    pub wb: [f32; 3],
    /// Row-major matrix converting white-balanced camera RGB to linear sRGB.
    pub cam_to_srgb: [[f32; 3]; 3],
    /// For the temperature slider; `None` if the file's metadata is unusable.
    pub white: Option<WhiteBalance>,
    /// Label and value rows for the metadata panel, in display order.
    pub metadata: Vec<(&'static str, String)>,
    /// The lens correction embedded in the file, if any (Fujifilm, Panasonic).
    pub lens: Option<LensCorrection>,
}

/// What the temperature slider needs to know about the camera and the
/// as-shot white balance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhiteBalance {
    xyz_to_cam: [[f32; 3]; 3],
    /// The as-shot illuminant's correlated colour temperature, in mireds
    /// (1e6 / kelvin).
    mired: f32,
}

/// Mireds a slider end shifts the white balance by.
const MAX_MIRED_SHIFT: f32 = 100.0;
/// Hottest setting offered (50000 K).
const MIN_MIRED: f32 = 20.0;

impl WhiteBalance {
    /// From the camera matrix and the as-shot multipliers (G = 1): the scene
    /// neutral in camera RGB is their reciprocal, which the inverse matrix
    /// takes to XYZ and McCamy's formula to a CCT.
    fn new(xyz_to_cam: [[f32; 3]; 3], wb: [f32; 3]) -> Option<Self> {
        let cam_to_xyz = invert3(xyz_to_cam)?;
        let neutral = wb.map(|c| 1.0 / c);
        let [x, y, z] = cam_to_xyz.map(|row| (0..3).map(|k| row[k] * neutral[k]).sum::<f32>());
        let sum = x + y + z;
        if !sum.is_finite() || sum <= 0.0 {
            return None;
        }
        let (x, y) = (x / sum, y / sum);
        let n = (x - 0.3320) / (0.1858 - y);
        let kelvin = 449.0 * n * n * n + 3525.0 * n * n + 6823.3 * n + 5520.33;
        (kelvin.is_finite() && kelvin > 1000.0 && kelvin < 40000.0).then(|| Self {
            xyz_to_cam,
            mired: 1e6 / kelvin,
        })
    }

    fn target_mired(&self, temperature: f32) -> f32 {
        (self.mired - temperature * MAX_MIRED_SHIFT).max(MIN_MIRED)
    }

    /// The colour temperature the slider shows at `temperature` (−1 … +1):
    /// the as-shot one, shifted in mireds. Warm (+) is higher, as the
    /// slider names the illuminant the scene is corrected for.
    pub fn kelvin(&self, temperature: f32) -> f32 {
        1e6 / self.target_mired(temperature)
    }

    /// The slider position (−1 … +1, or beyond) that shows `kelvin`: the
    /// inverse of [`Self::kelvin`] below the hottest setting.
    pub fn temperature_for(&self, kelvin: f32) -> f32 {
        (self.mired - 1e6 / kelvin) / MAX_MIRED_SHIFT
    }

    /// Camera RGB gains (G = 1) that move the as-shot neutral to the one of
    /// `kelvin(temperature)`, both taken on the same daylight/Planckian
    /// locus so zero is exactly no change.
    pub fn gains(&self, temperature: f32) -> [f32; 3] {
        let from = self.neutral(1e6 / self.mired);
        let to = self.neutral(self.kelvin(temperature));
        let g: [f32; 3] = std::array::from_fn(|c| from[c] / to[c]);
        [g[0] / g[1], 1.0, g[2] / g[1]]
    }

    /// Camera RGB of the white at `kelvin`.
    fn neutral(&self, kelvin: f32) -> [f32; 3] {
        // Kim et al. (2002): the Planckian locus below 4000 K, the daylight
        // locus above it.
        let t = kelvin.clamp(1667.0, 25000.0);
        let (t1, t2, t3) = (1e3 / t, 1e6 / (t * t), 1e9 / (t * t * t));
        let x = if t < 4000.0 {
            -0.2661239 * t3 - 0.2343589 * t2 + 0.8776956 * t1 + 0.179910
        } else {
            -3.0258469 * t3 + 2.1070379 * t2 + 0.2226347 * t1 + 0.240390
        };
        let y = if t < 2222.0 {
            -1.1063814 * x * x * x - 1.34811020 * x * x + 2.18555832 * x - 0.20219683
        } else if t < 4000.0 {
            -0.9549476 * x * x * x - 1.37418593 * x * x + 2.09137015 * x - 0.16748867
        } else {
            3.0817580 * x * x * x - 5.87338670 * x * x + 3.75112997 * x - 0.37001483
        };
        let xyz = [x / y, 1.0, (1.0 - x - y) / y];
        self.xyz_to_cam
            .map(|row| (0..3).map(|k| row[k] * xyz[k]).sum())
    }
}

pub fn decode(path: &Path) -> Result<RawFrame, String> {
    let image = rawler::decode_file(path).map_err(|e| format!("Failed to decode: {e}"))?;

    let cfa = match &image.photometric {
        RawPhotometricInterpretation::Cfa(config) => config.cfa.clone(),
        RawPhotometricInterpretation::LinearRaw => {
            return Err("Linear (already demosaiced) raw files are not supported yet".into());
        }
        RawPhotometricInterpretation::BlackIsZero => {
            return Err("Monochrome raw files are not supported yet".into());
        }
    };
    if image.cpp != 1 {
        return Err(format!("Expected 1 sample per pixel, got {}", image.cpp));
    }

    let camera = format!("{} {}", image.clean_make, image.clean_model);
    let wb = white_balance(&image);
    let cam_to_srgb = cam_to_srgb(&image);
    let white = xyz_to_cam(&image).and_then(|m| WhiteBalance::new(m, wb));

    let crop = image
        .crop_area
        .or(image.active_area)
        .map(|r| (r.p.x, r.p.y, r.d.w, r.d.h))
        .filter(|&(x, y, w, h)| w > 0 && h > 0 && x + w <= image.width && y + h <= image.height)
        .unwrap_or((0, 0, image.width, image.height));

    let cfa = cfa.shift(crop.0, crop.1);
    if cfa.width == 0 || cfa.height == 0 || cfa.width > MAX_CFA_SIZE || cfa.height > MAX_CFA_SIZE {
        return Err(format!("Unsupported CFA pattern '{}'", cfa.name));
    }
    let mut colors = Vec::with_capacity(cfa.width * cfa.height);
    for row in 0..cfa.height {
        for col in 0..cfa.width {
            colors.push(match cfa.color_at(row, col) {
                0 => 0,
                1 | 7 => 1, // 7 = FUJI_GREEN
                2 => 2,
                other => return Err(format!("Unsupported CFA colour index {other}")),
            });
        }
    }

    let data = crop_data(&image, crop);
    let lens = LensCorrection::read(path);
    let mut metadata = metadata(path, &image, crop);
    if let Some(lens) = &lens {
        metadata.push(("Lens profile", lens.description()));
    }

    Ok(RawFrame {
        path: path.to_path_buf(),
        camera,
        width: crop.2,
        height: crop.3,
        data,
        cfa: CfaPattern {
            width: cfa.width,
            height: cfa.height,
            colors,
        },
        wb,
        cam_to_srgb,
        white,
        metadata,
        lens,
    })
}

/// The rows of the metadata panel. Camera and sensor details come from the
/// decoded image, the shooting data from the EXIF; whatever the file lacks
/// is left out.
fn metadata(
    path: &Path,
    image: &RawImage,
    crop: (usize, usize, usize, usize),
) -> Vec<(&'static str, String)> {
    let mut rows = Vec::new();
    let mut add = |label, value: Option<String>| {
        if let Some(value) = value.filter(|v| !v.is_empty()) {
            rows.push((label, value));
        }
    };

    let exif = rawler::rawsource::RawSource::new(path)
        .ok()
        .and_then(|source| {
            let decoder = rawler::get_decoder(&source).ok()?;
            decoder.raw_metadata(&source, &Default::default()).ok()
        });
    let e = exif.as_ref().map(|m| &m.exif);
    let rational = |r: rawler::formats::tiff::Rational| (r.d != 0).then(|| r.n as f64 / r.d as f64);
    let srational =
        |r: rawler::formats::tiff::SRational| (r.d != 0).then(|| r.n as f64 / r.d as f64);

    add(
        "Camera",
        Some(format!("{} {}", image.clean_make, image.clean_model)),
    );
    add(
        "Lens",
        exif.as_ref()
            .and_then(|m| {
                m.lens
                    .as_ref()
                    .map(|l| format!("{} {}", l.lens_make, l.lens_model))
            })
            .or_else(|| e.and_then(|e| e.lens_model.clone())),
    );
    add(
        "Taken",
        e.and_then(|e| {
            e.date_time_original
                .clone()
                .or_else(|| e.create_date.clone())
        }),
    );
    add(
        "Focal length",
        e.and_then(|e| e.focal_length)
            .and_then(rational)
            .map(|v| format!("{v:.0} mm")),
    );
    add(
        "Aperture",
        e.and_then(|e| e.fnumber)
            .and_then(rational)
            .map(|v| format!("f/{v:.1}")),
    );
    add(
        "Shutter",
        e.and_then(|e| e.exposure_time).and_then(rational).map(|t| {
            if t > 0.0 && t < 0.5 {
                format!("1/{:.0} s", 1.0 / t)
            } else {
                format!("{t:.1} s")
            }
        }),
    );
    add(
        "ISO",
        e.and_then(|e| e.iso_speed_ratings.map(u32::from).or(e.iso_speed))
            .map(|v| v.to_string()),
    );
    add(
        "Exposure bias",
        e.and_then(|e| e.exposure_bias)
            .and_then(srational)
            .map(|v| format!("{v:+.1} EV")),
    );
    add("Artist", e.and_then(|e| e.artist.clone()));
    add("Copyright", e.and_then(|e| e.copyright.clone()));
    add("Size", Some(format!("{} × {}", crop.2, crop.3)));
    add(
        "Megapixels",
        Some(format!("{:.1} MP", (crop.2 * crop.3) as f64 / 1e6)),
    );
    add("Bit depth", Some(format!("{} bit", image.bps)));
    add(
        "File size",
        std::fs::metadata(path)
            .ok()
            .map(|m| format!("{:.1} MB", m.len() as f64 / 1e6)),
    );
    add(
        "File",
        path.file_name().map(|n| n.to_string_lossy().into_owned()),
    );
    rows
}

/// Cuts out the crop and normalises it per site: black → 0, white → 1.
/// Unlike rawler's `apply_scaling`, values below black are not clamped, and
/// black level patterns of any size (not just 2×2) are honoured.
fn crop_data(image: &RawImage, (x, y, w, h): (usize, usize, usize, usize)) -> Vec<f32> {
    let bl = &image.blacklevel;
    let (bw, bh) = (bl.width.max(1), bl.height.max(1));
    // Black level patterns are anchored at the sensor origin.
    let black_at = |row: usize, col: usize| match bl.levels.len() {
        0 => 0.0,
        n => bl.levels[((row % bh) * bw + col % bw) % n].as_f32(),
    };
    let white = image
        .whitelevel
        .0
        .first()
        .copied()
        .unwrap_or(u16::MAX as u32) as f32;

    let mut out = Vec::with_capacity(w * h);
    for row in y..y + h {
        let start = row * image.width;
        for col in x..x + w {
            let v = match &image.data {
                RawImageData::Integer(d) => d[start + col] as f32,
                RawImageData::Float(d) => d[start + col],
            };
            let black = black_at(row, col);
            out.push((v - black) / (white - black));
        }
    }
    out
}

/// The camera's XYZ → camera matrix. rawler's `xyz_to_cam` field is
/// deprecated and empty for most cameras; the DNG-style `color_matrix` map
/// has it per illuminant. Prefers the daylight ones, whose white points are
/// closest to a typical shot.
fn xyz_to_cam(image: &RawImage) -> Option<[[f32; 3]; 3]> {
    use Illuminant::*;
    let (_, flat) = image
        .color_matrix_find_first([D65, D55, D50, Daylight, FineWeather, A])
        .or_else(|| {
            image
                .color_matrix
                .iter()
                .next()
                .map(|(i, m)| (*i, m.clone()))
        })?;
    (flat.len() >= 9 && flat[..9].iter().all(|v| v.is_finite()))
        .then(|| std::array::from_fn(|i| std::array::from_fn(|j| flat[i * 3 + j])))
}

fn white_balance(image: &RawImage) -> [f32; 3] {
    let [r, g, b, _] = image.wb_coeffs;
    if [r, g, b].iter().all(|c| c.is_finite() && *c > 0.0) {
        [r / g, 1.0, b / g]
    } else {
        [1.0, 1.0, 1.0]
    }
}

/// Camera RGB → linear sRGB, following dcraw: build sRGB → camera from the
/// XYZ → camera matrix, normalise rows so white stays white after white
/// balance, then invert.
fn cam_to_srgb(image: &RawImage) -> [[f32; 3]; 3] {
    const SRGB_TO_XYZ: [[f32; 3]; 3] = [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ];
    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let xyz_to_cam = image.xyz_to_cam;
    let mut srgb_to_cam = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            srgb_to_cam[i][j] = (0..3).map(|k| xyz_to_cam[i][k] * SRGB_TO_XYZ[k][j]).sum();
        }
        let sum: f32 = srgb_to_cam[i].iter().sum();
        if !sum.is_finite() || sum.abs() < 1e-6 {
            return IDENTITY;
        }
        srgb_to_cam[i].iter_mut().for_each(|v| *v /= sum);
    }
    invert3(srgb_to_cam).unwrap_or(IDENTITY)
}

fn invert3(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if !det.is_finite() || det.abs() < 1e-9 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            c00 * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            c01 * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            c02 * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera that sees linear sRGB, shot with a D65 white balance.
    fn srgb_camera() -> WhiteBalance {
        let xyz_to_srgb = [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ];
        WhiteBalance::new(xyz_to_srgb, [1.0, 1.0, 1.0]).unwrap()
    }

    #[test]
    fn as_shot_temperature_and_gains() {
        let white = srgb_camera();
        assert!(
            (white.kelvin(0.0) - 6500.0).abs() < 150.0,
            "{}",
            white.kelvin(0.0)
        );
        assert_eq!(white.gains(0.0), [1.0, 1.0, 1.0]);
        let [r, g, b] = white.gains(0.5);
        assert!(white.kelvin(0.5) > white.kelvin(0.0) && r > 1.0 && g == 1.0 && b < 1.0);
    }
}
