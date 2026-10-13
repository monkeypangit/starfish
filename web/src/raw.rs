//! Raw decoding from the file's bytes: the app's `src/raw.rs` `decode`
//! without the metadata, the white-balance slider and the lens profile.

use rawler::rawimage::{RawImage, RawImageData, RawPhotometricInterpretation};
use rawler::rawsource::RawSource;

/// A decoded, normalised sensor mosaic, as the app's `RawFrame`.
pub struct Frame {
    pub camera: String,
    pub width: usize,
    pub height: usize,
    /// Black level subtracted and scaled so white = 1.0; not clamped at 0
    /// (the noise fit needs the samples below black).
    pub data: Vec<f32>,
    /// Row-major colour per CFA site, 0 = R, 1 = G, 2 = B, `cfa_size` (w, h).
    pub cfa: Vec<u32>,
    pub cfa_size: (usize, usize),
    /// White balance multipliers (R, G, B), G = 1.
    pub wb: [f32; 3],
    /// White-balanced camera RGB -> linear sRGB, row-major.
    pub cam_to_srgb: [[f32; 3]; 3],
}

/// Largest supported CFA period (X-Trans is 6×6).
const MAX_CFA_SIZE: usize = 6;

pub fn decode(bytes: &[u8]) -> Result<Frame, String> {
    let source = RawSource::new_from_slice(bytes);
    let image = rawler::decode(&source, &Default::default())
        .map_err(|e| format!("Failed to decode: {e}"))?;

    let cfa = match &image.photometric {
        RawPhotometricInterpretation::Cfa(config) => config.cfa.clone(),
        RawPhotometricInterpretation::LinearRaw => {
            return Err("Linear (already demosaiced) raw files are not supported".into());
        }
        RawPhotometricInterpretation::BlackIsZero => {
            return Err("Monochrome raw files are not supported".into());
        }
    };
    if image.cpp != 1 {
        return Err(format!("Expected 1 sample per pixel, got {}", image.cpp));
    }

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

    Ok(Frame {
        camera: format!("{} {}", image.clean_make, image.clean_model),
        width: crop.2,
        height: crop.3,
        data: crop_data(&image, crop),
        cfa: colors,
        cfa_size: (cfa.width, cfa.height),
        wb: white_balance(&image),
        cam_to_srgb: cam_to_srgb(&image),
    })
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
