//! Real camera raw -> the network's input, via rawler. A port of rawload.py's
//! load_raw (black/white normalisation, CFA phase rotation, WB, colour matrix);
//! the noise fit is in the library, noise.rs.
//!
//! The mosaic comes back *un-balanced*, because that is where the noise model
//! lives; the caller builds sigma there and then applies `gain` per CFA site to
//! both, as infer.py does.

use rawler::imgop::xyz::Illuminant;
use rawler::rawimage::{RawImageData, RawPhotometricInterpretation};

pub struct Raw {
    pub h: usize,
    pub w: usize,
    pub mosaic: Vec<f32>,        // un-balanced linear raw, phase matches `cfa`
    pub gain: [f32; 3],          // raw -> balanced, green anchored at 1
    pub srgb_from_cam: [f32; 9], // balanced camera -> linear sRGB, row-major
    pub notes: String,
}

/// data.SRGB_TO_XYZ
const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

/// data.CAM_XYZ_BANK[DEFAULT_CAM], the fallback when a file has no matrix.
const DEFAULT_CAM_XYZ: [[f64; 3]; 3] = [
    [1.0405, -0.3755, -0.1270],
    [-0.5461, 1.3787, 0.1793],
    [-0.1040, 0.2015, 0.6785],
];

fn matmul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

fn inv3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c =
        |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let det = m[0][0] * c(1, 1, 2, 2) - m[0][1] * c(1, 0, 2, 2) + m[0][2] * c(1, 0, 2, 1);
    [
        [
            c(1, 1, 2, 2) / det,
            -c(0, 1, 2, 2) / det,
            c(0, 1, 1, 2) / det,
        ],
        [
            -c(1, 0, 2, 2) / det,
            c(0, 0, 2, 2) / det,
            -c(0, 0, 1, 2) / det,
        ],
        [
            c(1, 0, 2, 1) / det,
            -c(0, 0, 2, 1) / det,
            c(0, 0, 1, 1) / det,
        ],
    ]
}

/// (srgb_from_cam, note): row-normalised linear-sRGB -> camera, inverted.
fn cam_matrix(img: &rawler::RawImage) -> ([f32; 9], &'static str) {
    let flat = img
        .color_matrix
        .get(&Illuminant::D65)
        .or_else(|| img.color_matrix.values().next())
        .filter(|v| v.len() >= 9 && v.iter().any(|x| *x != 0.0));
    let (xyz, note) = match flat {
        Some(v) => {
            let mut x = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    x[i][j] = v[3 * i + j] as f64;
                }
            }
            (x, "ColorMatrix from file")
        }
        None => (
            DEFAULT_CAM_XYZ,
            "no ColorMatrix in file; using data.CAM_FROM_SRGB",
        ),
    };
    let mut c = matmul(&xyz, &SRGB_TO_XYZ);
    for row in c.iter_mut() {
        let s: f64 = row.iter().sum();
        row.iter_mut().for_each(|v| *v /= s); // neutral -> neutral
    }
    let i = inv3(&c);
    let mut o = [0f32; 9];
    for r in 0..3 {
        for k in 0..3 {
            o[3 * r + k] = i[r][k] as f32;
        }
    }
    (o, note)
}

/// Decode `path` and rotate its CFA phase onto `cfa` (period x period, 0/1/2 =
/// R/G/B, the manifest's table). `crop` is (x, y, w, h) in the phase-corrected
/// image, snapped to whole periods.
pub fn load(
    path: &str,
    cfa: &[u32],
    period: usize,
    crop: Option<[usize; 4]>,
    exposure: f32,
    auto_exposure: bool,
) -> Result<Raw, String> {
    let img = rawler::decode_file(path).map_err(|e| format!("{path}: {e}"))?;
    let RawPhotometricInterpretation::Cfa(cfg) = &img.photometric else {
        return Err(format!("{path}: not a CFA sensor"));
    };
    if img.cpp != 1 {
        return Err(format!("{path}: {} components per pixel, want 1", img.cpp));
    }
    let p = period;
    if cfg.cfa.width == 0 || p % cfg.cfa.width != 0 || p % cfg.cfa.height != 0 {
        return Err(format!(
            "{path}: CFA {} ({}x{}) does not repeat every {p} px",
            cfg.cfa.name, cfg.cfa.width, cfg.cfa.height
        ));
    }
    // the usable area, in full-sensor coordinates
    let area = img.active_area.or(img.crop_area);
    let (x0, y0, vw, vh) = match area {
        Some(r) => (r.p.x, r.p.y, r.d.w, r.d.h),
        None => (0, 0, img.width, img.height),
    };
    let colour = |y: usize, x: usize| cfg.cfa.color_at(y, x);
    if (0..p)
        .flat_map(|y| (0..p).map(move |x| (y, x)))
        .any(|(y, x)| colour(y0 + y, x0 + x) > 2)
    {
        return Err(format!(
            "{path}: CFA {} is not an RGB pattern",
            cfg.cfa.name
        ));
    }
    // A CFA is only defined up to where you start reading it: find the shift
    // that puts the file's pattern on the manifest's.
    let off = (0..p * p).map(|i| (i / p, i % p)).find(|&(dy, dx)| {
        (0..p).all(|y| (0..p).all(|x| colour(y0 + dy + y, x0 + dx + x) as u32 == cfa[y * p + x]))
    });
    let Some((dy, dx)) = off else {
        let shown: Vec<String> = (0..p)
            .map(|y| {
                (0..p)
                    .map(|x| ['R', 'G', 'B'][colour(y0 + y, x0 + x)])
                    .collect()
            })
            .collect();
        return Err(format!(
            "{path}: its CFA ({}) is not the model's in any phase; \
                            use a model trained for this sensor's CFA",
            shown.join(" ")
        ));
    };

    // phase-corrected origin and size, then the optional crop, whole periods
    let (mut ox, mut oy) = (x0 + dx, y0 + dy);
    let (mut w, mut h) = (vw - dx, vh - dy);
    if let Some([cx, cy, cw, ch]) = crop {
        let (cx, cy) = (cx / p * p, cy / p * p);
        if cx >= w || cy >= h {
            return Err(format!(
                "--crop origin {cx},{cy} is outside the {w}x{h} image"
            ));
        }
        ox += cx;
        oy += cy;
        w = (cw.div_ceil(p) * p).min(w - cx);
        h = (ch.div_ceil(p) * p).min(h - cy);
    }
    let (w, h) = (w / p * p, h / p * p);

    // Black/white normalisation, per site. No clamp at 0: the noise floor
    // lives below black, and estimate_noise needs it.
    let bl = &img.blacklevel;
    let black_at = |y: usize, x: usize| -> f32 {
        let n = bl.levels.len();
        if n == 0 {
            0.0
        } else if n == 1 {
            bl.levels[0].as_f32()
        } else {
            let (bw, bh) = (bl.width.max(1), bl.height.max(1));
            bl.levels[((y % bh) * bw + x % bw) % n].as_f32()
        }
    };
    let white = img.whitelevel.0.first().copied().unwrap_or(u16::MAX as u32) as f32;
    let get: Box<dyn Fn(usize) -> f32> = match &img.data {
        RawImageData::Integer(v) => Box::new(move |i| v[i] as f32),
        RawImageData::Float(v) => Box::new(move |i| v[i]),
    };
    let mut m = vec![0f32; w * h];
    let (mut bmin, mut bmax) = (f32::MAX, f32::MIN);
    for y in 0..h {
        for x in 0..w {
            let (sy, sx) = (oy + y, ox + x);
            let b = black_at(sy, sx);
            bmin = bmin.min(b);
            bmax = bmax.max(b);
            m[y * w + x] = (get(sy * img.width + sx) - b) / (white - b);
        }
    }

    let wb = img.wb_coeffs;
    let (gain, wb_note) =
        if wb[0] > 0.0 && wb[1] > 0.0 && wb[2] > 0.0 && wb[..3].iter().all(|v| v.is_finite()) {
            ([wb[0] / wb[1], 1.0, wb[2] / wb[1]], "camera WB")
        } else {
            ([1.0, 1.0, 1.0], "no WB in file; gains 1,1,1")
        };
    let (srgb_from_cam, cam_note) = cam_matrix(&img);

    let mut exposure = exposure;
    if auto_exposure {
        // green's 99.9th percentile to 0.9
        let mut g: Vec<f32> = (0..h * w)
            .filter(|i| cfa[(i / w % p) * p + i % w % p] == 1)
            .map(|i| m[i])
            .collect();
        let k = ((g.len() - 1) as f64 * 0.999) as usize;
        let hi = *g.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1;
        exposure = 0.9 / hi.max(1e-6);
    }
    if exposure != 1.0 {
        m.iter_mut().for_each(|v| *v *= exposure);
    }

    let notes = format!(
        "{} {} {}x{} (used {w}x{h} at {ox},{oy}), CFA {} phase ({dy}, {dx}); {wb_note} {:.4?}; {cam_note}; \
         black {bmin:.0}-{bmax:.0} white {white:.0}; exposure x{exposure:.3}",
        img.clean_make, img.clean_model, img.width, img.height, cfg.cfa.name, gain);
    Ok(Raw {
        h,
        w,
        mosaic: m,
        gain,
        srgb_from_cam,
        notes,
    })
}
