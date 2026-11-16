// SPDX-License-Identifier: GPL-3.0-or-later
//
// Frank Markesteijn's demosaic for X-Trans sensors, a port of the CPU path of
// darktable's `markesteijn.c` (`xtrans_markesteijn_interpolate`, Copyright (C)
// 2010-2026 darktable developers; adapted there from dcraw 9.20), with 3
// passes (8 directions). See the module docs for the licence this brings
// with it.
//
// Differences from darktable: it has no final `_vng_lininterpolate` over
// the outermost pixels (the mirrored tile edges already cover them), and it
// stores each tile's results per tile row instead of in a shared output.

/// darktable's DT_MARKESTEIJN_TS: the tile size.
const TS: usize = 122;
const PAD_G1_G3: i32 = 3;
const PAD_G_INTERP: i32 = 3;
const PAD_G_RECALC: i32 = 6;
const HOMOGENITY_EPS: f32 = 1e-10;
const SOLITARY_EPS: f32 = 1e-12;
const PASSES: usize = 3;
const NDIR: usize = 4 << (PASSES > 1) as usize;
/// Extra passes propagate errors at the edges, hence the padding
const PAD_TILE: i32 = 17;

const ORTH: [i32; 12] = [1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1];
const PATT: [[i32; 16]; 2] = [
    [0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0],
    [0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1],
];

fn sq(x: f32) -> f32 {
    x * x
}

fn clamps(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// The X-Trans pattern, indexable at any (also negative) coordinates.
struct Pattern<'a>(&'a [u32]);

impl Pattern<'_> {
    fn fc(&self, row: i32, col: i32) -> usize {
        self.0[row.rem_euclid(6) as usize * 6 + col.rem_euclid(6) as usize] as usize
    }
}

/// The tile's interleaved RGB planes of the 8 directions: pixel `p` (as
/// `d * TS * TS + y * TS + x`, which offsets may leave) channel `c`.
struct Tiles(Vec<f32>);

impl Tiles {
    fn get(&self, p: i32, c: usize) -> f32 {
        self.0[p as usize * 3 + c]
    }
    fn set(&mut self, p: i32, c: usize, v: f32) {
        self.0[p as usize * 3 + c] = v;
    }
}

/// The input of a demosaic: `width` x `height` samples and the 6x6 pattern.
struct Frame<'a> {
    data: &'a [f32],
    width: usize,
    height: usize,
}

/// The demosaic of an X-Trans frame of `width` x `height` samples, `colors`
/// its 6x6 pattern (0 = R, 1 = G, 2 = B): interleaved R, G, B (as sampled,
/// no white balance), `None` if cancelled.
pub fn demosaic(
    data: &[f32],
    width: usize,
    height: usize,
    colors: &[u32],
    cancel: &std::sync::atomic::AtomicBool,
) -> Option<Vec<f32>> {
    let raw = &Frame {
        data,
        width,
        height,
    };
    let height = height as i32;
    let xtrans = Pattern(colors);
    let ts = TS as i32;

    // Map a green hexagon around each non-green pixel and vice versa
    let mut allhex = [[[0i32; 8]; 3]; 3];
    // the offset in the sensor matrix of the solitary green pixels
    let (mut sgrow, mut sgcol) = (0i32, 0i32);
    for row in 0..3 {
        for col in 0..3 {
            let mut ng = 0;
            for d in (0..10).step_by(2) {
                let g = (xtrans.fc(row, col) == 1) as i32;
                if xtrans.fc(row + ORTH[d], col + ORTH[d + 2]) == 1 {
                    ng = 0;
                } else {
                    ng += 1;
                }
                // four non-green pixels adjacent in cardinal directions: the solitary green pixel
                if ng == 4 {
                    sgrow = row;
                    sgcol = col;
                }
                if ng == g + 1 {
                    for c in 0..8usize {
                        let v = ORTH[d] * PATT[g as usize][c * 2]
                            + ORTH[d + 1] * PATT[g as usize][c * 2 + 1];
                        let h = ORTH[d + 2] * PATT[g as usize][c * 2]
                            + ORTH[d + 3] * PATT[g as usize][c * 2 + 1];
                        allhex[row as usize][col as usize][c ^ ((g * 2) & d as i32) as usize] =
                            h + v * ts;
                    }
                }
            }
        }
    }

    let step = ts - 2 * PAD_TILE;
    let tile_rows = ((height + step - 1) / step) as usize;
    let results = super::par_map(tile_rows, cancel, |k| {
        let top = -PAD_TILE + k as i32 * step;
        tile_row(raw, &xtrans, &allhex, sgrow, sgcol, top)
    })?;
    // each tile row made the image rows from `top + PAD_TILE` on
    Some(results.concat())
}

/// All the tiles of the row starting at image row `top`: the interleaved RGB
/// of the image rows [top + PAD_TILE, min(top + TS, height + PAD_TILE) - PAD_TILE).
fn tile_row(
    raw: &Frame,
    xtrans: &Pattern,
    allhex: &[[[i32; 8]; 3]; 3],
    sgrow: i32,
    sgcol: i32,
    top: i32,
) -> Vec<f32> {
    let (width, height) = (raw.width as i32, raw.height as i32);
    let ts = TS as i32;
    let tsp = ts * ts; // pixels per tile
    let ndir = NDIR;
    let dir = [1, ts, ts + 1, ts - 1];
    let hexmap =
        |row: i32, col: i32| &allhex[((row + 600) % 3) as usize][((col + 600) % 3) as usize];

    let mut rgb = Tiles(vec![0f32; ndir * TS * TS * 3]);
    let mut yuv = vec![0f32; 3 * TS * TS];
    let mut drv = vec![0f32; ndir * TS * TS];
    let mut gmin = vec![0f32; TS * TS];
    let mut gmax = vec![0f32; TS * TS];
    let mut homo = vec![0u8; ndir * TS * TS];
    let mut homosum = vec![0u8; ndir * TS * TS];

    let mrow_out = (top + ts).min(height + PAD_TILE) - PAD_TILE;
    let out_rows = (mrow_out - (top + PAD_TILE)).max(0) as usize;
    let mut out = vec![0f32; out_rows * raw.width * 3];

    let mut left = -PAD_TILE;
    while left < width - PAD_TILE {
        let mut mrow = (top + ts).min(height + PAD_TILE);
        let mut mcol = (left + ts).min(width + PAD_TILE);

        // Copy the tile in. Beyond the image edge, mirror a border pixel, or
        // interpolate if the mirror pixel is a different colour.
        let translate = |n: i32, size: i32| if n >= size { 2 * size - n - 2 } else { n.abs() };
        for row in top..mrow {
            for col in left..mcol {
                let p = (row - top) * ts + (col - left);
                let f = xtrans.fc(row, col);
                for c in 0..3 {
                    let v = if c != f {
                        0.0
                    } else if col >= 0 && row >= 0 && col < width && row < height {
                        raw.data[(width * row + col) as usize].max(0.0)
                    } else {
                        let (cy, cx) = (translate(row, height), translate(col, width));
                        if c == xtrans.fc(cy, cx) {
                            raw.data[(width * cy + cx) as usize].max(0.0)
                        } else {
                            let (mut sum, mut count) = (0.0f32, 0u8);
                            for y in row - 1..=row + 1 {
                                for x in col - 1..=col + 1 {
                                    let (yy, xx) = (translate(y, height), translate(x, width));
                                    if xtrans.fc(yy, xx) == c {
                                        sum += raw.data[(width * yy + xx) as usize].max(0.0);
                                        count += 1;
                                    }
                                }
                            }
                            sum / count as f32
                        }
                    };
                    rgb.set(p, c, v);
                }
            }
        }
        // duplicate rgb[0] to rgb[1], rgb[2] and rgb[3]
        for c in 1..4 {
            rgb.0.copy_within(0..tsp as usize * 3, c * tsp as usize * 3);
        }

        // Set green1 and green3 to the minimum and maximum allowed values:
        // for each red / blue pair, the min and max of the greens around it
        let mut row = top + PAD_G1_G3;
        while row < mrow - PAD_G1_G3 {
            // max 0 marks a new pair, which needs a new min / max
            let (mut min, mut max) = (f32::MAX, 0.0f32);
            let mut col = left + PAD_G1_G3;
            while col < mcol - PAD_G1_G3 {
                if xtrans.fc(row, col) == 1 {
                    min = f32::MAX;
                    max = 0.0;
                    col += 1;
                    continue;
                }
                if max == 0.0 {
                    let p = (row - top) * ts + (col - left);
                    let hex = hexmap(row, col);
                    for c in 0..6 {
                        let val = rgb.get(p + hex[c], 1);
                        if min > val {
                            min = val;
                        }
                        if max < val {
                            max = val;
                        }
                    }
                }
                let g = ((row - top) * ts + (col - left)) as usize;
                gmin[g] = min;
                gmax[g] = max;
                // vertical red / blue pairs
                match (row - sgrow) % 3 {
                    // hop down a row to the second pixel in the vertical pair
                    1 => {
                        if row < mrow - 4 {
                            row += 1;
                            col -= 1;
                        }
                    }
                    // then, if not done with the row, hop up and right to the next pair
                    2 => {
                        min = f32::MAX;
                        max = 0.0;
                        col += 2;
                        if col < mcol - 4 && row > top + 3 {
                            row -= 1;
                        }
                    }
                    _ => {}
                }
                col += 1;
            }
            row += 1;
        }

        // Interpolate green horizontally, vertically and along both diagonals
        for row in top + PAD_G_INTERP..mrow - PAD_G_INTERP {
            for col in left + PAD_G_INTERP..mcol - PAD_G_INTERP {
                let f = xtrans.fc(row, col);
                if f == 1 {
                    continue;
                }
                let p = (row - top) * ts + (col - left);
                let hex = hexmap(row, col);
                let px = |o: i32, c: usize| rgb.get(p + o, c);
                let mut color = [0f32; 4];
                color[0] = 0.6796875 * (px(hex[1], 1) + px(hex[0], 1))
                    - 0.1796875 * (px(2 * hex[1], 1) + px(2 * hex[0], 1));
                color[1] = 0.87109375 * px(hex[3], 1)
                    + px(hex[2], 1) * 0.13
                    + 0.359375 * (px(0, f) - px(-hex[2], f));
                for c in 0..2 {
                    color[2 + c] = 0.640625 * px(hex[4 + c], 1)
                        + 0.359375 * px(-2 * hex[4 + c], 1)
                        + 0.12890625
                            * (2.0 * px(0, f) - px(3 * hex[4 + c], f) - px(-3 * hex[4 + c], f));
                }
                let g = p as usize;
                for c in 0..4 {
                    let tile = c ^ ((row - sgrow) % 3 == 0) as usize;
                    rgb.set(tile as i32 * tsp + p, 1, clamps(color[c], gmin[g], gmax[g]));
                }
            }
        }

        // `base`: the first of the 4 tiles a pass works on (pass 1 works on 4..8)
        let mut base = 0i32;
        for pass in 0..PASSES {
            if pass == 1 {
                // copy rgb[0..4] into rgb[4..8] and process that second set
                rgb.0
                    .copy_within(0..4 * tsp as usize * 3, 4 * tsp as usize * 3);
                base = 4 * tsp;
            }

            // Recalculate green from interpolated values of closer pixels
            if pass > 0 {
                for row in top + PAD_G_RECALC..mrow - PAD_G_RECALC {
                    for col in left + PAD_G_RECALC..mcol - PAD_G_RECALC {
                        let f = xtrans.fc(row, col);
                        if f == 1 {
                            continue;
                        }
                        let hex = hexmap(row, col);
                        let p = (row - top) * ts + (col - left);
                        for d in 3..6usize {
                            let r =
                                base + ((d as i32 - 2) ^ ((row - sgrow) % 3 == 0) as i32) * tsp + p;
                            let h = hex[d];
                            let val = rgb.get(r - 2 * h, 1) + 2.0 * rgb.get(r + h, 1)
                                - rgb.get(r - 2 * h, f)
                                - 2.0 * rgb.get(r + h, f)
                                + 3.0 * rgb.get(r, f);
                            let g = p as usize;
                            rgb.set(r, 1, clamps(val / 3.0, gmin[g], gmax[g]));
                        }
                    }
                }
            }

            // Interpolate red and blue values for solitary green pixels
            let pad_rb_g = if PASSES == 1 { 6 } else { 5 };
            let mut row = (top - sgrow + pad_rb_g + 2) / 3 * 3 + sgrow;
            while row < mrow - pad_rb_g {
                let mut col = (left - sgcol + pad_rb_g + 2) / 3 * 3 + sgcol;
                while col < mcol - pad_rb_g {
                    let mut rfx = base + (row - top) * ts + (col - left);
                    let mut h = xtrans.fc(row, col + 1);
                    let mut diff = [0f32; 6];
                    // interpolated colour: red / blue, then the pass; double the result
                    let mut color = [[0f32; 6]; 2];
                    // Six passes, alternating horizontal / vertical (i), starting with
                    // red or blue (h), whichever is closest
                    let mut i = 1i32;
                    for d in 0..6usize {
                        // look 1 and 2 pixels away from the solitary green to red then blue, or blue then red
                        for c in 0..2 {
                            // the greens' rate of change between the pixel and the interpolated ones
                            let o = i << c;
                            let g =
                                2.0 * rgb.get(rfx, 1) - rgb.get(rfx + o, 1) - rgb.get(rfx - o, 1);
                            color[(h != 0) as usize][d] =
                                g + rgb.get(rfx + o, h) + rgb.get(rfx - o, h);
                            // diff is the slope for both red and blue differentials in this direction;
                            // for the 2nd and 3rd horizontal + vertical passes, the sum of both
                            if d > 1 {
                                diff[d] += sq(rgb.get(rfx + o, 1)
                                    - rgb.get(rfx - o, 1)
                                    - rgb.get(rfx + o, h)
                                    + rgb.get(rfx - o, h))
                                    + sq(g);
                            }
                            h ^= 2;
                        }
                        if d < 2 || d & 1 == 1 {
                            // output for passes 0, 1, 3, 5: 0 and 1 are horizontal / vertical, 3 and 5 the better direction
                            let d_out =
                                d - (d > 1 && diff[d - 1] + SOLITARY_EPS < diff[d]) as usize;
                            rgb.set(rfx, 0, color[0][d_out] / 2.0);
                            rgb.set(rfx, 2, color[1][d_out] / 2.0);
                            rfx += tsp;
                        }
                        i ^= ts ^ 1;
                        h ^= 2;
                    }
                    col += 3;
                }
                row += 3;
            }

            // Interpolate red for blue pixels and vice versa
            let pad_rb_br = if PASSES == 1 { 6 } else { 5 };
            for row in top + pad_rb_br..mrow - pad_rb_br {
                for col in left + pad_rb_br..mcol - pad_rb_br {
                    let f = 2 - xtrans.fc(row, col);
                    if f == 1 {
                        continue;
                    }
                    let mut rfx = base + (row - top) * ts + (col - left);
                    let c = if (row - sgrow) % 3 != 0 { ts } else { 1 };
                    let h = 3 * (c ^ ts ^ 1);
                    for d in 0..4i32 {
                        let g0 = rgb.get(rfx, 1);
                        let i = if d > 1
                            || ((d ^ c) & 1) != 0
                            || ((g0 - rgb.get(rfx + c, 1)).abs() + (g0 - rgb.get(rfx - c, 1)).abs())
                                < 2.0
                                    * ((g0 - rgb.get(rfx + h, 1)).abs()
                                        + (g0 - rgb.get(rfx - h, 1)).abs())
                        {
                            c
                        } else {
                            h
                        };
                        let v = (rgb.get(rfx + i, f) + rgb.get(rfx - i, f) + 2.0 * g0
                            - rgb.get(rfx + i, 1)
                            - rgb.get(rfx - i, 1))
                            / 2.0;
                        rgb.set(rfx, f, v);
                        rfx += tsp;
                    }
                }
            }

            // Fill in red and blue for 2x2 blocks of green
            let pad_g22 = if PASSES == 1 { 8 } else { 4 };
            for row in top + pad_g22..mrow - pad_g22 {
                if (row - sgrow) % 3 == 0 {
                    continue;
                }
                for col in left + pad_g22..mcol - pad_g22 {
                    if (col - sgcol) % 3 == 0 {
                        continue;
                    }
                    let hex = hexmap(row, col);
                    let mut rfx = base + (row - top) * ts + (col - left);
                    for d in (0..ndir).step_by(2) {
                        if hex[d] + hex[d + 1] != 0 {
                            let g = 3.0 * rgb.get(rfx, 1)
                                - 2.0 * rgb.get(rfx + hex[d], 1)
                                - rgb.get(rfx + hex[d + 1], 1);
                            for c in [0, 2] {
                                let v = (g
                                    + 2.0 * rgb.get(rfx + hex[d], c)
                                    + rgb.get(rfx + hex[d + 1], c))
                                    / 3.0;
                                rgb.set(rfx, c, v);
                            }
                        } else {
                            let g = 2.0 * rgb.get(rfx, 1)
                                - rgb.get(rfx + hex[d], 1)
                                - rgb.get(rfx + hex[d + 1], 1);
                            for c in [0, 2] {
                                let v =
                                    (g + rgb.get(rfx + hex[d], c) + rgb.get(rfx + hex[d + 1], c))
                                        / 2.0;
                                rgb.set(rfx, c, v);
                            }
                        }
                        rfx += tsp;
                    }
                }
            }
        } // end of the multipass loop

        // from here on, work within the tile: don't offset mrow / mcol by the tile's top / left
        mrow -= top;
        mcol -= left;

        // Convert to a perceptual colourspace (BT.2020 YPbPr) and differentiate in all directions
        for d in 0..ndir {
            let pad_yuv = if PASSES == 1 { 8 } else { 13 };
            for row in pad_yuv..mrow - pad_yuv {
                for col in pad_yuv..mcol - pad_yuv {
                    let p = (d as i32 * tsp + row * ts + col) as i32;
                    let (r, g, b) = (rgb.get(p, 0), rgb.get(p, 1), rgb.get(p, 2));
                    let y = 0.2627 * r + 0.6780 * g + 0.0593 * b;
                    let q = (row * ts + col) as usize;
                    yuv[q] = y;
                    yuv[TS * TS + q] = (b - y) * 0.56433;
                    yuv[2 * TS * TS + q] = (r - y) * 0.67815;
                }
            }
            let f = dir[d & 3];
            let pad_drv = if PASSES == 1 { 9 } else { 14 };
            for row in pad_drv..mrow - pad_drv {
                for col in pad_drv..mcol - pad_drv {
                    let q = row * ts + col;
                    let plane = |k: usize, o: i32| yuv[k * TS * TS + (q + o) as usize];
                    drv[d * TS * TS + q as usize] =
                        sq(2.0 * plane(0, 0) - plane(0, f) - plane(0, -f))
                            + sq(2.0 * plane(1, 0) - plane(1, f) - plane(1, -f))
                            + sq(2.0 * plane(2, 0) - plane(2, f) - plane(2, -f));
                }
            }
        }

        // Build the homogeneity maps from the derivatives
        homo.fill(0);
        let pad_homo = if PASSES == 1 { 10 } else { 15 };
        for row in pad_homo..mrow - pad_homo {
            for col in pad_homo..mcol - pad_homo {
                let q = (row * ts + col) as usize;
                let mut tr = f32::MAX;
                for d in 0..ndir {
                    if tr > drv[d * TS * TS + q] {
                        tr = drv[d * TS * TS + q];
                    }
                }
                tr = (8.0 * tr).max(HOMOGENITY_EPS);
                for d in 0..ndir {
                    for v in -1..=1 {
                        for h in -1..=1 {
                            let n = (d * TS * TS) as i32 + (row + v) * ts + col + h;
                            homo[d * TS * TS + q] += (drv[n as usize] <= tr) as u8;
                        }
                    }
                }
            }
        }

        // Build the 5x5 sum of the homogeneity maps for each pixel and direction
        for d in 0..ndir {
            for row in PAD_TILE..mrow - PAD_TILE {
                // start before the first column where homo[col + 2] != 0, so homosum is 0 there
                let mut col = PAD_TILE - 5;
                let mut v5sum = [0u8; 5];
                let at = |row: i32, col: i32| d * TS * TS + (row * ts + col) as usize;
                homosum[at(row, col)] = 0;
                // by rolling through column sums
                col += 1;
                while col < mcol - PAD_TILE {
                    let mut colsum = 0u8;
                    for v in -2..=2 {
                        colsum = colsum.wrapping_add(homo[at(row + v, col + 2)]);
                    }
                    homosum[at(row, col)] = homosum[at(row, col - 1)]
                        .wrapping_sub(v5sum[(col % 5) as usize])
                        .wrapping_add(colsum);
                    v5sum[(col % 5) as usize] = colsum;
                    col += 1;
                }
            }
        }

        // Average the most homogeneous pixels for the final result
        for row in PAD_TILE..mrow - PAD_TILE {
            for col in PAD_TILE..mcol - PAD_TILE {
                let q = (row * ts + col) as usize;
                let mut hm = [0u8; 8];
                let mut maxval = 0u8;
                for d in 0..ndir {
                    hm[d] = homosum[d * TS * TS + q];
                    maxval = maxval.max(hm[d]);
                }
                maxval -= maxval >> 3;
                for d in 0..ndir - 4 {
                    if hm[d] < hm[d + 4] {
                        hm[d] = 0;
                    } else if hm[d] > hm[d + 4] {
                        hm[d + 4] = 0;
                    }
                }
                let mut avg = [0f32; 4];
                for d in 0..ndir {
                    if hm[d] >= maxval {
                        for c in 0..3 {
                            avg[c] += rgb.get(d as i32 * tsp + row * ts + col, c);
                        }
                        avg[3] += 1.0;
                    }
                }
                // from the first image row this tile row makes
                let y = (row - PAD_TILE) as usize;
                let x = (col + left) as usize;
                for c in 0..3 {
                    out[(y * raw.width + x) * 3 + c] = (avg[c] / avg[3]).max(0.0);
                }
            }
        }
        left += ts - 2 * PAD_TILE;
    }
    out
}
