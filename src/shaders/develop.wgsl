// The per-pixel develop shared by the preview and the export: sampling the
// developed image (camera RGB), then the colour matrix, local contrast and
// the user's adjustments to display RGB. src/develop.rs joins this file
// in front of each shader that uses it. src/tone.rs describes the tone
// mapping and computes its curve's numbers (tone::Curve).

// Where the image's layers sit in its texture array (source::GridUniform).
// Layer (column, row) holds image pixels from (column, row) * step - overlap
// on, `layer` texels a side; beyond the image the edge pixels repeat.
struct Grid {
    // Image size in pixels.
    size: vec2<f32>,
    // Image pixels from one layer's start to the next one's.
    step: vec2<f32>,
    // Layer size in texels.
    layer: vec2<f32>,
    // Layers per row.
    columns: u32,
    // Pixels each layer shares with its neighbours, per side.
    overlap: f32,
}

// The layer of image point `p` (pixels): x, y = its column and row.
fn grid_cell(g: Grid, p: vec2<f32>) -> vec2<f32> {
    let last = ceil(g.size / g.step) - 1.0;
    return clamp(floor(p / g.step), vec2<f32>(0.0), last);
}

fn grid_layer(g: Grid, cell: vec2<f32>) -> u32 {
    return u32(cell.y) * g.columns + u32(cell.x);
}

// The lens correction drawn on top of the image (preview::LensUniform): the
// corrected image's point p reads the source at centre + (p - centre) * s,
// with s from the table at p's radius, and gains the vignetting there.
struct Lens {
    // Image centre in pixels.
    centre: vec2<f32>,
    // 1 / half-diagonal in pixels: the normalised radius is distance × this.
    inv_half_diagonal: f32,
    // Radius of the last table entry.
    lut_max: f32,
}

// The table (lens::LensCorrection::lut, one texel per entry) at normalised
// radius `rho`: source radius / output radius, red CA, blue CA, vignetting
// gain.
fn lens_lookup(lut: texture_2d<f32>, l: Lens, rho: f32) -> vec4<f32> {
    let last = textureDimensions(lut).x - 1u;
    let t = clamp(rho / l.lut_max, 0.0, 1.0) * f32(last);
    let i = u32(floor(t));
    let a = textureLoad(lut, vec2<u32>(i, 0u), 0);
    let b = textureLoad(lut, vec2<u32>(min(i + 1u, last), 0u), 0);
    return mix(a, b, t - floor(t));
}

// Where a point of the corrected image reads the source: green (by which
// the footprint and mip level go), red and blue, which lateral CA moves
// along the radius, and the vignetting gain.
struct LensPoint {
    green: vec2<f32>,
    red: vec2<f32>,
    blue: vec2<f32>,
    gain: f32,
}

// For point `p` of the corrected image. The CA (the radial scale offset of
// red and blue against green, lens::Measured) and the gain belong to the
// source pixel, so they are looked up at the source radius.
fn lens_source(lut: texture_2d<f32>, l: Lens, p: vec2<f32>) -> LensPoint {
    let offset = p - l.centre;
    let source = offset * lens_lookup(lut, l, length(offset) * l.inv_half_diagonal).x;
    let at = lens_lookup(lut, l, length(source) * l.inv_half_diagonal);
    return LensPoint(
        l.centre + source,
        l.centre + source * (1.0 + at.y),
        l.centre + source * (1.0 + at.z),
        at.w,
    );
}

// A texel with `grain` times its alpha, the noise the network removed there
// (wgpu-run's place shader), added back to every channel.
fn grained(t: vec4<f32>, grain: f32) -> vec3<f32> {
    return t.rgb + grain * t.a;
}

// The texel at point `p` (pixels, texel i centred on i + 0.5) from mip
// `level`, bilinear (trilinear between levels).
fn fetch_source(tex: texture_2d_array<f32>, samp: sampler, g: Grid, p: vec2<f32>, level: f32) -> vec4<f32> {
    let cell = grid_cell(g, p);
    let uv = (p - cell * g.step + g.overlap) / g.layer;
    return textureSampleLevel(tex, samp, uv, grid_layer(g, cell), level);
}

// The image at point `p`, as fetch_source, with `grain` of its noise.
fn sample_source(tex: texture_2d_array<f32>, samp: sampler, g: Grid, p: vec2<f32>, level: f32, grain: f32) -> vec3<f32> {
    return grained(fetch_source(tex, samp, g, p, level), grain);
}

// The image through the lens correction `at`, as sample_source: each channel
// from its own point. The grain is green's, so it stays neutral.
fn sample_lens(tex: texture_2d_array<f32>, samp: sampler, g: Grid, at: LensPoint, level: f32, grain: f32) -> vec3<f32> {
    let t = fetch_source(tex, samp, g, at.green, level);
    var rgb = t.rgb;
    if (any(at.red != at.green)) {
        rgb.r = fetch_source(tex, samp, g, at.red, level).r;
    }
    if (any(at.blue != at.green)) {
        rgb.b = fetch_source(tex, samp, g, at.blue, level).b;
    }
    return rgb + grain * t.a;
}

// The texel at point `p` (pixels) at full size: Catmull-Rom from nine
// bilinear fetches, as lens.wgsl. The overlap covers its 4x4 footprint.
fn fetch_source_cubic(tex: texture_2d_array<f32>, samp: sampler, g: Grid, p: vec2<f32>) -> vec4<f32> {
    let cell = grid_cell(g, p);
    let layer = grid_layer(g, cell);
    // Image point of the layer's top left corner.
    let origin = cell * g.step - g.overlap;
    let c1 = floor(p - 0.5) + 0.5;
    let f = p - c1;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let t0 = (c1 - 1.0 - origin) / g.layer;
    let t12 = (c1 + w2 / w12 - origin) / g.layer;
    let t3 = (c1 + 2.0 - origin) / g.layer;

    var sum = vec4<f32>(0.0);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t0.x, t0.y), layer, 0.0) * (w0.x * w0.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t12.x, t0.y), layer, 0.0) * (w12.x * w0.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t3.x, t0.y), layer, 0.0) * (w3.x * w0.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t0.x, t12.y), layer, 0.0) * (w0.x * w12.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t12.x, t12.y), layer, 0.0) * (w12.x * w12.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t3.x, t12.y), layer, 0.0) * (w3.x * w12.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t0.x, t3.y), layer, 0.0) * (w0.x * w3.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t12.x, t3.y), layer, 0.0) * (w12.x * w3.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t3.x, t3.y), layer, 0.0) * (w3.x * w3.y);
    return sum;
}

// sample_lens at full size, bicubic.
fn sample_lens_cubic(tex: texture_2d_array<f32>, samp: sampler, g: Grid, at: LensPoint, grain: f32) -> vec3<f32> {
    let t = fetch_source_cubic(tex, samp, g, at.green);
    var rgb = t.rgb;
    if (any(at.red != at.green)) {
        rgb.r = fetch_source_cubic(tex, samp, g, at.red).r;
    }
    if (any(at.blue != at.green)) {
        rgb.b = fetch_source_cubic(tex, samp, g, at.blue).b;
    }
    return rgb + grain * t.a;
}

// Sharpening: an unsharp mask on the denoised image's luma, in display-ish
// (gamma) values so a halo is as strong on the dark side of an edge as on the
// bright one. One slider moves two things: the gain, and the radius, which
// grows with it by mixing the detail of a fine and a coarse Gaussian
// (sigma in image pixels), so small amounts stay at the pixel level and large
// ones do not just amplify pixel noise. The detail is weighted by its own
// size (flat areas get little), and the result may not overshoot the 3x3
// neighbourhood's range by more than SHARPEN_OVERSHOOT of it, which
// removes most halos. The taps are the denoised image with no grain, so the
// grain is not sharpened.
const SHARPEN_GAIN: f32 = 2.0;
const SHARPEN_SIGMA_FINE: f32 = 0.6;
const SHARPEN_SIGMA_COARSE: f32 = 1.5;
const SHARPEN_OVERSHOOT: f32 = 0.2;

// The factor to scale the pixel at `at` by (rgb, linear, camera), for
// `amount` (Edit::sharpness, 0 … 1). Taps are image pixels apart, on the
// green point, at mip `level`.
fn sharpen(e: Edit, tex: texture_2d_array<f32>, samp: sampler, g: Grid, at: LensPoint, level: f32, amount: f32) -> f32 {
    // Luma of linear sRGB from camera RGB, as one row.
    let row = LUMA.x * e.cam_to_srgb[0].xyz + LUMA.y * e.cam_to_srgb[1].xyz + LUMA.z * e.cam_to_srgb[2].xyz;
    let scale = exp2(e.exposure) / e.limits.z;
    let fine = -0.5 / (SHARPEN_SIGMA_FINE * SHARPEN_SIGMA_FINE);
    let coarse = -0.5 / (SHARPEN_SIGMA_COARSE * SHARPEN_SIGMA_COARSE);

    var centre = 0.0;
    var low = 1e9;
    var high = -1e9;
    var fine_sum = 0.0;
    var fine_weight = 0.0;
    var coarse_sum = 0.0;
    var coarse_weight = 0.0;
    // The 3x3 and the eight taps two pixels out along the axes and diagonals.
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let near = max(abs(dx), abs(dy)) <= 1;
            if (!near && (dx % 2 != 0 || dy % 2 != 0)) {
                continue;
            }
            let t = fetch_source(tex, samp, g, at.green + vec2<f32>(f32(dx), f32(dy)), level).rgb;
            let v = pow(max(dot(row, t) * scale, 1e-6), 1.0 / DISPLAY_GAMMA);
            let d2 = f32(dx * dx + dy * dy);
            let wf = exp(d2 * fine);
            let wc = exp(d2 * coarse);
            fine_sum += wf * v;
            fine_weight += wf;
            coarse_sum += wc * v;
            coarse_weight += wc;
            if (near) {
                low = min(low, v);
                high = max(high, v);
            }
            if (dx == 0 && dy == 0) {
                centre = v;
            }
        }
    }
    if (centre < 1e-3) {
        return 1.0;
    }
    let to_coarse = smoothstep(0.3, 1.0, amount);
    let detail = mix(centre - fine_sum / fine_weight, centre - coarse_sum / coarse_weight, to_coarse);
    let weight = 0.2 + 0.8 * smoothstep(0.002, 0.015, abs(detail));
    let slack = SHARPEN_OVERSHOOT * (high - low);
    let sharp = clamp(centre + SHARPEN_GAIN * amount * weight * detail, low - slack, high + slack);
    // The value is luma ^ (1 / gamma), so the luma scales by the ratio ^ gamma.
    return clamp(pow(max(sharp, 0.0) / centre, DISPLAY_GAMMA), 0.0, 4.0);
}

// The user's edits (develop::EditUniform).
struct Edit {
    // Camera RGB → linear sRGB, row-major.
    cam_to_srgb: array<vec4<f32>, 3>,
    // Tone sigmoid (tone::Curve): pivot x, pivot y, slope, toe power.
    curve: vec4<f32>,
    // Its toe and shoulder limits, the white point (tone::WHITE), then the
    // shoulder power.
    limits: vec4<f32>,
    // The colour mixer (color_edit::ColorMix::pack): per colour, hue shift,
    // chroma scale - 1, lightness shift and its own hue, in radians.
    colors: array<vec4<f32>, 8>,
    // The grading (color_edit::Grading::pack): per range, the Oklab (a, b)
    // offset of its tint.
    grade: array<vec4<f32>, 3>,
    // Exposure compensation in stops.
    exposure: f32,
    saturation: f32,
    // Local contrast amount (Adjustments::local_contrast).
    local: f32,
    // Black point (Adjustments::black_point), linear, subtracted before
    // exposure.
    black: f32,
    // The fraction of the removed noise put back (Adjustments::grain), as
    // sample_source's `grain`.
    grain: f32,
    // Sharpening amount (Adjustments::sharpness), as `sharpen`'s `amount`.
    sharpness: f32,
    // Bit 0: the mixer does something; bit 1: the grading does.
    edit_flags: u32,
    // Vibrance (Adjustments::vibrance) and the chroma it falls to half at.
    vibrance: f32,
    vibrance_range: f32,
    // Artificial grain (Adjustments::added_grain), as `add_grain`'s amount.
    added_grain: f32,
}

const DISPLAY_GAMMA: f32 = 2.4;

const LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);

// Linear → display-encoded, 0 at black and 1 at white and above
// (tone::encode). The floor keeps pow() away from 0, where it is undefined.
fn encode(e: Edit, c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c / e.limits.z, vec3<f32>(1e-20), vec3<f32>(1.0));
    return pow(x, vec3<f32>(1.0 / DISPLAY_GAMMA));
}

fn sigmoid(e: Edit, x: vec3<f32>) -> vec3<f32> {
    let z = e.curve.z * (x - e.curve.x);
    let limit = select(vec3<f32>(e.limits.y), vec3<f32>(e.limits.x), z < vec3<f32>(0.0));
    // A negative limit is the inverse curve (tone::Curve). The floors keep
    // pow() away from 0, where it is undefined.
    let power = select(vec3<f32>(e.limits.w), vec3<f32>(e.curve.w), z < vec3<f32>(0.0));
    let q = pow(max(abs(z) / abs(limit), vec3<f32>(1e-20)), power);
    let bend = max(1.0 + sign(limit) * q, vec3<f32>(1e-20));
    let y = e.curve.y + z / pow(bend, 1.0 / power);
    return clamp(y, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn cam_to_srgb(e: Edit, cam: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(e.cam_to_srgb[0].xyz, cam),
        dot(e.cam_to_srgb[1].xyz, cam),
        dot(e.cam_to_srgb[2].xyz, cam),
    );
}

// The colour edits (src/color_edit.rs, which has the maths in Rust).
const TAU = 6.28318531;
// Below this Oklab chroma the mixer fades out (HUE_FADE_CHROMA).
const HUE_FADE_CHROMA = 0.02;
// The Oklab lightness the shadows end and the highlights begin at (SHADOWS,
// HIGHLIGHTS); the midtones are what is left.
const SHADOWS = vec2<f32>(0.2, 0.6);
const HIGHLIGHTS = vec2<f32>(0.6, 0.9);

// `x` brought into 0 .. TAU.
fn wrap_turn(x: f32) -> f32 {
    return x - TAU * floor(x / TAU);
}

fn linear_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = pow(max(dot(vec3<f32>(0.4122214708, 0.5363325363, 0.0514459929), c), 0.0), 1.0 / 3.0);
    let m = pow(max(dot(vec3<f32>(0.2119034982, 0.6806995451, 0.1073969566), c), 0.0), 1.0 / 3.0);
    let s = pow(max(dot(vec3<f32>(0.0883024619, 0.2817188376, 0.6299787005), c), 0.0), 1.0 / 3.0);
    return vec3<f32>(
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    );
}

fn oklab_to_linear(lab: vec3<f32>) -> vec3<f32> {
    let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    return vec3<f32>(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    );
}

// The mixer's settings at `hue`: a smooth blend between the colours either
// side of it (`mix_at`).
fn mixer_at(e: Edit, hue: f32) -> vec3<f32> {
    let h = wrap_turn(hue);
    // How many colours are at or below this hue (they ascend), which says
    // between which two it is; past the last, it is between it and the first.
    var k = 0u;
    for (var i = 0u; i < 8u; i++) {
        if (h >= e.colors[i].w) {
            k = i + 1u;
        }
    }
    let a = e.colors[(k + 7u) % 8u];
    let b = e.colors[k % 8u];
    let t = wrap_turn(h - a.w) / wrap_turn(b.w - a.w);
    return mix(a.xyz, b.xyz, smoothstep(0.0, 1.0, t));
}

// The colour edits applied to a linear sRGB colour: the mixer, then the
// grading's offset, both from the colour as it came in.
fn apply_color_edits(e: Edit, linear: vec3<f32>) -> vec3<f32> {
    let lab = linear_to_oklab(linear);
    let chroma = length(lab.yz);
    var hue = atan2(lab.z, lab.y);
    var c = chroma;
    var lightness = lab.x;
    var offset = vec2<f32>(0.0);
    if ((e.edit_flags & 1u) != 0u) {
        let fade = smoothstep(0.0, HUE_FADE_CHROMA, chroma);
        let found = mixer_at(e, hue);
        hue += found.x * fade;
        c = max(chroma * (1.0 + found.y * fade), 0.0);
        lightness += found.z * fade;
    }
    if ((e.edit_flags & 2u) != 0u) {
        let shadows = 1.0 - smoothstep(SHADOWS.x, SHADOWS.y, lab.x);
        let highlights = smoothstep(HIGHLIGHTS.x, HIGHLIGHTS.y, lab.x);
        offset = e.grade[0].xy * shadows
            + e.grade[1].xy * (1.0 - shadows - highlights)
            + e.grade[2].xy * highlights;
    }
    return oklab_to_linear(vec3<f32>(lightness, c * cos(hue) + offset.x, c * sin(hue) + offset.y));
}

// Camera RGB → linear display RGB in 0..1: the colour matrix, local
// contrast, the black point, then the tone mapping src/tone.rs describes. `base` is the local contrast base layer
// (local.rs) at this pixel, in stops; it is not read when `e.local` is 0.
// The display-encoded luma `develop` last fed the tone curve.
var<private> curve_input_luma: f32 = 0.0;

// Three independent standard normal samples from a pixel's integer
// coordinates (PCG3D hash, then Box-Muller): the same at a pixel whatever
// the view, so the preview, the histogram and the export agree.
fn pcg3d(p: vec3<u32>) -> vec3<u32> {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

fn gaussian3(pixel: vec2<i32>) -> vec3<f32> {
    let a = pcg3d(bitcast<vec3<u32>>(vec3<i32>(pixel, 0)));
    let b = pcg3d(bitcast<vec3<u32>>(vec3<i32>(pixel, 1)));
    // Uniform in (0, 1): the 24 high bits, half a step in.
    let u1 = (vec2<f32>(vec2<u32>(a.x >> 8u, b.x >> 8u)) + 0.5) / 16777216.0;
    let u2 = (vec2<f32>(vec2<u32>(a.y >> 8u, b.y >> 8u)) + 0.5) / 16777216.0;
    let r = sqrt(-2.0 * log(u1));
    let t = 6.28318531 * u2;
    return vec3<f32>(r.x * cos(t.x), r.x * sin(t.x), r.y * cos(t.y));
}

// Unit-variance Gaussian noise correlated over neighbouring pixels: the
// independent samples of the 3x3 around `pixel` weighted by a Gaussian of
// sigma `size` and rescaled to unit variance, as if each photon were a small
// blob rather than a point. `size` 0 is independent per pixel.
fn blotchy3(pixel: vec2<i32>, size: f32) -> vec3<f32> {
    if (size <= 0.0) {
        return gaussian3(pixel);
    }
    let k = -0.5 / (size * size);
    var sum = vec3<f32>(0.0);
    var norm = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let w = exp(f32(dx * dx + dy * dy) * k);
            sum += w * gaussian3(pixel + vec2<i32>(dx, dy));
            norm += w * w;
        }
    }
    return sum / sqrt(norm);
}

// The spread of the artificial grain at amount 1: shot noise (variance
// proportional to the linear value) plus read noise, as the networks' noise
// model `sigma^2 = a x + b`, for roughly ISO 6400 on an APS-C sensor.
const GRAIN_A: f32 = 4.5e-3;
const GRAIN_B: f32 = 4.5e-6;
// How much of its colour noise the grain keeps (0 = the same noise in R, G
// and B, 1 = independent per channel; the luma part is the same either way),
// and how blotchy it is (the Gaussian's sigma in pixels, see `blotchy3`),
// which grows from the first to the second with the amount.
const GRAIN_CHROMA: f32 = 0.08;
const GRAIN_SIZE: vec2<f32> = vec2<f32>(0.2, 1.0);

// Gaussian noise added to a linear sRGB colour, per channel. `fade` shrinks
// it where a screen pixel averages several image pixels, as the average of
// that many independent samples would be.
fn add_grain(e: Edit, rgb: vec3<f32>, pixel: vec2<f32>, fade: f32) -> vec3<f32> {
    let sigma = e.added_grain * fade * sqrt(GRAIN_A * max(rgb, vec3<f32>(0.0)) + GRAIN_B);
    // Split the three samples into the part common to the channels (luma
    // noise) and what is left (chroma noise), and keep `GRAIN_CHROMA` of the
    // latter. The luma part is the same whatever the balance.
    let n = blotchy3(vec2<i32>(floor(pixel)), mix(GRAIN_SIZE.x, GRAIN_SIZE.y, e.added_grain));
    let shared_part = vec3<f32>((n.x + n.y + n.z) / 3.0);
    let noise = shared_part + GRAIN_CHROMA * (n - shared_part);
    return rgb + sigma * noise;
}

fn develop(e: Edit, cam: vec3<f32>, base: f32, pixel: vec2<f32>, fade: f32) -> vec3<f32> {
    var rgb = cam_to_srgb(e, cam);

    // Local contrast: scale each pixel by how far its log luma is above the
    // edge-preserving base layer, in stops.
    if (e.local != 0.0) {
        let detail = log2(max(dot(rgb, LUMA), 1e-4)) - base;
        rgb = rgb * exp2(clamp(detail, -3.0, 3.0) * e.local);
    }

    // Grain: in scene units like the sensor's own noise, so exposure
    // amplifies it with the signal.
    if (e.added_grain != 0.0) {
        rgb = add_grain(e, rgb, pixel, fade);
    }

    // Black point: in scene units, so exposure scales about the new black.
    rgb = rgb - e.black;
    rgb = rgb * exp2(e.exposure);

    let curve_in = encode(e, rgb);
    // For the preview's crossover overlay.
    curve_input_luma = dot(curve_in, vec3<f32>(0.2126, 0.7152, 0.0722));
    var encoded = sigmoid(e, curve_in);

    let luma = dot(encoded, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Saturation scales the colour (the distance from the grey of the same
    // luma); vibrance adds to the scale where the colour is small, by the
    // chroma the pixel has before either.
    let colour = encoded - vec3<f32>(luma);
    let chroma = length(colour) / e.vibrance_range;
    let scale = e.saturation * (1.0 + e.vibrance / (1.0 + chroma * chroma));
    encoded = max(vec3<f32>(luma) + colour * scale, vec3<f32>(0.0));

    // pow() is undefined at 0.
    var linear = select(pow(encoded, vec3<f32>(DISPLAY_GAMMA)), vec3<f32>(0.0), encoded <= vec3<f32>(0.0));
    if (e.edit_flags != 0u) {
        linear = apply_color_edits(e, linear);
    }
    return clamp(linear, vec3<f32>(0.0), vec3<f32>(1.0));
}
