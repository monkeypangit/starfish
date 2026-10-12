// The per-pixel develop shared by the preview and the export: sampling the
// developed image (camera RGB), then the colour matrix, local contrast and
// the user's adjustments to AgX display RGB. src/develop.rs joins this file
// in front of each shader that uses it. src/agx.rs describes the tone
// mapping and computes its curve's numbers (agx::Curve).

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
// radius `rho`: source radius / output radius, red CA, blue CA (applied by
// the demosaic stage, unused here), vignetting gain.
fn lens_lookup(lut: texture_2d<f32>, l: Lens, rho: f32) -> vec4<f32> {
    let last = textureDimensions(lut).x - 1u;
    let t = clamp(rho / l.lut_max, 0.0, 1.0) * f32(last);
    let i = u32(floor(t));
    let a = textureLoad(lut, vec2<u32>(i, 0u), 0);
    let b = textureLoad(lut, vec2<u32>(min(i + 1u, last), 0u), 0);
    return mix(a, b, t - floor(t));
}

// For point `p` of the corrected image: xy = the source point it reads, z =
// the vignetting gain, which belongs to the source pixel (as lens.wgsl).
fn lens_source(lut: texture_2d<f32>, l: Lens, p: vec2<f32>) -> vec3<f32> {
    let offset = p - l.centre;
    let source = offset * lens_lookup(lut, l, length(offset) * l.inv_half_diagonal).x;
    let gain = lens_lookup(lut, l, length(source) * l.inv_half_diagonal).w;
    return vec3<f32>(l.centre + source, gain);
}

// The image at point `p` (pixels, texel i centred on i + 0.5) from mip
// `level`, bilinear (trilinear between levels).
fn sample_source(tex: texture_2d_array<f32>, samp: sampler, g: Grid, p: vec2<f32>, level: f32) -> vec3<f32> {
    let cell = grid_cell(g, p);
    let uv = (p - cell * g.step + g.overlap) / g.layer;
    return textureSampleLevel(tex, samp, uv, grid_layer(g, cell), level).rgb;
}

// The image at point `p` (pixels) at full size: Catmull-Rom from nine
// bilinear fetches, as lens.wgsl. The overlap covers its 4x4 footprint.
fn sample_source_cubic(tex: texture_2d_array<f32>, samp: sampler, g: Grid, p: vec2<f32>) -> vec3<f32> {
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

    var sum = vec3<f32>(0.0);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t0.x, t0.y), layer, 0.0).rgb * (w0.x * w0.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t12.x, t0.y), layer, 0.0).rgb * (w12.x * w0.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t3.x, t0.y), layer, 0.0).rgb * (w3.x * w0.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t0.x, t12.y), layer, 0.0).rgb * (w0.x * w12.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t12.x, t12.y), layer, 0.0).rgb * (w12.x * w12.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t3.x, t12.y), layer, 0.0).rgb * (w3.x * w12.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t0.x, t3.y), layer, 0.0).rgb * (w0.x * w3.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t12.x, t3.y), layer, 0.0).rgb * (w12.x * w3.y);
    sum += textureSampleLevel(tex, samp, vec2<f32>(t3.x, t3.y), layer, 0.0).rgb * (w3.x * w3.y);
    return sum;
}

// The user's edits (develop::EditUniform).
struct Edit {
    // Camera RGB → linear sRGB, row-major.
    cam_to_srgb: array<vec4<f32>, 3>,
    // Colour grading offsets (Adjustments::grade_tints) for the shadows,
    // midtones and highlights; w unused.
    grade: array<vec4<f32>, 3>,
    // AgX sigmoid (agx::Curve): pivot x, pivot y, slope, power.
    curve: vec4<f32>,
    // Its toe and shoulder limits, then the white point (agx::WHITE); w
    // unused.
    limits: vec4<f32>,
    // Shadows, midtones and highlights brightness (Adjustments::levels); w
    // unused.
    levels: vec4<f32>,
    // Exposure compensation in stops.
    exposure: f32,
    saturation: f32,
    // Local contrast amount (Adjustments::local_contrast).
    local: f32,
    // Black point (Adjustments::black_point), linear, subtracted before
    // exposure.
    black: f32,
}

const DISPLAY_GAMMA: f32 = 2.4;

// AgX's inset matrix, which compresses the primaries towards white so
// saturated colours roll off to white instead of skewing, and its inverse,
// the outset, restoring saturation after the curve. mat3x3 is column-major:
// each line is a column.
const AGX_INSET = mat3x3<f32>(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104,
);
const AGX_OUTSET = mat3x3<f32>(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116,
);

const LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);

// Linear → display-encoded, 0 at black and 1 at white and above
// (agx::encode). The floor keeps pow() away from 0, where it is undefined.
fn encode(e: Edit, c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c / e.limits.z, vec3<f32>(1e-20), vec3<f32>(1.0));
    return pow(x, vec3<f32>(1.0 / DISPLAY_GAMMA));
}

fn sigmoid(e: Edit, x: vec3<f32>) -> vec3<f32> {
    let z = e.curve.z * (x - e.curve.x);
    let limit = select(vec3<f32>(e.limits.y), vec3<f32>(e.limits.x), z < vec3<f32>(0.0));
    // A negative limit is the inverse curve (agx::Curve). The floors keep
    // pow() away from 0, where it is undefined.
    let q = pow(max(abs(z) / abs(limit), vec3<f32>(1e-20)), vec3<f32>(e.curve.w));
    let bend = max(1.0 + sign(limit) * q, vec3<f32>(1e-20));
    let y = e.curve.y + z / pow(bend, vec3<f32>(1.0 / e.curve.w));
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

// Camera RGB → linear display RGB in 0..1: the colour matrix, local
// contrast, the black point, then the tone mapping src/agx.rs describes. `base` is the local contrast base layer
// (local.rs) at this pixel, in stops; it is not read when `e.local` is 0.
fn develop(e: Edit, cam: vec3<f32>, base: f32) -> vec3<f32> {
    var rgb = cam_to_srgb(e, cam);

    // Local contrast: scale each pixel by how far its log luma is above the
    // edge-preserving base layer, in stops.
    if (e.local != 0.0) {
        let detail = log2(max(dot(rgb, LUMA), 1e-4)) - base;
        rgb = rgb * exp2(clamp(detail, -3.0, 3.0) * e.local);
    }

    // Black point: in scene units, so exposure scales about the new black.
    rgb = rgb - e.black;
    rgb = rgb * exp2(e.exposure);

    // Brightness per range: each level (-1 … +1) is that many stops, weighted
    // by how much of the pixel's display-encoded luma falls in its range.
    // The ranges come from the display-encoded luma.
    let level_luma = pow(clamp(dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0), 1.0 / DISPLAY_GAMMA);
    let level_shadows = 1.0 - smoothstep(0.0, 0.6, level_luma);
    let level_highlights = smoothstep(0.4, 1.0, level_luma);
    let level_midtones = max(1.0 - level_shadows - level_highlights, 0.0);
    // The highlights weight stays 1 above white, so overexposed values follow
    // the slider like the rest.
    rgb = rgb * exp2(dot(e.levels.xyz, vec3<f32>(level_shadows, level_midtones, level_highlights)));

    rgb = AGX_INSET * rgb;
    var encoded = sigmoid(e, encode(e, rgb));

    let luma = dot(encoded, vec3<f32>(0.2126, 0.7152, 0.0722));
    encoded = max(mix(vec3<f32>(luma), encoded, e.saturation), vec3<f32>(0.0));

    // Colour grading: the shadow, midtone and highlight tints
    // (Adjustments::grade_tints), weighted by the luma's ranges (summing to 1).
    let graded_luma = dot(encoded, vec3<f32>(0.2126, 0.7152, 0.0722));
    let shadows = 1.0 - smoothstep(0.0, 0.6, graded_luma);
    let highlights = smoothstep(0.4, 1.0, graded_luma);
    let midtones = max(1.0 - shadows - highlights, 0.0);
    encoded = max(
        encoded + e.grade[0].xyz * shadows + e.grade[1].xyz * midtones + e.grade[2].xyz * highlights,
        vec3<f32>(0.0),
    );
    // pow() is undefined at 0.
    let linear = select(pow(encoded, vec3<f32>(DISPLAY_GAMMA)), vec3<f32>(0.0), encoded <= vec3<f32>(0.0));
    return clamp(AGX_OUTSET * linear, vec3<f32>(0.0), vec3<f32>(1.0));
}
