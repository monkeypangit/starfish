// Lens correction (src/lens.rs): resamples the developed image so each output
// pixel reads the source at the radius the distortion table gives (red and
// blue a little off that for chromatic aberration), and multiplies in the
// vignetting gain.

struct Params {
    // Image size in pixels.
    size: vec2<f32>,
    // Pixel index of the image centre.
    centre: vec2<f32>,
    // 1 / half-diagonal in pixels: the normalised radius is distance × this.
    inv_half_diagonal: f32,
    // Radius of the last table entry.
    lut_max: f32,
    // Index of the last table entry.
    lut_last: f32,
    _pad: f32,
}

const LUT_SIZE = 257;

@group(0) @binding(0) var<uniform> params: Params;
// Per normalised radius: source radius / output radius, red CA, blue CA,
// vignetting gain (lens::LensCorrection::lut).
@group(0) @binding(1) var<uniform> lut: array<vec4<f32>, LUT_SIZE>;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;

fn lookup(rho: f32) -> vec4<f32> {
    let t = clamp(rho / params.lut_max, 0.0, 1.0) * params.lut_last;
    let i = u32(floor(t));
    return mix(lut[i], lut[min(i + 1u, LUT_SIZE - 1u)], t - floor(t));
}

// Catmull-Rom at `pos` (pixel index, texel i centred on i) from nine
// bilinear fetches.
fn sample_cubic(pos: vec2<f32>) -> vec4<f32> {
    let p = pos + 0.5;
    let c1 = floor(p - 0.5) + 0.5;
    let f = p - c1;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let t0 = (c1 - 1.0) / params.size;
    let t12 = (c1 + w2 / w12) / params.size;
    let t3 = (c1 + 2.0) / params.size;

    var sum = vec4<f32>(0.0);
    sum += textureSampleLevel(src, samp, vec2<f32>(t0.x, t0.y), 0.0) * (w0.x * w0.y);
    sum += textureSampleLevel(src, samp, vec2<f32>(t12.x, t0.y), 0.0) * (w12.x * w0.y);
    sum += textureSampleLevel(src, samp, vec2<f32>(t3.x, t0.y), 0.0) * (w3.x * w0.y);
    sum += textureSampleLevel(src, samp, vec2<f32>(t0.x, t12.y), 0.0) * (w0.x * w12.y);
    sum += textureSampleLevel(src, samp, vec2<f32>(t12.x, t12.y), 0.0) * (w12.x * w12.y);
    sum += textureSampleLevel(src, samp, vec2<f32>(t3.x, t12.y), 0.0) * (w3.x * w12.y);
    sum += textureSampleLevel(src, samp, vec2<f32>(t0.x, t3.y), 0.0) * (w0.x * w3.y);
    sum += textureSampleLevel(src, samp, vec2<f32>(t12.x, t3.y), 0.0) * (w12.x * w3.y);
    sum += textureSampleLevel(src, samp, vec2<f32>(t3.x, t3.y), 0.0) * (w3.x * w3.y);
    return sum;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (f32(id.x) >= params.size.x || f32(id.y) >= params.size.y) {
        return;
    }
    let offset = vec2<f32>(id.xy) - params.centre;
    let l = lookup(length(offset) * params.inv_half_diagonal);

    let green = offset * l.x;
    let red = sample_cubic(params.centre + green * (1.0 + l.y)).r;
    let centre = sample_cubic(params.centre + green);
    let blue = sample_cubic(params.centre + green * (1.0 + l.z)).b;

    // The vignetting belongs to the source pixel, so its gain is looked up
    // at the source radius.
    let gain = lookup(length(green) * params.inv_half_diagonal).w;
    textureStore(dst, vec2<i32>(id.xy), vec4<f32>(vec3<f32>(red, centre.g, blue) * gain, centre.a));
}
