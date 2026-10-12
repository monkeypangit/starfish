// Counts the display values of the histogram's draw (histogram.rs): sRGB
// encoded R, G, B as preview.wgsl writes them, plus the luminance of the
// linear values, each into the bin of the 8-bit value it rounds to. Pixels
// the draw left out (alpha 0: past the frame) do not count.

const BINS: u32 = 256u;

@group(0) @binding(0) var image: texture_2d<f32>;
// R, G, B, then luminance, BINS each.
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 1024>;

fn srgb_decode(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn srgb_encode(c: f32) -> f32 {
    if (c <= 0.0031308) {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

// The bin of display-encoded `c`: the 8-bit value it rounds to.
fn bin(c: f32) -> u32 {
    return min(u32(max(floor(c * 255.0 + 0.5), 0.0)), BINS - 1u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(image);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let texel = textureLoad(image, id.xy, 0);
    if (texel.a < 0.5) {
        return;
    }
    let encoded = clamp(texel.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let luma = dot(srgb_decode(encoded), vec3<f32>(0.2126, 0.7152, 0.0722));
    atomicAdd(&bins[bin(encoded.r)], 1u);
    atomicAdd(&bins[BINS + bin(encoded.g)], 1u);
    atomicAdd(&bins[2u * BINS + bin(encoded.b)], 1u);
    atomicAdd(&bins[3u * BINS + bin(srgb_encode(luma))], 1u);
}
