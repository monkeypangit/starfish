// Draws the demosaiced image on the canvas: one triangle over the whole
// canvas, each pixel mapped back to the image. Zoomed in, image pixels are
// flat squares (to see what the network made of each one); zoomed out, the
// matching mip level is sampled trilinearly. Colour: the camera matrix,
// exposure, a clip at white, and saturation.

struct View {
    // white-balanced camera RGB -> linear sRGB, rows
    m0: vec4f,
    m1: vec4f,
    m2: vec4f,
    // the image point (px) at the canvas centre, and the canvas size (px)
    center: vec2f,
    canvas: vec2f,
    // canvas px per image px
    scale: f32,
    // 2^EV
    exposure: f32,
    // 1: the canvas is not sRGB, so encode here
    encode: u32,
    // 1 = unchanged; mixed away from the luma on display-encoded values, as
    // develop.wgsl does (there after AgX, which the viewer does not have)
    saturation: f32,
}

const LUMA: vec3f = vec3f(0.2126, 0.7152, 0.0722);

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var image: texture_2d<f32>;
@group(0) @binding(2) var trilinear: sampler;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn srgb(c: vec3f) -> vec3f {
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308));
}

fn srgb_to_linear(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let size = vec2f(textureDimensions(image));
    let p = (pos.xy - view.canvas * 0.5) / view.scale + view.center;
    if any(p < vec2f(0.0)) || any(p >= size) {
        return vec4f(0.11, 0.11, 0.12, 1.0);
    }
    var rgb: vec3f;
    if view.scale >= 1.0 {
        rgb = textureLoad(image, vec2i(floor(p)), 0).rgb;
    } else {
        rgb = textureSampleLevel(image, trilinear, p / size, log2(1.0 / view.scale)).rgb;
    }
    let rgb_out = vec3f(dot(view.m0.xyz, rgb), dot(view.m1.xyz, rgb), dot(view.m2.xyz, rgb));
    var c = srgb(clamp(rgb_out * view.exposure, vec3f(0.0), vec3f(1.0)));
    c = clamp(mix(vec3f(dot(c, LUMA)), c, view.saturation), vec3f(0.0), vec3f(1.0));
    if view.encode == 0u {
        // an sRGB canvas encodes on store
        c = srgb_to_linear(c);
    }
    return vec4f(c, 1.0);
}
