// Guided filter (He et al.) of the image's log luma, self-guided, on a
// small overview of the whole image. The result is the edge-preserving
// base layer that develop.wgsl takes the detail from. develop.wgsl is
// joined in front of this file (for sample_source). Five passes:
//   fs_overview  the overview of the lens-corrected image, from the image's
//                mipmaps
//   fs_h      horizontal box of (L, L²)
//   fs_v_ab   vertical box → mean, variance → a = var/(var+eps), b = mean(1-a)
//   fs_h_ab   horizontal box of (a, b)
//   fs_final  vertical box of (a, b) → base = mean(a)·L + mean(b)

struct Params {
    // Camera RGB → linear sRGB, row-major.
    cam_to_srgb: array<vec4<f32>, 3>,
    radius: i32,
    eps: f32,
    // The source's mip level whose texels are the overview's pixels.
    level: f32,
    grid: Grid,
    lens: Lens,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var overview: texture_2d<f32>;
// fs_overview only: the image (source.rs).
@group(0) @binding(3) var source: texture_2d_array<f32>;
@group(0) @binding(4) var source_samp: sampler;
@group(0) @binding(5) var lens_lut: texture_2d<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VertexOut;
    out.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return out;
}

fn luma_at(p: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(overview));
    let c = textureLoad(overview, clamp(p, vec2<i32>(0), size - 1), 0).rgb;
    let rgb = vec3<f32>(
        dot(params.cam_to_srgb[0].xyz, c),
        dot(params.cam_to_srgb[1].xyz, c),
        dot(params.cam_to_srgb[2].xyz, c),
    );
    return log2(max(dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-4));
}

fn src_at(p: vec2<i32>) -> vec2<f32> {
    let size = vec2<i32>(textureDimensions(src));
    return textureLoad(src, clamp(p, vec2<i32>(0), size - 1), 0).xy;
}

fn taps() -> f32 {
    return f32(2 * params.radius + 1);
}

@fragment
fn fs_h(in: VertexOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.position.xy);
    var acc = vec2<f32>(0.0);
    for (var d = -params.radius; d <= params.radius; d++) {
        let l = luma_at(p + vec2<i32>(d, 0));
        acc += vec2<f32>(l, l * l);
    }
    return vec4<f32>(acc / taps(), 0.0, 0.0);
}

@fragment
fn fs_v_ab(in: VertexOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.position.xy);
    var acc = vec2<f32>(0.0);
    for (var d = -params.radius; d <= params.radius; d++) {
        acc += src_at(p + vec2<i32>(0, d));
    }
    let m = acc / taps();
    let a = max(m.y - m.x * m.x, 0.0) / (max(m.y - m.x * m.x, 0.0) + params.eps);
    return vec4<f32>(a, m.x * (1.0 - a), 0.0, 0.0);
}

@fragment
fn fs_h_ab(in: VertexOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.position.xy);
    var acc = vec2<f32>(0.0);
    for (var d = -params.radius; d <= params.radius; d++) {
        acc += src_at(p + vec2<i32>(d, 0));
    }
    return vec4<f32>(acc / taps(), 0.0, 0.0);
}

@fragment
fn fs_final(in: VertexOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.position.xy);
    var acc = vec2<f32>(0.0);
    for (var d = -params.radius; d <= params.radius; d++) {
        acc += src_at(p + vec2<i32>(0, d));
    }
    let m = acc / taps();
    return vec4<f32>(m.x * luma_at(p) + m.y, 0.0, 0.0, 0.0);
}

@fragment
fn fs_overview(in: VertexOut) -> @location(0) vec4<f32> {
    // The centre of the block of (corrected) image pixels under this
    // overview pixel, through the lens correction, as the preview draws it.
    let at = lens_source(lens_lut, params.lens, in.position.xy * exp2(params.level));
    return vec4<f32>(sample_source(source, source_samp, params.grid, at.xy, params.level) * at.z, 1.0);
}
