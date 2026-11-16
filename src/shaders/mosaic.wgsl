// The CFA mosaic as an image: each site shows its own sample, white balanced,
// in its own colour's channel, the other two channels 0.

struct Params {
    cfa_size: vec2<u32>,
    _pad: vec2<u32>,
    wb: vec4<f32>,
    // 6×6 colour indices (0 = R, 1 = G, 2 = B), packed four per vec4.
    pattern: array<vec4<u32>, 9>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var raw_tex: texture_2d<f32>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba16float, write>;

fn cfa_color(p: vec2<u32>) -> u32 {
    let i = (p.y % params.cfa_size.y) * params.cfa_size.x + (p.x % params.cfa_size.x);
    return params.pattern[i / 4u][i % 4u];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out_tex);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let c = cfa_color(id.xy);
    var rgb = vec3<f32>(0.0);
    rgb[c] = textureLoad(raw_tex, vec2<i32>(id.xy), 0).r * params.wb[c];
    // alpha is the grain: none, nothing was denoised
    textureStore(out_tex, vec2<i32>(id.xy), vec4<f32>(rgb, 0.0));
}
