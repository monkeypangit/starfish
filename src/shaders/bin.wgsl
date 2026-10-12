// Bins each `bin`×`bin` block of the CFA mosaic into one RGB pixel by
// averaging the samples of each colour, then applies white balance.

struct Params {
    size: vec2<u32>,
    cfa_size: vec2<u32>,
    bin: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
    let out_size = textureDimensions(out_tex);
    if (id.x >= out_size.x || id.y >= out_size.y) {
        return;
    }

    var sum = vec3<f32>(0.0);
    var count = vec3<f32>(0.0);
    for (var dy = 0u; dy < params.bin; dy++) {
        for (var dx = 0u; dx < params.bin; dx++) {
            let p = id.xy * params.bin + vec2<u32>(dx, dy);
            if (p.x >= params.size.x || p.y >= params.size.y) {
                continue;
            }
            let c = cfa_color(p);
            sum[c] += textureLoad(raw_tex, vec2<i32>(p), 0).r;
            count[c] += 1.0;
        }
    }

    let mean = select(vec3<f32>(0.0), sum / max(count, vec3<f32>(1.0)), count > vec3<f32>(0.0));
    textureStore(out_tex, vec2<i32>(id.xy), vec4<f32>(mean * params.wb.rgb, 1.0));
}
