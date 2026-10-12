// Bilinearly upscales the binned image back to full sensor resolution.

struct Params {
    bin: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var binned: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_size = textureDimensions(out_tex);
    if (id.x >= out_size.x || id.y >= out_size.y) {
        return;
    }

    let binned_size = vec2<f32>(textureDimensions(binned));
    let uv = (vec2<f32>(id.xy) + 0.5) / params.bin / binned_size;
    let color = textureSampleLevel(binned, samp, uv, 0.0);
    textureStore(out_tex, vec2<i32>(id.xy), color);
}
