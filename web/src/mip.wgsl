// One mip level from the one above it: the mean of each 2x2 block (an odd
// last row or column is left out).

@group(0) @binding(0) var above: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let p = vec2i(pos.xy) * 2;
    return (textureLoad(above, p, 0) + textureLoad(above, p + vec2i(1, 0), 0)
        + textureLoad(above, p + vec2i(0, 1), 0) + textureLoad(above, p + vec2i(1, 1), 0)) * 0.25;
}
