enable f16;
alias T = f16;
alias A = f16;

@group(0) @binding(0) var<storage, read> up: array<T>;     // 3 x H x W estimate
@group(0) @binding(1) var<storage, read> pw: array<vec4<T>>;
@group(0) @binding(2) var<storage, read> mos: array<T>;
@group(0) @binding(3) var<storage, read> sig: array<T>;
@group(0) @binding(4) var<storage, read_write> outp: array<T>;  // 3 x H x W

const H: u32 = 1056u;
const W: u32 = 1056u;
const P: u32 = 2u;
const NV: u32 = 8u;       // head width / 4
const NPY: u32 = 2u;      // pixels per invocation, 8 phase rows apart
const PHASE_OUT: bool = false;  // final 1x1 has one bank per CFA phase
const W0: u32 = 38980u;      // per bank: [tap][estimate, m, sigma][NV vec4], then NV vec4 bias
const NW0: u32 = 368u;
const W1: u32 = 40452u;      // per bank: [rgb][NV vec4], then one vec4 bias
const NW1: u32 = 25u;
var<workgroup> sw: array<vec4<T>, NW0>;
var<workgroup> so: array<vec4<T>, NW1>;

struct Valid { h: u32, w: u32 }
@group(0) @binding(5) var<uniform> valid: Valid;
const S: u32 = 1u;
fn valid_h() -> u32 { return (valid.h + S - 1u) / S; }
fn valid_w() -> u32 { return (valid.w + S - 1u) / S; }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let bank = wg.z;
    if (wg.x * 8u * P + bank % P >= valid_w() || wg.y * NPY * 8u * P + bank / P >= valid_h()) {
        return;
    }
    let bo = select(0u, bank, PHASE_OUT);
    for (var i = li; i < NW0; i += 64u) { sw[i] = pw[W0 + bank * NW0 + i]; }
    for (var i = li; i < NW1; i += 64u) { so[i] = pw[W1 + bo * NW1 + i]; }
    workgroupBarrier();
    let x = (wg.x * 8u + l.x) * P + bank % P;
    if (x >= W) { return; }
    var y: array<u32, NPY>;
    for (var p = 0u; p < NPY; p++) {
        // past the bottom: computed on a wrapped row, not stored
        y[p] = (((wg.y * NPY + p) * 8u + l.y) * P + bank / P) % H;
    }
    var h0_0 = vec4<A>(sw[360u]);
    var h0_1 = vec4<A>(sw[361u]);
    var h0_2 = vec4<A>(sw[362u]);
    var h0_3 = vec4<A>(sw[363u]);
    var h0_4 = vec4<A>(sw[364u]);
    var h0_5 = vec4<A>(sw[365u]);
    var h0_6 = vec4<A>(sw[366u]);
    var h0_7 = vec4<A>(sw[367u]);
    var h1_0 = vec4<A>(sw[360u]);
    var h1_1 = vec4<A>(sw[361u]);
    var h1_2 = vec4<A>(sw[362u]);
    var h1_3 = vec4<A>(sw[363u]);
    var h1_4 = vec4<A>(sw[364u]);
    var h1_5 = vec4<A>(sw[365u]);
    var h1_6 = vec4<A>(sw[366u]);
    var h1_7 = vec4<A>(sw[367u]);
    for (var dy = 0u; dy < 3u; dy++) {
        for (var dx = 0u; dx < 3u; dx++) {
            let xx = (x + W + dx - 1u) % W;
            let wi = (dy * 3u + dx) * 5u * NV;
            let i0 = ((y[0] + H + dy - 1u) % H) * W + xx;
            let v0 = array<A, 5>(A(up[i0]), A(up[H * W + i0]), A(up[2u * H * W + i0]), A(mos[i0]), A(sig[i0]));
            let i1 = ((y[1] + H + dy - 1u) % H) * W + xx;
            let v1 = array<A, 5>(A(up[i1]), A(up[H * W + i1]), A(up[2u * H * W + i1]), A(mos[i1]), A(sig[i1]));
            let w0_0 = vec4<A>(sw[wi + 0u]);
            let w0_1 = vec4<A>(sw[wi + 1u]);
            let w0_2 = vec4<A>(sw[wi + 2u]);
            let w0_3 = vec4<A>(sw[wi + 3u]);
            let w0_4 = vec4<A>(sw[wi + 4u]);
            let w0_5 = vec4<A>(sw[wi + 5u]);
            let w0_6 = vec4<A>(sw[wi + 6u]);
            let w0_7 = vec4<A>(sw[wi + 7u]);
            h0_0 += v0[0] * w0_0;
            h0_1 += v0[0] * w0_1;
            h0_2 += v0[0] * w0_2;
            h0_3 += v0[0] * w0_3;
            h0_4 += v0[0] * w0_4;
            h0_5 += v0[0] * w0_5;
            h0_6 += v0[0] * w0_6;
            h0_7 += v0[0] * w0_7;
            h1_0 += v1[0] * w0_0;
            h1_1 += v1[0] * w0_1;
            h1_2 += v1[0] * w0_2;
            h1_3 += v1[0] * w0_3;
            h1_4 += v1[0] * w0_4;
            h1_5 += v1[0] * w0_5;
            h1_6 += v1[0] * w0_6;
            h1_7 += v1[0] * w0_7;
            let w1_0 = vec4<A>(sw[wi + 8u]);
            let w1_1 = vec4<A>(sw[wi + 9u]);
            let w1_2 = vec4<A>(sw[wi + 10u]);
            let w1_3 = vec4<A>(sw[wi + 11u]);
            let w1_4 = vec4<A>(sw[wi + 12u]);
            let w1_5 = vec4<A>(sw[wi + 13u]);
            let w1_6 = vec4<A>(sw[wi + 14u]);
            let w1_7 = vec4<A>(sw[wi + 15u]);
            h0_0 += v0[1] * w1_0;
            h0_1 += v0[1] * w1_1;
            h0_2 += v0[1] * w1_2;
            h0_3 += v0[1] * w1_3;
            h0_4 += v0[1] * w1_4;
            h0_5 += v0[1] * w1_5;
            h0_6 += v0[1] * w1_6;
            h0_7 += v0[1] * w1_7;
            h1_0 += v1[1] * w1_0;
            h1_1 += v1[1] * w1_1;
            h1_2 += v1[1] * w1_2;
            h1_3 += v1[1] * w1_3;
            h1_4 += v1[1] * w1_4;
            h1_5 += v1[1] * w1_5;
            h1_6 += v1[1] * w1_6;
            h1_7 += v1[1] * w1_7;
            let w2_0 = vec4<A>(sw[wi + 16u]);
            let w2_1 = vec4<A>(sw[wi + 17u]);
            let w2_2 = vec4<A>(sw[wi + 18u]);
            let w2_3 = vec4<A>(sw[wi + 19u]);
            let w2_4 = vec4<A>(sw[wi + 20u]);
            let w2_5 = vec4<A>(sw[wi + 21u]);
            let w2_6 = vec4<A>(sw[wi + 22u]);
            let w2_7 = vec4<A>(sw[wi + 23u]);
            h0_0 += v0[2] * w2_0;
            h0_1 += v0[2] * w2_1;
            h0_2 += v0[2] * w2_2;
            h0_3 += v0[2] * w2_3;
            h0_4 += v0[2] * w2_4;
            h0_5 += v0[2] * w2_5;
            h0_6 += v0[2] * w2_6;
            h0_7 += v0[2] * w2_7;
            h1_0 += v1[2] * w2_0;
            h1_1 += v1[2] * w2_1;
            h1_2 += v1[2] * w2_2;
            h1_3 += v1[2] * w2_3;
            h1_4 += v1[2] * w2_4;
            h1_5 += v1[2] * w2_5;
            h1_6 += v1[2] * w2_6;
            h1_7 += v1[2] * w2_7;
            let w3_0 = vec4<A>(sw[wi + 24u]);
            let w3_1 = vec4<A>(sw[wi + 25u]);
            let w3_2 = vec4<A>(sw[wi + 26u]);
            let w3_3 = vec4<A>(sw[wi + 27u]);
            let w3_4 = vec4<A>(sw[wi + 28u]);
            let w3_5 = vec4<A>(sw[wi + 29u]);
            let w3_6 = vec4<A>(sw[wi + 30u]);
            let w3_7 = vec4<A>(sw[wi + 31u]);
            h0_0 += v0[3] * w3_0;
            h0_1 += v0[3] * w3_1;
            h0_2 += v0[3] * w3_2;
            h0_3 += v0[3] * w3_3;
            h0_4 += v0[3] * w3_4;
            h0_5 += v0[3] * w3_5;
            h0_6 += v0[3] * w3_6;
            h0_7 += v0[3] * w3_7;
            h1_0 += v1[3] * w3_0;
            h1_1 += v1[3] * w3_1;
            h1_2 += v1[3] * w3_2;
            h1_3 += v1[3] * w3_3;
            h1_4 += v1[3] * w3_4;
            h1_5 += v1[3] * w3_5;
            h1_6 += v1[3] * w3_6;
            h1_7 += v1[3] * w3_7;
            let w4_0 = vec4<A>(sw[wi + 32u]);
            let w4_1 = vec4<A>(sw[wi + 33u]);
            let w4_2 = vec4<A>(sw[wi + 34u]);
            let w4_3 = vec4<A>(sw[wi + 35u]);
            let w4_4 = vec4<A>(sw[wi + 36u]);
            let w4_5 = vec4<A>(sw[wi + 37u]);
            let w4_6 = vec4<A>(sw[wi + 38u]);
            let w4_7 = vec4<A>(sw[wi + 39u]);
            h0_0 += v0[4] * w4_0;
            h0_1 += v0[4] * w4_1;
            h0_2 += v0[4] * w4_2;
            h0_3 += v0[4] * w4_3;
            h0_4 += v0[4] * w4_4;
            h0_5 += v0[4] * w4_5;
            h0_6 += v0[4] * w4_6;
            h0_7 += v0[4] * w4_7;
            h1_0 += v1[4] * w4_0;
            h1_1 += v1[4] * w4_1;
            h1_2 += v1[4] * w4_2;
            h1_3 += v1[4] * w4_3;
            h1_4 += v1[4] * w4_4;
            h1_5 += v1[4] * w4_5;
            h1_6 += v1[4] * w4_6;
            h1_7 += v1[4] * w4_7;
        }
    }
    let plane = H * W;
    let rb = vec4<A>(so[3u * NV]);
    for (var p = 0u; p < NPY; p++) {
        if (((wg.y * NPY + p) * 8u + l.y) * P + bank / P >= H) { break; }
        let px = y[p] * W + x;
        for (var k = 0u; k < 3u; k++) {
            var r = rb[k];
            if (p == 0u) {
                r += dot(max(h0_0, h0_0 * A(0.1)), vec4<A>(so[k * NV + 0u]));
                r += dot(max(h0_1, h0_1 * A(0.1)), vec4<A>(so[k * NV + 1u]));
                r += dot(max(h0_2, h0_2 * A(0.1)), vec4<A>(so[k * NV + 2u]));
                r += dot(max(h0_3, h0_3 * A(0.1)), vec4<A>(so[k * NV + 3u]));
                r += dot(max(h0_4, h0_4 * A(0.1)), vec4<A>(so[k * NV + 4u]));
                r += dot(max(h0_5, h0_5 * A(0.1)), vec4<A>(so[k * NV + 5u]));
                r += dot(max(h0_6, h0_6 * A(0.1)), vec4<A>(so[k * NV + 6u]));
                r += dot(max(h0_7, h0_7 * A(0.1)), vec4<A>(so[k * NV + 7u]));
            } else if (p == 1u) {
                r += dot(max(h1_0, h1_0 * A(0.1)), vec4<A>(so[k * NV + 0u]));
                r += dot(max(h1_1, h1_1 * A(0.1)), vec4<A>(so[k * NV + 1u]));
                r += dot(max(h1_2, h1_2 * A(0.1)), vec4<A>(so[k * NV + 2u]));
                r += dot(max(h1_3, h1_3 * A(0.1)), vec4<A>(so[k * NV + 3u]));
                r += dot(max(h1_4, h1_4 * A(0.1)), vec4<A>(so[k * NV + 4u]));
                r += dot(max(h1_5, h1_5 * A(0.1)), vec4<A>(so[k * NV + 5u]));
                r += dot(max(h1_6, h1_6 * A(0.1)), vec4<A>(so[k * NV + 6u]));
                r += dot(max(h1_7, h1_7 * A(0.1)), vec4<A>(so[k * NV + 7u]));
            }
            outp[k * plane + px] = T(r);
        }
    }
}
