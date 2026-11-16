enable f16;
alias T = f16;
alias A = f16;

@group(0) @binding(0) var<storage, read> inp: array<T>;
@group(0) @binding(3) var<storage, read> pw: array<vec4<T>>;
@group(0) @binding(4) var<storage, read_write> outp: array<T>;

const CIN: u32 = 32u;
const CINP: u32 = 32u;     // CIN rounded up to CI (packed weights are zero there)
const COUT: u32 = 32u;
const NOB: u32 = 2u;       // output channels / OCB, rounded up
const K: u32 = 3u;
const KK: u32 = 3u * 3u;
const R: i32 = 3 / 2;
const H: i32 = 528;
const W: i32 = 528;
const PWOFF: u32 = 9832u;   // in vec4s
const OUT_SHUF: bool = false;
const SF: u32 = 1u;
const ACT: bool = false;

const TX: u32 = 16u;
const PY: u32 = 3u;             // output rows per invocation, 4 apart
const TY: u32 = 4u * PY;
const OCB: u32 = 16u;           // output channels per workgroup, all in each invocation
const NV: u32 = OCB / 4u;
const CI: u32 = 16u;             // input channels staged per chunk
const SW: u32 = TX + K - 1u;
const SH: u32 = TY + K - 1u;
const SP: u32 = SH * SW;          // one staged input plane
const NWC: u32 = CI * KK * NV;    // staged weight vec4s per chunk

var<workgroup> sx: array<T, 4032>;
var<workgroup> sw: array<vec4<T>, 576>;

fn out_index(oc: u32, y: u32, x: u32) -> u32 {
    if (OUT_SHUF) {
        let s = oc % (SF * SF);
        return ((oc / (SF * SF)) * SF * u32(H) + SF * y + s / SF) * SF * u32(W) + SF * x + s % SF;
    }
    return (oc * u32(H) + y) * u32(W) + x;
}

struct Valid { h: u32, w: u32 }
@group(0) @binding(5) var<uniform> valid: Valid;
const S: u32 = 2u;
fn valid_h() -> u32 { return (valid.h + S - 1u) / S; }
fn valid_w() -> u32 { return (valid.w + S - 1u) / S; }

fn load(c: u32, y0: i32, x0: i32) -> T {
    if (c >= CIN || y0 < 0 || y0 >= H || x0 < 0 || x0 >= W) { return T(0); }
    let y = y0;
    let x = x0;
    let v = inp[(c * u32(H) + u32(y)) * u32(W) + u32(x)];
    return max(v, v * T(0.1));
}

fn store(oc0: u32, y: i32, x: i32, a: vec4<A>) {
    for (var o = 0u; o < 4u; o++) {
        let oc = oc0 + o;
        if (oc >= COUT) { break; }
        var r = a[o];
        r += A(inp[(oc * u32(H) + u32(y)) * u32(W) + u32(x)]);
        if (ACT) { r = max(r, r * A(0.1)); }
        outp[out_index(oc, u32(y), u32(x))] = T(r);
    }
}

@compute @workgroup_size(16, 4, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    if (wg.x * TX >= valid_w() || wg.y * TY >= valid_h()) { return; }
    let ox0 = i32(wg.x * TX);
    let oy0 = i32(wg.y * TY);
    let ob = wg.z;
    let x = ox0 + i32(l.x);
    let y0 = oy0 + i32(l.y);

    var a0_0 = vec4<A>(0);
    var a0_1 = vec4<A>(0);
    var a0_2 = vec4<A>(0);
    var a0_3 = vec4<A>(0);
    var a1_0 = vec4<A>(0);
    var a1_1 = vec4<A>(0);
    var a1_2 = vec4<A>(0);
    var a1_3 = vec4<A>(0);
    var a2_0 = vec4<A>(0);
    var a2_1 = vec4<A>(0);
    var a2_2 = vec4<A>(0);
    var a2_3 = vec4<A>(0);

    for (var c0 = 0u; c0 < CINP; c0 += CI) {
        for (var i = li; i < SP; i += 64u) {
            let gy = oy0 + i32(i / SW) - R;
            let gx = ox0 + i32(i % SW) - R;
            for (var ci = 0u; ci < CI; ci++) {
                sx[ci * SP + i] = load(c0 + ci, gy, gx);
            }
        }
        for (var i = li; i < NWC; i += 64u) {
            sw[i] = pw[PWOFF + (ob * CINP + c0) * KK * NV + i];
        }
        workgroupBarrier();
        for (var ci = 0u; ci < CI; ci++) {
            for (var ky = 0u; ky < K; ky++) {
                for (var kx = 0u; kx < K; kx++) {
                    let s = (ci * SH + l.y + ky) * SW + l.x + kx;
                    let wt = (ci * KK + ky * K + kx) * NV;
                    let v0 = A(sx[s + 0u * SW]);
                    let v1 = A(sx[s + 4u * SW]);
                    let v2 = A(sx[s + 8u * SW]);
                    let w0 = vec4<A>(sw[wt + 0u]);
                    let w1 = vec4<A>(sw[wt + 1u]);
                    let w2 = vec4<A>(sw[wt + 2u]);
                    let w3 = vec4<A>(sw[wt + 3u]);
                    a0_0 += v0 * w0;
                    a0_1 += v0 * w1;
                    a0_2 += v0 * w2;
                    a0_3 += v0 * w3;
                    a1_0 += v1 * w0;
                    a1_1 += v1 * w1;
                    a1_2 += v1 * w2;
                    a1_3 += v1 * w3;
                    a2_0 += v2 * w0;
                    a2_1 += v2 * w1;
                    a2_2 += v2 * w2;
                    a2_3 += v2 * w3;
                }
            }
        }
        workgroupBarrier();
    }

    if (x >= W) { return; }
    var accs = array<vec4<A>, 12>(a0_0, a0_1, a0_2, a0_3, a1_0, a1_1, a1_2, a1_3, a2_0, a2_1, a2_2, a2_3);
    for (var p = 0u; p < PY; p++) {
        let y = y0 + i32(4u * p);
        if (y >= H) { continue; }
        for (var h = 0u; h < NV; h++) {
            let oc0 = ob * OCB + 4u * h;
            let bo = PWOFF + NOB * CINP * KK * NV + oc0 / 4u;
            store(oc0, y, x, accs[NV * p + h] + vec4<A>(pw[bo]));
        }
    }
}
