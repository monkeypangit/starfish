enable f16;
alias T = f16;
alias A = f16;

@group(0) @binding(0) var<storage, read> inp: array<T>;
@group(0) @binding(3) var<storage, read> pw: array<vec4<T>>;
@group(0) @binding(4) var<storage, read_write> outp: array<T>;

const CIN: u32 = 288u;
const CINP: u32 = 288u;
const COUT: u32 = 108u;
const NOG: u32 = 4u;       // output-channel groups of OCW
const H: u32 = 176u;
const W: u32 = 176u;
const NGX: u32 = 484u;       // workgroups along x of the 2-D dispatch
const PWOFF: u32 = 278676u;   // in vec4s
const OUT_SHUF: bool = true;
const SF: u32 = 6u;
const ACT: bool = false;

const NP: u32 = 64u;
const OCW: u32 = 32u;
const CI: u32 = 32u;

var<workgroup> sx: array<T, 2048>;           // [ci][pixel]
var<workgroup> sw: array<vec4<T>, 256>;     // [ci][8 vec4 = 32 channels]

fn out_index(oc: u32, y: u32, x: u32) -> u32 {
    if (OUT_SHUF) {
        let s = oc % (SF * SF);
        return ((oc / (SF * SF)) * SF * u32(H) + SF * y + s / SF) * SF * u32(W) + SF * x + s % SF;
    }
    return (oc * u32(H) + y) * u32(W) + x;
}

struct Valid { h: u32, w: u32 }
@group(0) @binding(5) var<uniform> valid: Valid;
const S: u32 = 6u;
fn valid_h() -> u32 { return (valid.h + S - 1u) / S; }
fn valid_w() -> u32 { return (valid.w + S - 1u) / S; }

fn load(c: u32, p: u32) -> T {
    if (c >= CIN || p >= H * W) { return T(0); }
    return inp[c * H * W + p];
}

fn store(oc0: u32, p: u32, a: vec4<A>) {
    if (p >= H * W) { return; }
    for (var o = 0u; o < 4u; o++) {
        let oc = oc0 + o;
        if (oc >= COUT) { break; }
        var r = a[o];
        if (ACT) { r = max(r, r * A(0.1)); }
        outp[out_index(oc, p / W, p % W)] = T(r);
    }
}

@compute @workgroup_size(16, 4, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let p0 = (wg.y * NGX + wg.x) * NP;
    // the NP pixels run along a row, possibly into the next: skipped if they
    // start below the valid rows, or right of the valid columns and end there
    // or on a row below them
    let py = p0 / W;
    let px0 = p0 % W;
    if (py >= valid_h() || (px0 >= valid_w() && (px0 + NP <= W || py + 1u >= valid_h()))) {
        return;
    }
    let og = wg.z;
    var a00 = vec4<A>(0); var a01 = vec4<A>(0);
    var a10 = vec4<A>(0); var a11 = vec4<A>(0);
    var a20 = vec4<A>(0); var a21 = vec4<A>(0);
    var a30 = vec4<A>(0); var a31 = vec4<A>(0);
    let wo = l.y * 2u;           // this invocation's 8 channels = vec4s wo, wo+1
    for (var c0 = 0u; c0 < CINP; c0 += CI) {
        for (var i = li; i < CI * NP; i += 64u) {
            sx[i] = load(c0 + i / NP, p0 + i % NP);
        }
        for (var i = li; i < CI * 8u; i += 64u) {
            sw[i] = pw[PWOFF + (og * CINP + c0) * 8u + i];
        }
        workgroupBarrier();
        for (var ci = 0u; ci < CI; ci++) {
            let s = ci * NP + l.x;
            let v0 = A(sx[s]);
            let v1 = A(sx[s + 16u]);
            let v2 = A(sx[s + 32u]);
            let v3 = A(sx[s + 48u]);
            let w0 = vec4<A>(sw[ci * 8u + wo]);
            let w1 = vec4<A>(sw[ci * 8u + wo + 1u]);
            a00 += v0 * w0; a01 += v0 * w1;
            a10 += v1 * w0; a11 += v1 * w1;
            a20 += v2 * w0; a21 += v2 * w1;
            a30 += v3 * w0; a31 += v3 * w1;
        }
        workgroupBarrier();
    }
    let oc0 = og * OCW + l.y * 8u;
    let bo = PWOFF + NOG * CINP * 8u + oc0 / 4u;
    let b0 = vec4<A>(pw[bo]);
    let b1 = vec4<A>(pw[bo + 1u]);
    let px = p0 + l.x;
    store(oc0, px, a00 + b0);       store(oc0 + 4u, px, a01 + b1);
    store(oc0, px + 16u, a10 + b0); store(oc0 + 4u, px + 16u, a11 + b1);
    store(oc0, px + 32u, a20 + b0); store(oc0 + 4u, px + 32u, a21 + b1);
    store(oc0, px + 48u, a30 + b0); store(oc0 + 4u, px + 48u, a31 + b1);
}
