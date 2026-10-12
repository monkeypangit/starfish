alias T = f32;
alias A = f32;

@group(0) @binding(0) var<storage, read> inp: array<T>;
@group(0) @binding(3) var<storage, read> pw: array<vec4<T>>;
@group(0) @binding(4) var<storage, read_write> outp: array<T>;

const CIN: u32 = 288u;
const CINP: u32 = 288u;     // CIN rounded up to CI (packed weights are zero there)
const COUT: u32 = 144u;
const NOB: u32 = 18u;       // output channels / 8, rounded up
const K: u32 = 3u;
const KK: u32 = 3u * 3u;
const R: i32 = 3 / 2;
const H: i32 = 176;
const W: i32 = 176;
const PWOFF: u32 = 45396u;   // in vec4s
const OUT_SHUF: bool = false;
const SF: u32 = 1u;
const ACT: bool = true;

const TX: u32 = 16u;
const TY: u32 = 16u;
const CI: u32 = 8u;             // input channels staged per chunk
const SW: u32 = TX + K - 1u;
const SH: u32 = TY + K - 1u;
const SP: u32 = SH * SW;          // one staged input plane
const NWC: u32 = CI * KK * 2u;    // staged weight vec4s per chunk

var<workgroup> sx: array<T, 2592>;
var<workgroup> sw: array<vec4<T>, 144>;

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

fn load(c: u32, y0: i32, x0: i32) -> T {
    if (c >= CIN) { return T(0); }
    let y = (y0 + H) % H;
    let x = (x0 + W) % W;
    return inp[(c * u32(H) + u32(y)) * u32(W) + u32(x)];
}

fn store(oc0: u32, y: i32, x: i32, a: vec4<A>) {
    for (var o = 0u; o < 4u; o++) {
        let oc = oc0 + o;
        if (oc >= COUT) { break; }
        var r = a[o];
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

    var a00 = vec4<A>(0); var a01 = vec4<A>(0);
    var a10 = vec4<A>(0); var a11 = vec4<A>(0);
    var a20 = vec4<A>(0); var a21 = vec4<A>(0);
    var a30 = vec4<A>(0); var a31 = vec4<A>(0);

    for (var c0 = 0u; c0 < CINP; c0 += CI) {
        for (var i = li; i < SP; i += 64u) {
            let gy = oy0 + i32(i / SW) - R;
            let gx = ox0 + i32(i % SW) - R;
            for (var ci = 0u; ci < CI; ci++) {
                sx[ci * SP + i] = load(c0 + ci, gy, gx);
            }
        }
        for (var i = li; i < NWC; i += 64u) {
            sw[i] = pw[PWOFF + (ob * CINP + c0) * KK * 2u + i];
        }
        workgroupBarrier();
        for (var ci = 0u; ci < CI; ci++) {
            for (var ky = 0u; ky < K; ky++) {
                for (var kx = 0u; kx < K; kx++) {
                    let s = (ci * SH + l.y + ky) * SW + l.x + kx;
                    let wt = (ci * KK + ky * K + kx) * 2u;
                    let v0 = A(sx[s]);
                    let v1 = A(sx[s + 4u * SW]);
                    let v2 = A(sx[s + 8u * SW]);
                    let v3 = A(sx[s + 12u * SW]);
                    let w0 = vec4<A>(sw[wt]);
                    let w1 = vec4<A>(sw[wt + 1u]);
                    a00 += v0 * w0;
                    a01 += v0 * w1;
                    a10 += v1 * w0;
                    a11 += v1 * w1;
                    a20 += v2 * w0;
                    a21 += v2 * w1;
                    a30 += v3 * w0;
                    a31 += v3 * w1;
                }
            }
        }
        workgroupBarrier();
    }

    if (x >= W) { return; }
    var accs = array<vec4<A>, 8>(a00, a01, a10, a11, a20, a21, a30, a31);
    for (var p = 0u; p < 4u; p++) {
        let y = y0 + i32(4u * p);
        if (y >= H) { continue; }
        for (var h = 0u; h < 2u; h++) {
            let oc0 = ob * 8u + 4u * h;
            let bo = PWOFF + NOB * CINP * KK * 2u + oc0 / 4u;
            store(oc0, y, x, accs[2u * p + h] + vec4<A>(pw[bo]));
        }
    }
}
