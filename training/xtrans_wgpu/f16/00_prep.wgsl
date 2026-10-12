enable f16;
alias T = f16;
alias A = f16;

@group(0) @binding(0) var<storage, read> mos: array<T>;
@group(0) @binding(2) var<storage, read> sig: array<T>;
@group(0) @binding(4) var<storage, read_write> outp: array<T>;

const H: u32 = 176u;    // packed grid
const W: u32 = 176u;
const TL: u32 = 6u;  // tile

struct Valid { h: u32, w: u32 }
@group(0) @binding(5) var<uniform> valid: Valid;
const S: u32 = 6u;
fn valid_h() -> u32 { return (valid.h + S - 1u) / S; }
fn valid_w() -> u32 { return (valid.w + S - 1u) / S; }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x;
    let y = g.y;
    if (x >= min(W, valid_w()) || y >= min(H, valid_h())) { return; }
    let plane = H * W;
    let px = y * W + x;
    var s = A(0);
    for (var i = 0u; i < TL; i++) {
        for (var j = 0u; j < TL; j++) {
            let src = (TL * y + i) * TL * W + TL * x + j;
            outp[(i * TL + j) * plane + px] = mos[src];
            s += A(sig[src]);
        }
    }
    outp[TL * TL * plane + px] = T(s / A(TL * TL));
}
