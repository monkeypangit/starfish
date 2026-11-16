enable f16;
alias T = f16;
alias A = f16;

@group(0) @binding(0) var<storage, read> mos: array<T>;
@group(0) @binding(2) var<storage, read> sig: array<T>;
@group(0) @binding(4) var<storage, read_write> outp: array<T>;

const H: u32 = 352u;    // packed grid
const W: u32 = 352u;
const TL: u32 = 3u;  // tile
const R: u32 = 2u;    // cells per CFA period, per side
const NP: u32 = 14u;  // planes before sigma

struct Valid { h: u32, w: u32 }
@group(0) @binding(5) var<uniform> valid: Valid;
const S: u32 = 3u;
fn valid_h() -> u32 { return (valid.h + S - 1u) / S; }
fn valid_w() -> u32 { return (valid.w + S - 1u) / S; }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x;
    let y = g.y;
    if (x >= min(W, valid_w()) || y >= min(H, valid_h())) { return; }
    let plane = H * W;
    let px = y * W + x;
    let src = TL * y * TL * W + TL * x;      // the cell's first site
    let q = (y % R) * R + x % R;             // its phase
    outp[0u * plane + px] = mos[src + 0u * TL * W + 0u];
    outp[1u * plane + px] = mos[src + 0u * TL * W + 1u];
    outp[2u * plane + px] = select(T(0), mos[src + 0u * TL * W + 2u], ((9u >> q) & 1u) != 0u);
    outp[3u * plane + px] = select(T(0), mos[src + 0u * TL * W + 2u], ((6u >> q) & 1u) != 0u);
    outp[4u * plane + px] = mos[src + 1u * TL * W + 0u];
    outp[5u * plane + px] = mos[src + 1u * TL * W + 1u];
    outp[6u * plane + px] = select(T(0), mos[src + 1u * TL * W + 2u], ((6u >> q) & 1u) != 0u);
    outp[7u * plane + px] = select(T(0), mos[src + 1u * TL * W + 2u], ((9u >> q) & 1u) != 0u);
    outp[8u * plane + px] = select(T(0), mos[src + 2u * TL * W + 0u], ((6u >> q) & 1u) != 0u);
    outp[9u * plane + px] = select(T(0), mos[src + 2u * TL * W + 0u], ((9u >> q) & 1u) != 0u);
    outp[10u * plane + px] = select(T(0), mos[src + 2u * TL * W + 1u], ((9u >> q) & 1u) != 0u);
    outp[11u * plane + px] = select(T(0), mos[src + 2u * TL * W + 1u], ((6u >> q) & 1u) != 0u);
    outp[12u * plane + px] = mos[src + 2u * TL * W + 2u];
    outp[13u * plane + px] = select(T(0), T(1), ((6u >> q) & 1u) != 0u);
    var s = A(0);
    for (var i = 0u; i < TL; i++) {
        for (var j = 0u; j < TL; j++) {
            s += A(sig[src + i * TL * W + j]);
        }
    }
    outp[NP * plane + px] = T(s / A(TL * TL));
}
