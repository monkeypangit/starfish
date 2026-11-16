// Lateral CA taken out of the raw mosaic before the demosaic: training/
// common.py's ca_correct, which has the method and its measurements, on the
// frame as decoded (not balanced: the method is the same at any channel
// gain). Five passes over the frame, one entry point each:
//
//   green     G: a normalised 3x3 tent over the green samples, at every
//             pixel (green ones too: a guide of raw samples passed green's
//             noise into red and blue at full strength)
//   moved     Gs: at each red and blue site q, G at its scene point q - d(q)
//             (Catmull-Rom, edges repeated)
//   fit       per pixel and colour c, c = a Gs + b over the c sites of the
//             5x5 window by least squares, a regularised towards the colour
//             ratio mean c / mean Gs, bounded to 0..ratio_max (as balanced
//             0..CA_FIX_RATIO_MAX): a (r = red's, g = blue's). Sites whose own
//             sample is clipped (>= P.clip) are left out, and with fewer than
//             MIN_VALID left a is 0
//   fit_mean  a averaged over the 5x5 window
//   gather    at each c site p, the c samples q whose scene points land near
//             p, each moved along green, c(q) + a (G(p) - Gs(q)), Gaussian
//             weights on how far q - d(p) lands from p; a is 0 for a clipped
//             q (it is then resampled along the field only)

struct Params {
    w: u32, h: u32, cfa_w: u32, cfa_h: u32,
    cy: f32, cx: f32, inv_radius: f32, eps: f32,
    kr0: f32, kr1: f32, kr2: f32, reach_r: f32,
    kb0: f32, kb1: f32, kb2: f32, reach_b: f32,
    sigma: f32, ratio_max_r: f32, ratio_max_b: f32, clip: f32,
    shift_r: vec2<f32>, shift_b: vec2<f32>,
    // x = the guided move's strength (1; 0 = plain resample along the
    // field, the benchmark's baseline); y = the result's bound: it may leave
    // the range of the c samples within CLAMP_REACH of its scene point by
    // y times that range (negative: no bound)
    tune: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> cfa: array<u32>;
@group(0) @binding(2) var frame: texture_2d<f32>;
@group(0) @binding(3) var green_in: texture_2d<f32>;
@group(0) @binding(4) var moved_in: texture_2d<f32>;
@group(0) @binding(5) var fit_in: texture_2d<f32>;
@group(0) @binding(6) var fit_mean_in: texture_2d<f32>;
@group(0) @binding(7) var green_out: texture_storage_2d<r32float, write>;
@group(0) @binding(8) var moved_out: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var fit_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(10) var fit_mean_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(11) var out: texture_storage_2d<r32float, write>;

const RADIUS: i32 = 2;   // common.CA_FIX_RADIUS
// the c samples within this many px of the scene point bound the result
const CLAMP_REACH: f32 = 2.5;
const GATHER: i32 = 3;   // px around round(d) the gather looks: the Gaussian is ~0 beyond
// fewer unclipped c sites than this in a fit window and a is 0
const MIN_VALID: f32 = 3.0;

fn inside(p: vec2<i32>) -> bool {
    return p.x >= 0 && p.y >= 0 && p.x < i32(P.w) && p.y < i32(P.h);
}

// A clipped sample says nothing about the scene's edge (the channels clip at
// different brightness): it is left out of the fit and not moved along green.
fn own_clipped(v: f32) -> bool {
    return v >= P.clip;
}

// 0 = R, 1 = G, 2 = B at an in-image pixel
fn colour(p: vec2<i32>) -> u32 {
    return cfa[(u32(p.y) % P.cfa_h) * P.cfa_w + u32(p.x) % P.cfa_w];
}

// (dy, dx): the red (c 0) or blue sample at p saw the scene at p - d,
// d = s(rho) (p - centre), as the networks' cut shader makes it
fn field(c: u32, p: vec2<i32>) -> vec2<f32> {
    let v = vec2<f32>(f32(p.y) - P.cy, f32(p.x) - P.cx);
    let rho = length(v) * P.inv_radius;
    var s: f32;
    if (c == 0u) {
        let r2 = min(rho, P.reach_r) * min(rho, P.reach_r);
        s = P.kr0 + r2 * (P.kr1 + r2 * P.kr2);
        return s * v + P.shift_r;
    } else {
        let r2 = min(rho, P.reach_b) * min(rho, P.reach_b);
        s = P.kb0 + r2 * (P.kb1 + r2 * P.kb2);
    }
    return s * v + P.shift_b;
}

fn load(t: texture_2d<f32>, p: vec2<i32>) -> f32 {
    return textureLoad(t, vec2<u32>(u32(p.x), u32(p.y)), 0).r;
}

fn pixel(g: vec3<u32>) -> vec2<i32> {
    return vec2<i32>(i32(g.x), i32(g.y));
}

@compute @workgroup_size(16, 16, 1)
fn green(@builtin(global_invocation_id) g: vec3<u32>) {
    let p = pixel(g);
    if (!inside(p)) { return; }
    var num = 0.0;
    var den = 0.0;
    for (var i = -1; i <= 1; i++) {
        for (var j = -1; j <= 1; j++) {
            let q = p + vec2<i32>(j, i);
            if (inside(q) && colour(q) == 1u) {
                let w = f32((2 - abs(i)) * (2 - abs(j)));
                num += w * load(frame, q);
                den += w;
            }
        }
    }
    textureStore(green_out, vec2<u32>(g.xy), vec4<f32>(num / max(den, 1e-6), 0.0, 0.0, 0.0));
}

fn catmull_rom(f: f32) -> vec4<f32> {
    return vec4<f32>(f * (-0.5 + f * (1.0 - 0.5 * f)), 1.0 + f * f * (-2.5 + 1.5 * f),
                     f * (0.5 + f * (2.0 - 1.5 * f)), f * f * (-0.5 + 0.5 * f));
}

@compute @workgroup_size(16, 16, 1)
fn moved(@builtin(global_invocation_id) g: vec3<u32>) {
    let p = pixel(g);
    if (!inside(p)) { return; }
    let c = colour(p);
    var v = 0.0;
    if (c != 1u) {
        let at = vec2<f32>(f32(p.y), f32(p.x)) - field(c, p);
        let i0 = floor(at);
        let wy = catmull_rom(at.x - i0.x);
        let wx = catmull_rom(at.y - i0.y);
        for (var i = 0; i < 4; i++) {
            let y = clamp(i32(i0.x) + i - 1, 0, i32(P.h) - 1);
            var row = 0.0;
            for (var j = 0; j < 4; j++) {
                let x = clamp(i32(i0.y) + j - 1, 0, i32(P.w) - 1);
                row += wx[j] * load(green_in, vec2<i32>(x, y));
            }
            v += wy[i] * row;
        }
    }
    textureStore(moved_out, vec2<u32>(g.xy), vec4<f32>(v, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn fit(@builtin(global_invocation_id) g: vec3<u32>) {
    let p = pixel(g);
    if (!inside(p)) { return; }
    var a = vec2<f32>(0.0);
    for (var k = 0; k < 2; k++) {
        let c = u32(2 * k);
        var n = 0.0;
        var sm = 0.0;
        var sg = 0.0;
        var sgg = 0.0;
        var smg = 0.0;
        for (var i = -RADIUS; i <= RADIUS; i++) {
            for (var j = -RADIUS; j <= RADIUS; j++) {
                let q = p + vec2<i32>(j, i);
                if (inside(q) && colour(q) == c && !own_clipped(load(frame, q))) {
                    let m = load(frame, q);
                    let gs = load(moved_in, q);
                    n += 1.0;
                    sm += m;
                    sg += gs;
                    sgg += gs * gs;
                    smg += m * gs;
                }
            }
        }
        if (n >= MIN_VALID) {
            let mm = sm / n;
            let mg = sg / n;
            // regularised towards the colour ratio, not 0: where green
            // explains little (fine texture, noise) the samples still move;
            // bounded, as it explodes where a saturated colour has no green
            let ratio = clamp(mm / max(mg, 1e-4), 0.0, select(P.ratio_max_b, P.ratio_max_r, k == 0));
            a[k] = (smg / n - mm * mg + P.eps * ratio) / (sgg / n - mg * mg + P.eps);
        }
    }
    textureStore(fit_out, vec2<u32>(g.xy), vec4<f32>(a * P.tune.x, 0.0, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn fit_mean(@builtin(global_invocation_id) g: vec3<u32>) {
    let p = pixel(g);
    if (!inside(p)) { return; }
    var sum = vec2<f32>(0.0);
    var n = 0.0;
    for (var i = -RADIUS; i <= RADIUS; i++) {
        for (var j = -RADIUS; j <= RADIUS; j++) {
            let q = p + vec2<i32>(j, i);
            if (inside(q)) {
                sum += textureLoad(fit_in, vec2<u32>(u32(q.x), u32(q.y)), 0).rg;
                n += 1.0;
            }
        }
    }
    textureStore(fit_mean_out, vec2<u32>(g.xy), vec4<f32>(sum / n, 0.0, 0.0));
}

@compute @workgroup_size(16, 16, 1)
fn gather(@builtin(global_invocation_id) g: vec3<u32>) {
    let p = pixel(g);
    if (!inside(p)) { return; }
    let c = colour(p);
    var v = load(frame, p);
    if (c != 1u) {
        let d = field(c, p);
        let k = i32(c / 2u);
        let centre = vec2<i32>(i32(round(d.y)), i32(round(d.x)));   // (x, y)
        var num = 0.0;
        var num_a = 0.0;
        var den = 0.0;
        var lo = 1e30;
        var hi = -1e30;
        for (var i = -GATHER; i <= GATHER; i++) {
            for (var j = -GATHER; j <= GATHER; j++) {
                let o = centre + vec2<i32>(j, i);
                let q = p + o;
                if (inside(q) && colour(q) == c) {
                    let off = vec2<f32>(f32(o.y), f32(o.x)) - d;
                    let w = exp(-dot(off, off) / (2.0 * P.sigma * P.sigma));
                    if (dot(off, off) <= CLAMP_REACH * CLAMP_REACH) {
                        lo = min(lo, load(frame, q));
                        hi = max(hi, load(frame, q));
                    }
                    var a = textureLoad(fit_mean_in, vec2<u32>(u32(q.x), u32(q.y)), 0)[k];
                    // no move along a clipped sample
                    if (own_clipped(load(frame, q))) {
                        a = 0.0;
                    }
                    num += w * (load(frame, q) - a * load(moved_in, q));
                    num_a += w * a;
                    den += w;
                }
            }
        }
        if (den > 1e-12) {
            v = (num + num_a * load(green_in, p)) / den;
            if (P.tune.y >= 0.0 && lo <= hi) {
                let margin = P.tune.y * (hi - lo);
                v = clamp(v, lo - margin, hi + margin);
            }
        }
    }
    textureStore(out, vec2<u32>(g.xy), vec4<f32>(v, 0.0, 0.0, 0.0));
}
