// Draws the image over the whole widget: each pixel is mapped back to the
// lens-corrected image, through the lens correction to the source image,
// sampled (bicubic at an output pixel's centre when zoomed in, from the
// mipmaps when zoomed out), developed with develop.wgsl (joined in front of
// this file) and, in crop mode, the viewfinder drawn over it. The export
// draws with it too, at one image pixel per output pixel.

struct View {
    // The user's edits (develop.wgsl).
    edit: Edit,
    // The image's layers (source.rs); its size is the unrotated image's.
    grid: Grid,
    // Frame point shown at the centre of the widget.
    center: vec2<f32>,
    // Widget size in logical pixels.
    viewport: vec2<f32>,
    // Logical pixels per image pixel.
    scale: f32,
    // 1 if the render target is not an sRGB format and we must encode.
    encode_srgb: u32,
    // Clockwise quarter turns (preview::Rotation).
    rotation: u32,
    // 1 in crop mode: darken outside the frame; otherwise discard it.
    shade: f32,
    // Rotated-image point at the centre of the export frame (preview::Crop).
    crop_center: vec2<f32>,
    // The frame's size in image pixels.
    frame_size: vec2<f32>,
    // Cosine and sine of the image's clockwise tilt against the frame.
    crop_cs: vec2<f32>,
    // Corner radius of the frame in frame space; 0 = square.
    corner: f32,
    // 1: every target pixel is the output pixel under its centre, sampled as
    // the export does, whatever the scale (the histogram's strided draw).
    point: u32,
    // 1: draw clipped highlights black / clipped shadows white (the overlay).
    clip_highlights: u32,
    clip_shadows: u32,
    // 1: tint pixels near the tone curve's crossover (the midpoint slider).
    crossover: u32,
    // 1: draw the rule-of-thirds grid over the frame (also pads lens to its
    // 16-byte alignment).
    grid_lines: u32,
    // Distortion and vignetting (develop.wgsl); an identity table if off.
    lens: Lens,
}

@group(0) @binding(0) var<uniform> view: View;
// Trilinear, clamped.
@group(0) @binding(1) var samp: sampler;
// Log luma with the fine detail filtered out (local.rs), over the whole image.
@group(0) @binding(2) var base_tex: texture_2d<f32>;
@group(0) @binding(3) var base_samp: sampler;
// The image (source.rs).
@group(0) @binding(4) var tex: texture_2d_array<f32>;
// The lens table (develop.wgsl's lens_lookup).
@group(0) @binding(5) var lens_lut: texture_2d<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    // Frame-space position.
    @location(0) frame: vec2<f32>,
}

// Frame space -> point of the (lens-corrected) image, the inverse of the
// tilt about the frame's centre and then of the quarter turns
// (Crop::to_image, Rotation::unapply).
fn to_image(f: vec2<f32>) -> vec2<f32> {
    let d = f - view.frame_size * 0.5;
    let c = view.crop_cs.x;
    let s = view.crop_cs.y;
    let q = vec2<f32>(d.x * c + d.y * s, d.y * c - d.x * s) + view.crop_center;
    let size = view.grid.size;
    switch view.rotation {
        case 1u: { return vec2<f32>(q.y, size.y - q.x); }
        case 2u: { return size - q; }
        case 3u: { return vec2<f32>(size.x - q.y, q.x); }
        default: { return q; }
    }
}

// The whole widget.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let corner = corners[index];
    let screen = corner * view.viewport;
    var out: VertexOut;
    out.position = vec4<f32>(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    out.frame = (screen - view.viewport * 0.5) / view.scale + view.center;
    return out;
}

// The crop-mode frame, drawn here per pixel in screen pixels.
// Brightness outside the frame, and the blur there: radius in screen pixels
// far from the edge, and the distance over which it grows from the edge.
const OUTSIDE_DIM: f32 = 0.2;
const BLUR_RADIUS: f32 = 10.0;
const BLUR_RAMP: f32 = 20.0;
const BLUR_TAPS: i32 = 16;
// Line width in screen pixels, and each line's opacity (black, and the white
// line beside it).
const LINE_WIDTH: f32 = 1.0;
const OUTLINE_ALPHA: f32 = 0.2;
// The composition grid's lines: opacity, as the outline's.
const GRID_ALPHA: f32 = 0.35;

fn fill_of(d: f32) -> f32 {
    return clamp(0.5 - d, 0.0, 1.0);
}

fn line_of(d: f32) -> f32 {
    return clamp(LINE_WIDTH * 0.5 + 0.5 - abs(d), 0.0, 1.0);
}

// `rgb` with a line along d = 0 (d is a signed distance in screen pixels,
// negative inside the shape): white just inside the black one.
fn outlined(rgb_in: vec3<f32>, d: f32, alpha: f32) -> vec3<f32> {
    var rgb = mix(rgb_in, vec3<f32>(1.0), alpha * line_of(d + LINE_WIDTH));
    return rgb * (1.0 - alpha * line_of(d));
}

// The display-encoded colour `rgb` with the frame's outline over it, and
// everything outside the frame dimmed. `edge` is the distance outside the
// frame in screen pixels.
fn viewfinder(rgb_in: vec3<f32>, edge: f32) -> vec3<f32> {
    let rgb = rgb_in * mix(OUTSIDE_DIM, 1.0, fill_of(edge));
    return outlined(rgb, edge, OUTLINE_ALPHA);
}

// The rule-of-thirds grid over the frame at `frame` (frame space): two lines
// each way, drawn black with a white line beside them like the outline.
fn thirds(rgb_in: vec3<f32>, frame: vec2<f32>) -> vec3<f32> {
    var rgb = rgb_in;
    let size = view.frame_size * view.scale;
    let p = frame * view.scale;
    for (var i = 1; i < 3; i++) {
        let t = f32(i) / 3.0;
        rgb = outlined(rgb, p.x - size.x * t, GRID_ALPHA);
        rgb = outlined(rgb, p.y - size.y * t, GRID_ALPHA);
    }
    return rgb;
}

// The linear values from which a channel counts as clipped: what rounds to
// 255 and to 0 in 8-bit sRGB.
const CLIP_HIGH: f32 = 0.9977;
const CLIP_LOW: f32 = 0.00015;
// The crossover overlay: half-width of the band around the midpoint, in
// display-encoded values, and its tint (linear white).
const CROSSOVER_BAND: f32 = 0.05;
const CROSSOVER_TINT: vec3<f32> = vec3<f32>(0.5, 0.0, 1.0);

// Whether `p`, in image pixels, is on the image.
fn on_image(p: vec2<f32>) -> bool {
    return all(p >= vec2<f32>(0.0)) && all(p < view.grid.size);
}

// Signed distance from `p` to the image's rectangle, in image pixels:
// negative inside.
fn image_sd(p: vec2<f32>) -> f32 {
    let half = view.grid.size * 0.5;
    let q = abs(p - half) - half;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

// Signed distance from `f` (frame space) to the frame's rounded rectangle, in
// frame pixels: negative inside.
fn frame_sd(f: vec2<f32>) -> f32 {
    let half = view.frame_size * 0.5;
    let q = abs(f - half) - (half - vec2<f32>(view.corner));
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - view.corner;
}

// Anti-aliased coverage of the inside (d < 0) of a shape and of a line along
// d = 0, for a signed distance in screen pixels.
// The source at point `q` of the corrected image.
fn sample_at(q: vec2<f32>, level: f32) -> vec3<f32> {
    return sample_lens(tex, samp, view.grid, lens_source(lens_lut, view.lens, q), level, view.edit.grain);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let here = to_image(in.frame);
    // Corrected-image pixels per target pixel, for the blur, and source
    // pixels per target pixel, for the mip level (needs uniform control
    // flow).
    let dp_dx = dpdx(here);
    let dp_dy = dpdy(here);
    let footprint = max(length(dp_dx), length(dp_dy));
    let source = lens_source(lens_lut, view.lens, here).green;
    let source_footprint = max(length(dpdx(source)), length(dpdy(source)));

    // Distance outside the frame in screen pixels (negative inside).
    let edge = frame_sd(in.frame) * view.scale;
    if (edge > 0.0 && view.shade == 0.0) {
        discard;
    }
    // Outside the frame, the blur fades the picture out a little beyond the
    // image's edge; past that the pane shows.
    if (edge > 0.0 && image_sd(here) / footprint > BLUR_RADIUS) {
        discard;
    }

    // At or above one target pixel per image pixel, the output pixel under
    // this one (as the export makes it): its centre, sampled bicubically.
    // Zoomed out, the mip level that matches the source pixels per target
    // pixel (which the distortion changes a little).
    let exact = footprint <= 1.001 || view.point == 1u;
    let level = max(log2(source_footprint), 0.0);
    var p = here;
    if (exact) {
        p = to_image(floor(in.frame) + 0.5);
    }
    let at = lens_source(lens_lut, view.lens, p);
    // Black where the image does not reach.
    var cam = vec3<f32>(0.0);
    if (on_image(p)) {
        if (exact) {
            cam = sample_lens_cubic(tex, samp, view.grid, at, view.edit.grain);
        } else {
            cam = sample_lens(tex, samp, view.grid, at, level, view.edit.grain);
        }
    }
    if (edge > 0.0) {
        // Outside the frame the picture is blurred, more the further out. The
        // taps sit on a spiral.
        let radius = BLUR_RADIUS * smoothstep(0.0, BLUR_RAMP, edge);
        // Taps beyond the image count as black, so its edge fades out like
        // frosted glass.
        var sum = cam;
        for (var i = 0; i < BLUR_TAPS; i++) {
            let t = (f32(i) + 0.5) / f32(BLUR_TAPS);
            let angle = f32(i) * 2.39996323;
            let offset = sqrt(t) * radius * vec2<f32>(cos(angle), sin(angle));
            let tap = p + offset.x * dp_dx + offset.y * dp_dy;
            if (on_image(tap)) {
                sum += sample_at(tap, level);
            }
        }
        cam = sum / f32(BLUR_TAPS + 1);
    }
    // Sharpening, inside the frame and the image only.
    if (view.edit.sharpness != 0.0 && edge <= 0.0 && on_image(p)) {
        cam = cam * sharpen(view.edit, tex, samp, view.grid, at, level, view.edit.sharpness);
    }
    cam = cam * at.gain;
    var base = 0.0;
    if (view.edit.local != 0.0) {
        base = textureSampleLevel(base_tex, base_samp, p / view.grid.size, 0.0).r;
    }
    // Where an image pixel is smaller than a target pixel, the grain of the
    // many it covers averages out.
    var rgb = develop(view.edit, cam, base, p, select(1.0 / max(footprint, 1.0), 1.0, exact));
    // Clipping overlay, on the developed (clamped) value: a channel at white
    // is a clipped highlight, one at black a clipped shadow.
    if (edge <= 0.0 && on_image(p)) {
        if (view.clip_highlights == 1u && any(rgb >= vec3<f32>(CLIP_HIGH))) {
            rgb = vec3<f32>(0.0);
        } else if (view.clip_shadows == 1u && any(rgb <= vec3<f32>(CLIP_LOW))) {
            rgb = vec3<f32>(1.0);
        }
    }
    // Crossover overlay: pixels whose tone curve input is near the midpoint.
    if (view.crossover == 1u && edge <= 0.0 && on_image(p)) {
        let near = 1.0 - smoothstep(0.0, CROSSOVER_BAND, abs(curve_input_luma - view.edit.curve.x));
        rgb = mix(rgb, CROSSOVER_TINT, 0.1 * near);
    }
    if (view.encode_srgb == 1u) {
        rgb = srgb_encode(rgb);
    }
    if (view.grid_lines == 1u && edge <= 0.0) {
        rgb = thirds(rgb, in.frame);
    }
    if (view.shade != 0.0) {
        rgb = viewfinder(rgb, edge);
    }
    return vec4<f32>(rgb, 1.0);
}
