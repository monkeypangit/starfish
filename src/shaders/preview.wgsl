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

// The crop-mode viewfinder (preview::Crop), drawn here per pixel in screen
// pixels around the frame's centre. Sizes are fractions of the frame's short
// side unless noted.
// Radius of the split-image disc; the microprism collar is twice that.
const SPLIT_RADIUS: f32 = 0.035;
// How dark the halves get at their dark side, and how far across the disc
// (0..1) the clear side stays clear before darkening starts.
const SPLIT_DARKNESS: f32 = 0.25;
const SPLIT_START: f32 = 0.15;
// The microprism collar is a grid of small pyramids, each tilting the picture
// towards one of its four faces: the cell size and the shift (fractions of
// the frame's short side), how much the red and blue shift apart from green
// (dispersion), and how much the faces' shading varies.
const PRISM_CELL: f32 = 0.005;
const PRISM_SHIFT: f32 = 0.002;
const PRISM_DISPERSION: f32 = 0.05;
const PRISM_SHADE: f32 = 0.05;
// The split-image disc: each half shifts this far sideways (fraction of the
// short side), the upper one to the right.
const SPLIT_SHIFT: f32 = 0.0015;
// Brightness outside the frame, and the blur there: radius in screen pixels
// far from the edge, and the distance over which it grows from the edge.
const OUTSIDE_DIM: f32 = 0.1;
const BLUR_RADIUS: f32 = 10.0;
const BLUR_RAMP: f32 = 20.0;
const BLUR_TAPS: i32 = 16;
// Black shade inside the frame but outside the collar, and the vignette
// strength at the corners.
const SHADE_ALPHA: f32 = 0.0;
const VIGNETTE: f32 = 0.8;
// Line width in screen pixels, and each line's opacity (black, and the white
// line beside it).
const LINE_WIDTH: f32 = 1.0;
const OUTLINE_ALPHA: f32 = 0.2;
const RING_ALPHA: f32 = 0.2;

// The face of the microprism pyramid under `p` (screen pixels from the
// frame's centre): the unit step towards which it tilts the picture. The
// pyramids sit on a grid turned 45 degrees, so they are diamonds on screen
// and their faces point diagonally.
fn prism_face(p: vec2<f32>, short: f32) -> vec2<f32> {
    let turned = vec2<f32>(p.x - p.y, p.x + p.y) * 0.70710678;
    let local = fract(turned / (PRISM_CELL * short)) * 2.0 - 1.0;
    var face = vec2<f32>(0.0, sign(local.y));
    if (abs(local.x) > abs(local.y)) {
        face = vec2<f32>(sign(local.x), 0.0);
    }
    // Back to screen axes.
    return vec2<f32>(face.x + face.y, face.y - face.x) * 0.70710678;
}

// How the collar moves the picture at `frame`: xy is the shift in screen
// pixels, z the dispersion (red goes 1 + z times as far, blue 1 - z). Zero
// outside the collar.
fn collar_shift(frame: vec2<f32>) -> vec3<f32> {
    let size = view.frame_size * view.scale;
    let short = min(size.x, size.y);
    let p = (frame - view.frame_size * 0.5) * view.scale;
    let r = length(p);
    let disc = SPLIT_RADIUS * short;
    if (r < disc) {
        return vec3<f32>(select(-1.0, 1.0, p.y < 0.0) * SPLIT_SHIFT * short, 0.0, 0.0);
    }
    if (r < 2.0 * disc) {
        return vec3<f32>(prism_face(p, short) * PRISM_SHIFT * short, PRISM_DISPERSION);
    }
    return vec3<f32>(0.0);
}

// `rgb` with a line along d = 0 (d is a signed distance in screen pixels,
// negative inside the shape): white just inside the black one.
fn outlined(rgb_in: vec3<f32>, d: f32, alpha: f32) -> vec3<f32> {
    var rgb = mix(rgb_in, vec3<f32>(1.0), alpha * line_of(d + LINE_WIDTH));
    return rgb * (1.0 - alpha * line_of(d));
}

// The display-encoded colour `rgb` at `frame` (frame space) with the
// viewfinder over it: dimmed outside, then inside the frame the vignette, the
// shade outside the collar, the prism facets, the split-image disc and the
// lines. `edge` is the distance outside the frame in screen pixels.
fn viewfinder(rgb_in: vec3<f32>, frame: vec2<f32>, edge: f32) -> vec3<f32> {
    var rgb = rgb_in;
    let size = view.frame_size * view.scale;
    let short = min(size.x, size.y);
    let p = (frame - view.frame_size * 0.5) * view.scale;
    let r = length(p);
    let disc = SPLIT_RADIUS * short;
    let collar = 2.0 * disc;
    let ring = size.x / 6.0;
    let inside = fill_of(edge);

    rgb = rgb * mix(OUTSIDE_DIM, 1.0, inside);

    // Vignette: radial, 0 at the centre to VIGNETTE at the corners.
    let corner_r = length(p / (size * 0.5)) * 0.70710678;
    rgb = rgb * (1.0 - VIGNETTE * smoothstep(0.35, 1.0, corner_r) * inside);

    // Shade between the collar and the frame's edge.
    let in_collar = fill_of(r - collar);
    rgb = rgb * (1.0 - SHADE_ALPHA * inside * (1.0 - in_collar));

    // The pyramids' faces catch the light differently.
    let face = prism_face(p, short);
    let in_ring = in_collar * (1.0 - fill_of(r - disc));
    rgb = rgb * (1.0 + PRISM_SHADE * dot(face, vec2<f32>(-0.6, -0.8)) * in_ring);

    // Split-image disc, as when stopped down: the halves darken towards
    // opposite sides (the upper one to the right).
    let across = clamp((p.x / disc * 0.5 + 0.5 - SPLIT_START) / (1.0 - SPLIT_START), 0.0, 1.0);
    let dark = select(1.0 - across, across, p.y < 0.0);
    rgb = rgb * (1.0 - SPLIT_DARKNESS * dark * fill_of(r - disc));

    // The split line across the disc.
    rgb = rgb * (1.0 - RING_ALPHA * line_of(p.y) * fill_of(r - disc));

    // Lines: the frame, the ground-glass ring, the collar and the disc, each
    // black with a white line of the same width just inside it, so it reads
    // against both bright and dark pictures.
    rgb = outlined(rgb, edge, OUTLINE_ALPHA);
    rgb = outlined(rgb, r - ring, RING_ALPHA);
    rgb = outlined(rgb, r - collar, RING_ALPHA);
    rgb = outlined(rgb, r - disc, RING_ALPHA);
    return rgb;
}

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
fn fill_of(d: f32) -> f32 {
    return clamp(0.5 - d, 0.0, 1.0);
}

fn line_of(d: f32) -> f32 {
    return clamp(LINE_WIDTH * 0.5 + 0.5 - abs(d), 0.0, 1.0);
}

// The source at point `q` of the corrected image.
fn sample_at(q: vec2<f32>, level: f32) -> vec3<f32> {
    return sample_source(tex, samp, view.grid, lens_source(lens_lut, view.lens, q).xy, level);
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
    let source = lens_source(lens_lut, view.lens, here).xy;
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
            cam = sample_source_cubic(tex, samp, view.grid, at.xy);
        } else {
            cam = sample_source(tex, samp, view.grid, at.xy, level);
        }
    }
    if (view.shade != 0.0 && edge <= 0.0) {
        // The collar and the split disc move the picture under them.
        let shift = collar_shift(in.frame);
        if (any(shift != vec3<f32>(0.0))) {
            let red = shift.xy * (1.0 + shift.z);
            let blue = shift.xy * (1.0 - shift.z);
            cam = vec3<f32>(
                sample_at(p + red.x * dp_dx + red.y * dp_dy, level).r,
                sample_at(p + shift.x * dp_dx + shift.y * dp_dy, level).g,
                sample_at(p + blue.x * dp_dx + blue.y * dp_dy, level).b,
            );
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
    cam = cam * at.z;
    var base = 0.0;
    if (view.edit.local != 0.0) {
        base = textureSampleLevel(base_tex, base_samp, p / view.grid.size, 0.0).r;
    }
    var rgb = develop(view.edit, cam, base);
    if (view.encode_srgb == 1u) {
        rgb = srgb_encode(rgb);
    }
    if (view.shade != 0.0) {
        rgb = viewfinder(rgb, in.frame, edge);
    }
    return vec4<f32>(rgb, 1.0);
}
