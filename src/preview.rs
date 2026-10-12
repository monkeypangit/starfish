//! Center-pane image preview: an iced `shader` widget that draws the
//! developed image on iced's wgpu device, with zoom, pan and live
//! adjustments applied in the fragment shader.
//!
//! The image is uploaded as the layers of a texture array
//! ([`Source`]), so images larger than iced's device texture limit can
//! still be shown at full resolution.

use std::sync::{Arc, OnceLock};

use bytemuck::{Pod, Zeroable};
use iced::advanced::graphics::Viewport;
use iced::widget::shader::{self, Action};
use iced::{Event, Point, Rectangle, Size, Vector, mouse};

use crate::agx;
use crate::develop::{self, EditUniform, LensUniform};
use crate::lens::{Corrections, LUT_RHO_MAX, LUT_SIZE};
use crate::local::LocalContrast;
use crate::processing::Image;
use crate::raw::WhiteBalance;
use crate::source::{GridUniform, Source};
use crate::wheel;

/// Extra stops of detail per stop at the local contrast slider's maximum.
pub const LOCAL_GAIN: f32 = 0.7;

/// `preview.wgsl`, which calls `develop.wgsl`.
const PREVIEW_SHADER: &str = include_str!("shaders/preview.wgsl");
pub const MIN_SCALE: f32 = 0.01;
pub const MAX_SCALE: f32 = 32.0;
const ZOOM_STEP: f32 = 1.25;
/// Linear 18% grey.
pub const MID_GRAY: f32 = 0.18;
pub const DEFAULT_SATURATION: f32 = 1.30;

/// User adjustments applied at display time, through AgX (see `agx`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adjustments {
    /// Warm (+) / cool (−) shift, −1 … +1, of the as-shot colour
    /// temperature, in mireds (see `WhiteBalance`).
    pub temperature: f32,
    /// Magenta (+) / green (−) shift, −1 … +1: the green gain.
    pub tint: f32,
    /// Linear value subtracted before exposure, as a fraction of
    /// [`agx::WHITE`], −0.05 … +0.05: positive removes haze and flare,
    /// negative lifts black. 0 = unchanged.
    pub black_point: f32,
    /// Stops, −5 … +5, applied to the linear image.
    pub exposure: f32,
    /// Slope of the tone curve where it crosses the diagonal; 1 = no curve.
    pub contrast: f32,
    /// Linear value (after exposure) where the tone curve crosses the
    /// diagonal; it stays put while contrast swings the rest about it.
    pub midpoint: f32,
    /// Applied after the sigmoid. 0 = greyscale, 1 = unchanged, 2 = double.
    pub saturation: f32,
    /// Colour wheel pucks for the shadows, midtones and highlights: right,
    /// down, in the unit disc (see `wheel`). 0, 0 = no tint.
    pub grading: [[f32; 2]; 3],
    /// Brightness of the shadows, midtones and highlights, −1 … +1, applied
    /// last, after the tone curve and grading. 0 = unchanged.
    pub levels: [f32; 3],
    /// Contrast of detail against its surroundings, 0 … 1. 0 = unchanged.
    pub local_contrast: f32,
}

impl Default for Adjustments {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            tint: 0.0,
            black_point: 0.0,
            exposure: 0.0,
            contrast: agx::DEFAULT_CONTRAST,
            midpoint: MID_GRAY,
            saturation: DEFAULT_SATURATION,
            grading: [[0.0; 2]; 3],
            levels: [0.0; 3],
            local_contrast: 0.0,
        }
    }
}

/// Display-encoded offset of a fully pushed wheel, at its strongest channel.
const GRADE_STRENGTH: f32 = 0.2;

/// Stops of gain at a slider end.
const WB_STOPS: f32 = 0.5;

impl Adjustments {
    /// Camera RGB gains for the temperature and tint, applied to the
    /// already white-balanced demosaic output. Without `white` the
    /// temperature falls back to plain red/blue gains.
    pub fn white_balance_gains(&self, white: Option<WhiteBalance>) -> [f32; 3] {
        let [r, _, b] = match white {
            Some(white) => white.gains(self.temperature),
            None => [
                (self.temperature * WB_STOPS).exp2(),
                1.0,
                (-self.temperature * WB_STOPS).exp2(),
            ],
        };
        [r, (-self.tint * WB_STOPS).exp2(), b]
    }

    /// Display-encoded RGB offsets for the three wheels, w unused: the
    /// wheel's hue with its brightness removed, scaled by the square of
    /// its distance from the centre so small moves near the middle are gentle.
    pub fn grade_tints(&self) -> [[f32; 4]; 3] {
        self.grading.map(|position| {
            let (angle, strength) = wheel::polar(position);
            let rgb = wheel::hue_rgb(angle);
            let mean = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
            let [r, g, b] = rgb.map(|c| (c - mean) * strength * strength * GRADE_STRENGTH);
            [r, g, b, 0.0]
        })
    }

    /// `cam_to_srgb` with the gains folded in: M · diag(gains).
    pub fn cam_to_srgb(&self, m: [[f32; 3]; 3], white: Option<WhiteBalance>) -> [[f32; 3]; 3] {
        let g = self.white_balance_gains(white);
        m.map(|row| [row[0] * g[0], row[1] * g[1], row[2] * g[2]])
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Zoom {
    /// Scale the whole image to fit the pane.
    Fit,
    /// Logical pixels per image pixel.
    Scale(f32),
}

/// Zoom and pan state, owned by the application.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewState {
    pub zoom: Zoom,
    /// Image-space point at the centre of the pane (ignored when fitting).
    pub center: Point,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            zoom: Zoom::Fit,
            center: Point::ORIGIN,
        }
    }
}

impl ViewState {
    pub fn fit_scale(pane: Size, image: Size) -> f32 {
        (pane.width / image.width).min(pane.height / image.height)
    }

    /// Concrete (scale, center) for the given pane and image size.
    pub fn resolve(&self, pane: Size, image: Size) -> (f32, Point) {
        match self.zoom {
            Zoom::Fit => (
                Self::fit_scale(pane, image),
                Point::new(image.width / 2.0, image.height / 2.0),
            ),
            Zoom::Scale(scale) => (scale, self.center),
        }
    }

    /// Zooms by `factor`, keeping the image point under `anchor` (offset from
    /// the pane centre, in logical pixels) fixed on screen.
    pub fn zoom_by(&self, factor: f32, anchor: Vector, pane: Size, image: Size) -> Self {
        let (scale, center) = self.resolve(pane, image);
        let new_scale = (scale * factor).clamp(MIN_SCALE, MAX_SCALE);
        let under_cursor = center + anchor * (1.0 / scale);
        Self {
            zoom: Zoom::Scale(new_scale),
            center: under_cursor - anchor * (1.0 / new_scale),
        }
    }

    pub fn zoom_step(&self, zoom_in: bool, pane: Size, image: Size) -> Self {
        let factor = if zoom_in { ZOOM_STEP } else { 1.0 / ZOOM_STEP };
        self.zoom_by(factor, Vector::ZERO, pane, image)
    }

    /// Pans by a screen-space delta in logical pixels.
    pub fn pan_by(&self, delta: Vector, pane: Size, image: Size) -> Self {
        let (scale, center) = self.resolve(pane, image);
        Self {
            zoom: Zoom::Scale(scale),
            center: center - delta * (1.0 / scale),
        }
    }

    /// The same view after turning the shown image (of size `image`, already
    /// rotated) by `turn`.
    pub fn rotated(&self, turn: Rotation, image: Size) -> Self {
        Self {
            zoom: self.zoom,
            center: turn.apply(self.center, image),
        }
    }

    pub fn actual_size(image: Size) -> Self {
        Self {
            zoom: Zoom::Scale(1.0),
            center: Point::new(image.width / 2.0, image.height / 2.0),
        }
    }
}

/// Clockwise quarter turns applied to the image for display. Zoom and pan
/// ([`ViewState`]) work in the rotated image's coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rotation(u8);

impl Rotation {
    pub const CLOCKWISE: Self = Self(1);
    pub const COUNTER_CLOCKWISE: Self = Self(3);

    /// This rotation followed by `turn`.
    pub fn then(self, turn: Self) -> Self {
        Self((self.0 + turn.0) % 4)
    }

    /// The size of `image` once rotated.
    pub fn size(self, image: Size) -> Size {
        if self.0 % 2 == 1 {
            Size::new(image.height, image.width)
        } else {
            image
        }
    }

    /// Where point `p` of `image` lands once rotated.
    pub fn apply(self, p: Point, image: Size) -> Point {
        match self.0 {
            0 => p,
            1 => Point::new(image.height - p.y, p.x),
            2 => Point::new(image.width - p.x, image.height - p.y),
            _ => Point::new(p.y, image.width - p.x),
        }
    }
}

impl Rotation {
    /// The vector `v` turned like the image.
    pub fn vector(self, v: Vector) -> Vector {
        match self.0 {
            0 => v,
            1 => Vector::new(-v.y, v.x),
            2 => Vector::new(-v.x, -v.y),
            _ => Vector::new(v.y, -v.x),
        }
    }

    /// Where the point `q` of the rotated image lies in the unrotated image
    /// of size `image` (the inverse of [`Rotation::apply`], as
    /// `preview.wgsl`'s `to_image`).
    pub fn unapply(self, q: Point, image: Size) -> Point {
        match self.0 {
            0 => q,
            1 => Point::new(q.y, image.height - q.x),
            2 => Point::new(image.width - q.x, image.height - q.y),
            _ => Point::new(image.width - q.y, q.x),
        }
    }
}

const MIN_CROP_ZOOM: f32 = 0.5;
const MAX_CROP_ZOOM: f32 = 20.0;
/// Space around the frame in crop mode, as a fraction of the pane's short side.
const FRAME_MARGIN: f32 = 0.07;
/// At least this much room (logical pixels) around the frame, so the crop
/// bar fits under it.
const MIN_FRAME_MARGIN: f32 = 76.0;
/// Corner radius of the frame in crop mode, as a fraction of its short side.
pub const FRAME_CORNER: f32 = 0.025;

/// The export frame: a window on the (quarter-turned) image with the image's
/// aspect, free to move, zoom and tilt against the image. The result of the
/// export is what lies inside it, at one image pixel per output pixel.
///
/// *Frame space* has its origin at the frame's top left and the image's pixel
/// units; the preview and the export both work in it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crop {
    /// Image point at the frame's centre, from the rotated image's centre.
    pub offset: Vector,
    /// 1 = the frame spans the whole width of the rotated image.
    pub zoom: f32,
    /// Clockwise tilt of the image against the frame, in radians.
    pub angle: f32,
    pub aspect: Aspect,
}

impl Default for Crop {
    fn default() -> Self {
        Self {
            offset: Vector::ZERO,
            zoom: 1.0,
            angle: 0.0,
            aspect: Aspect::Original,
        }
    }
}

/// The shape of the crop frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aspect {
    /// The image's own.
    Original,
    /// Width to height.
    Ratio(u8, u8),
}

impl Aspect {
    pub const ALL: [Aspect; 11] = [
        Aspect::Original,
        Aspect::Ratio(1, 1),
        Aspect::Ratio(3, 2),
        Aspect::Ratio(2, 3),
        Aspect::Ratio(4, 3),
        Aspect::Ratio(3, 4),
        Aspect::Ratio(5, 4),
        Aspect::Ratio(4, 5),
        Aspect::Ratio(16, 9),
        Aspect::Ratio(9, 16),
        Aspect::Ratio(2, 1),
    ];
}

impl std::fmt::Display for Aspect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Aspect::Original => f.write_str("Original"),
            Aspect::Ratio(w, h) => write!(f, "{w}:{h}"),
        }
    }
}

fn turned(v: Vector, angle: f32) -> Vector {
    let (sin, cos) = angle.sin_cos();
    Vector::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos)
}

impl Crop {
    /// The frame's size in image pixels, for a rotated image of size `image`:
    /// the largest rectangle of the frame's shape inside the image, divided
    /// by the zoom.
    pub fn frame_size(&self, image: Size) -> Size {
        let base = match self.aspect {
            Aspect::Original => image,
            Aspect::Ratio(w, h) => {
                let shape = f32::from(w) / f32::from(h);
                if image.width / image.height > shape {
                    Size::new(image.height * shape, image.height)
                } else {
                    Size::new(image.width, image.width / shape)
                }
            }
        };
        Size::new(base.width / self.zoom, base.height / self.zoom)
    }

    /// The frame's width over its height.
    pub fn frame_aspect(&self, image: Size) -> f32 {
        let frame = self.frame_size(image);
        frame.width / frame.height
    }

    /// The point of the rotated image at `f` in frame space.
    pub fn to_image(&self, f: Point, image: Size) -> Point {
        let frame = self.frame_size(image);
        let from_centre = Vector::new(f.x - frame.width / 2.0, f.y - frame.height / 2.0);
        let v = turned(from_centre, -self.angle);
        Point::new(
            image.width / 2.0 + self.offset.x + v.x,
            image.height / 2.0 + self.offset.y + v.y,
        )
    }

    /// Logical pixels per image pixel in crop mode, where the frame has a
    /// fixed size in the pane.
    pub fn scale(&self, pane: Size, image: Size) -> f32 {
        frame_rect(pane, self.frame_aspect(image)).width / self.frame_size(image).width
    }

    /// Moves the image with a screen-space drag of `delta` at `scale`.
    pub fn pan_by(&self, delta: Vector, scale: f32) -> Self {
        let d = turned(delta, -self.angle);
        Self {
            offset: self.offset - d * (1.0 / scale),
            ..*self
        }
    }

    /// Zooms by `factor` keeping the image point under `anchor` (offset from
    /// the frame's centre, in logical pixels) fixed on screen.
    pub fn zoom_by(&self, factor: f32, anchor: Vector, scale: f32) -> Self {
        let zoom = (self.zoom * factor).clamp(MIN_CROP_ZOOM, MAX_CROP_ZOOM);
        let factor = zoom / self.zoom;
        let a = turned(anchor, -self.angle);
        Self {
            // The point under the anchor is offset + a / scale, before and after.
            offset: self.offset + a * (1.0 / scale) - a * (1.0 / (scale * factor)),
            zoom,
            ..*self
        }
    }

    /// Tilts the image by `delta` radians clockwise about the frame's centre.
    pub fn tilt_by(&self, delta: f32) -> Self {
        self.with_angle(self.angle + delta)
    }

    /// Sets the tilt, in radians, wrapped to ±180°.
    pub fn with_angle(&self, angle: f32) -> Self {
        use std::f32::consts::{PI, TAU};
        Self {
            angle: (angle + PI).rem_euclid(TAU) - PI,
            ..*self
        }
    }

    /// The same crop after turning the image by `turn`.
    pub fn rotated(&self, turn: Rotation) -> Self {
        Self {
            offset: turn.vector(self.offset),
            ..*self
        }
    }
}

/// The frame in crop mode: the largest rectangle of `aspect` (width over
/// height) centred in the pane, leaving a margin.
pub fn frame_rect(pane: Size, aspect: f32) -> Rectangle {
    let margin = (pane.width.min(pane.height) * FRAME_MARGIN).max(MIN_FRAME_MARGIN);
    let (w, h) = ((pane.width - 2.0 * margin).max(1.0), (pane.height - 2.0 * margin).max(1.0));
    let width = w.min(h * aspect);
    let size = Size::new(width, width / aspect);
    Rectangle::new(
        Point::new((pane.width - size.width) / 2.0, (pane.height - size.height) / 2.0),
        size,
    )
}

/// Interactions the preview reports back to the application.
#[derive(Debug, Clone, Copy)]
pub enum Interaction {
    Resized(Size),
    /// Zoom by a factor around an anchor offset from the pane centre.
    Zoom {
        factor: f32,
        anchor: Vector,
    },
    /// Pan by a screen-space delta.
    Pan(Vector),
    /// Turn clockwise about the pane's centre, in radians (right drag, or a
    /// horizontal drag above or below the frame).
    Rotate(f32),
}

/// The `shader::Program` for the preview pane.
pub struct Preview<'a, Message> {
    pub image: Option<&'a Image>,
    pub view: ViewState,
    pub adjustments: Adjustments,
    pub rotation: Rotation,
    pub crop: Crop,
    /// Crop mode: the frame stays put in the pane and the image moves under
    /// it, the part outside shaded.
    pub cropping: bool,
    /// The distortion and vignetting corrections to draw with.
    pub lens: Corrections,
    pub on_interaction: fn(Interaction) -> Message,
}

/// What a drag does in crop mode, by where it starts relative to the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Drag {
    /// Inside the frame: move the image.
    #[default]
    Pan,
    /// Left or right of the frame: up zooms in, down zooms out.
    Zoom,
    /// Above the frame, dragging right turns the image clockwise; below it
    /// the other way, as on a wheel seen from its centre.
    Rotate { above: bool },
}

/// Logical pixels of dragging per factor `e` of zoom, and per radian of tilt.
const ZOOM_DRAG: f32 = 200.0;
const ROTATE_DRAG: f32 = 600.0;

#[derive(Default)]
pub struct State {
    size: Size,
    drag_from: Option<Point>,
    drag: Drag,
    rotate_from: Option<Point>,
}

impl<Message> Preview<'_, Message> {
    /// The drag started at `position` (in the pane) in crop mode.
    fn drag_at(&self, position: Point, pane: Size) -> Drag {
        let Some(image) = self.image else {
            return Drag::Pan;
        };
        let rotated = self.rotation.size(Size::new(image.width as f32, image.height as f32));
        let frame = frame_rect(pane, self.crop.frame_aspect(rotated));
        // How far outside the frame along each axis.
        let over_x = (frame.x - position.x).max(position.x - (frame.x + frame.width)).max(0.0);
        let over_y = (frame.y - position.y).max(position.y - (frame.y + frame.height)).max(0.0);
        if over_x == 0.0 && over_y == 0.0 {
            Drag::Pan
        } else if over_x >= over_y {
            Drag::Zoom
        } else {
            Drag::Rotate { above: position.y < frame.y }
        }
    }
}

impl<Message> shader::Program<Message> for Preview<'_, Message> {
    type State = State;
    type Primitive = PreviewPrimitive;

    fn update(
        &self,
        state: &mut State,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<Action<Message>> {
        let publish = |i| Some(Action::publish((self.on_interaction)(i)).and_capture());

        if bounds.size() != state.size {
            state.size = bounds.size();
            return Some(Action::publish((self.on_interaction)(
                Interaction::Resized(bounds.size()),
            )));
        }
        self.image?;

        match event {
            Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                let position = cursor.position_in(bounds)?;
                let lines = match *delta {
                    mouse::ScrollDelta::Lines { y, .. } => y,
                    mouse::ScrollDelta::Pixels { y, .. } => y / 50.0,
                };
                let anchor = position - Point::new(bounds.width / 2.0, bounds.height / 2.0);
                publish(Interaction::Zoom {
                    factor: ZOOM_STEP.powf(lines),
                    anchor,
                })
            }
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                state.drag_from = Some(cursor.position_over(bounds)?);
                state.drag = match cursor.position_in(bounds) {
                    Some(position) if self.cropping => self.drag_at(position, bounds.size()),
                    _ => Drag::Pan,
                };
                Some(Action::capture())
            }
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Right)) if self.cropping => {
                state.rotate_from = Some(cursor.position_over(bounds)?);
                Some(Action::capture())
            }
            Event::Mouse(mouse::Event::CursorMoved { position }) => {
                if let Some(from) = state.rotate_from {
                    state.rotate_from = Some(*position);
                    let centre = Point::new(bounds.width / 2.0, bounds.height / 2.0);
                    let angle = |p: Point| (p.y - centre.y).atan2(p.x - centre.x);
                    let delta = angle(*position) - angle(from);
                    // Across the ±180° seam the short way round.
                    let delta = (delta + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                        - std::f32::consts::PI;
                    return publish(Interaction::Rotate(delta));
                }
                let from = state.drag_from?;
                state.drag_from = Some(*position);
                let delta = *position - from;
                publish(match state.drag {
                    Drag::Pan => Interaction::Pan(delta),
                    Drag::Zoom => Interaction::Zoom {
                        factor: (-delta.y / ZOOM_DRAG).exp(),
                        anchor: Vector::ZERO,
                    },
                    Drag::Rotate { above } => {
                        let turn = delta.x / ROTATE_DRAG;
                        Interaction::Rotate(if above { turn } else { -turn })
                    }
                })
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                state.drag_from.take().map(|_| Action::capture())
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Right)) => {
                state.rotate_from.take().map(|_| Action::capture())
            }
            _ => None,
        }
    }

    fn draw(&self, _state: &State, _cursor: mouse::Cursor, bounds: Rectangle) -> PreviewPrimitive {
        let Some(image) = self.image else {
            return PreviewPrimitive { image: None };
        };
        let image_size = Size::new(image.width as f32, image.height as f32);
        let rotated = self.rotation.size(image_size);
        let frame = self.crop.frame_size(rotated);
        let (scale, center) = if self.cropping {
            (
                self.crop.scale(bounds.size(), rotated),
                Point::new(frame.width / 2.0, frame.height / 2.0),
            )
        } else {
            self.view.resolve(bounds.size(), frame)
        };
        PreviewPrimitive {
            image: Some(PrimitiveImage {
                image: image.clone(),
                scale,
                center,
                adjustments: self.adjustments,
                rotation: self.rotation,
                crop: self.crop,
                cropping: self.cropping,
                lens: self.lens,
                point: false,
            }),
        }
    }

    fn mouse_interaction(
        &self,
        state: &State,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if self.image.is_none() {
            mouse::Interaction::default()
        } else if state.drag_from.is_some() {
            match state.drag {
                Drag::Pan => mouse::Interaction::Grabbing,
                Drag::Zoom => mouse::Interaction::ResizingVertically,
                Drag::Rotate { .. } => mouse::Interaction::ResizingHorizontally,
            }
        } else if let Some(position) = cursor.position_in(bounds) {
            match self.cropping.then(|| self.drag_at(position, bounds.size())) {
                Some(Drag::Zoom) => mouse::Interaction::ResizingVertically,
                Some(Drag::Rotate { .. }) => mouse::Interaction::ResizingHorizontally,
                _ => mouse::Interaction::Grab,
            }
        } else {
            mouse::Interaction::default()
        }
    }
}

#[derive(Debug)]
pub struct PreviewPrimitive {
    image: Option<PrimitiveImage>,
}

/// What one draw of the image shows, and how.
#[derive(Debug)]
pub struct PrimitiveImage {
    pub image: Image,
    /// Target pixels per image pixel.
    pub scale: f32,
    /// The frame point at the centre of the target.
    pub center: Point,
    pub adjustments: Adjustments,
    pub rotation: Rotation,
    pub crop: Crop,
    pub cropping: bool,
    /// Distortion and vignetting from the image's lens profile to apply
    /// (CA is the demosaic stage's).
    pub lens: Corrections,
    /// Each target pixel is the output pixel under its centre, at any scale
    /// (for drawing a subset of the export's pixels).
    pub point: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewUniform {
    edit: EditUniform,
    grid: GridUniform,
    /// Frame point at the target's centre.
    center: [f32; 2],
    viewport: [f32; 2],
    scale: f32,
    encode_srgb: u32,
    rotation: u32,
    /// 1 in crop mode: shade what is outside the frame instead of hiding it.
    shade: f32,
    /// Rotated-image point at the frame's centre.
    crop_center: [f32; 2],
    frame_size: [f32; 2],
    /// Cosine and sine of the tilt.
    crop_cs: [f32; 2],
    /// Radius of the shaded frame's corners, in frame space (0 = square).
    corner: f32,
    point: u32,
    lens: LensUniform,
}

pub struct PreviewPipeline {
    pipeline: wgpu::RenderPipeline,
    view_buffer: wgpu::Buffer,
    /// Group 0, once there is an image; built again for a new image or a
    /// new local contrast base.
    view_group: Option<wgpu::BindGroup>,
    view_layout: wgpu::BindGroupLayout,
    /// Trilinear, clamped: the image and its mipmaps.
    sampler: wgpu::Sampler,
    local: LocalContrast,
    encode_srgb: bool,
    source: Option<Arc<Source>>,
    /// The image whose upload failed last and why, so it is not tried
    /// every frame.
    failed: Option<(u64, String)>,
    /// The lens table ([`LUT_SIZE`] texels) and what it was made for.
    lens_lut: wgpu::Texture,
    lens_key: Option<(u64, Corrections)>,
    lens: LensUniform,
}

/// iced's device and queue, kept when the preview first builds its pipeline
/// so the export can render on them.
static GPU: OnceLock<(wgpu::Device, wgpu::Queue)> = OnceLock::new();

/// iced's device and queue, once the preview has been drawn.
pub fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    GPU.get().cloned()
}

impl shader::Pipeline for PreviewPipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        GPU.get_or_init(|| (device.clone(), queue.clone()));
        Self::with_format(device, format)
    }
}

impl PreviewPipeline {
    /// The pipeline drawing into targets of `format` (the window's, or the
    /// export's).
    pub fn with_format(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(develop::shader("preview", PREVIEW_SHADER));

        let texture_entry = |binding, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview view"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                sampler_entry(1),
                // The local contrast base and its sampler.
                texture_entry(2, wgpu::TextureViewDimension::D2),
                sampler_entry(3),
                // The image.
                texture_entry(4, wgpu::TextureViewDimension::D2Array),
                // The lens table, read with textureLoad.
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preview"),
            bind_group_layouts: &[&view_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            view_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("preview view"),
                size: std::mem::size_of::<ViewUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            view_group: None,
            view_layout,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("preview"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            local: LocalContrast::new(device),
            encode_srgb: !format.is_srgb(),
            source: None,
            failed: None,
            lens_lut: device.create_texture(&wgpu::TextureDescriptor {
                label: Some("lens table"),
                size: wgpu::Extent3d {
                    width: LUT_SIZE as u32,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }),
            lens_key: None,
            lens: LensUniform::zeroed(),
        }
    }

    /// Fills the lens table for `image` with the distortion and vignetting
    /// in `on`, unless it already is.
    fn update_lens(&mut self, queue: &wgpu::Queue, image: &Image, on: Corrections) {
        let on = on.without(Corrections::CHROMATIC);
        if self.lens_key == Some((image.id, on)) {
            return;
        }
        self.lens_key = Some((image.id, on));
        let (half_diagonal, lut) = match &image.profile {
            Some(profile) => profile.lut(image.width, image.height, on),
            // The identity.
            None => (1.0, vec![[1.0, 0.0, 0.0, 1.0]; LUT_SIZE]),
        };
        self.lens = LensUniform {
            centre: [image.width as f32 / 2.0, image.height as f32 / 2.0],
            inv_half_diagonal: 1.0 / half_diagonal,
            lut_max: LUT_RHO_MAX,
        };
        queue.write_texture(
            self.lens_lut.as_image_copy(),
            bytemuck::cast_slice(&lut),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(LUT_SIZE as u32 * 16),
                rows_per_image: Some(1),
            },
            self.lens_lut.size(),
        );
    }

    /// Group 0 again, for a new image or local contrast base.
    fn rebuild_view_group(&mut self, device: &wgpu::Device) {
        let Some(source) = &self.source else {
            return;
        };
        self.view_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview view"),
            layout: &self.view_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.view_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(self.local.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.local.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&source.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.lens_lut.create_view(&Default::default())),
                },
            ],
        }));
    }
}

impl shader::Primitive for PreviewPrimitive {
    type Pipeline = PreviewPipeline;

    fn prepare(
        &self,
        pipeline: &mut PreviewPipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        _viewport: &Viewport,
    ) {
        if let Some(p) = &self.image {
            let new = pipeline.failed.as_ref().is_none_or(|(id, _)| *id != p.image.id);
            if let Err(error) = pipeline.prepare_image(device, queue, p, [bounds.width, bounds.height])
                && new
            {
                crate::log::write(&format!("Error: {error}"));
            }
        }
    }

    fn draw(&self, pipeline: &PreviewPipeline, render_pass: &mut wgpu::RenderPass<'_>) -> bool {
        if self.image.is_some() {
            pipeline.draw_image(render_pass);
        }
        true
    }
}

impl PreviewPipeline {
    /// Gets `p` ready to draw into a target of `viewport` units (logical
    /// pixels on screen): uploads the image if new (or takes the upload
    /// another pipeline made of it), writes the uniform and builds the local
    /// contrast base if needed.
    /// Fails if the image cannot be uploaded (out of memory); then nothing
    /// is drawn, and the upload is not tried again for the same image.
    pub fn prepare_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        p: &PrimitiveImage,
        viewport: [f32; 2],
    ) -> Result<(), String> {
        if self.source.as_ref().is_none_or(|s| s.image_id != p.image.id) {
            if let Some((_, error)) = self.failed.as_ref().filter(|(id, _)| *id == p.image.id) {
                return Err(error.clone());
            }
            // Let go of the old image before the new one takes the memory.
            self.release();
            match Source::get(device, queue, &p.image) {
                Ok(source) => self.source = Some(source),
                Err(error) => {
                    self.failed = Some((p.image.id, error.clone()));
                    return Err(error);
                }
            }
            self.failed = None;
            self.rebuild_view_group(device);
        }
        let Some(source) = self.source.clone() else {
            return Ok(());
        };
        self.update_lens(queue, &p.image, p.lens);

        let image_size = Size::new(p.image.width as f32, p.image.height as f32);
        let rotated = p.rotation.size(image_size);
        let frame = p.crop.frame_size(rotated);
        let edit = EditUniform::new(&p.adjustments, &p.image);
        let cam_to_srgb = edit.cam_to_srgb;
        let uniform = ViewUniform {
            edit,
            grid: source.grid,
            center: [p.center.x, p.center.y],
            viewport,
            scale: p.scale,
            encode_srgb: self.encode_srgb as u32,
            rotation: u32::from(p.rotation.0),
            shade: p.cropping as u32 as f32,
            crop_center: [rotated.width / 2.0 + p.crop.offset.x, rotated.height / 2.0 + p.crop.offset.y],
            frame_size: [frame.width, frame.height],
            crop_cs: [p.crop.angle.cos(), p.crop.angle.sin()],
            corner: if p.cropping {
                FRAME_CORNER * frame.width.min(frame.height)
            } else {
                0.0
            },
            point: p.point as u32,
            lens: self.lens,
        };
        queue.write_buffer(&self.view_buffer, 0, bytemuck::bytes_of(&uniform));

        if p.adjustments.local_contrast != 0.0 {
            let lut = self.lens_lut.create_view(&Default::default());
            let lens = (self.lens, &lut, self.lens_key.map_or(Corrections::NONE, |(_, on)| on));
            if self.local.update(device, queue, &source, cam_to_srgb, lens) {
                self.rebuild_view_group(device);
            }
        }
        Ok(())
    }

    /// Drops the image, the bind group that holds it and the local contrast
    /// base made from it: the image's GPU memory is freed once nothing else
    /// holds it.
    pub fn release(&mut self) {
        self.view_group = None;
        self.source = None;
        self.local.clear();
    }

    /// Draws the image prepared last into `render_pass`, over its whole
    /// viewport.
    pub fn draw_image(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        let Some(view_group) = &self.view_group else {
            return;
        };
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, view_group, &[]);
        render_pass.draw(0..6, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE: Size = Size::new(6000.0, 4000.0);

    fn close(a: Point, b: Point) {
        assert!((a.x - b.x).abs() < 1e-2 && (a.y - b.y).abs() < 1e-2, "{a:?} vs {b:?}");
    }

    #[test]
    fn default_crop_is_the_whole_image() {
        let crop = Crop::default();
        assert_eq!(crop.frame_size(IMAGE), IMAGE);
        close(crop.to_image(Point::new(10.0, 20.0), IMAGE), Point::new(10.0, 20.0));
    }

    #[test]
    fn frame_takes_the_chosen_shape_inside_the_image() {
        for aspect in Aspect::ALL {
            let crop = Crop { aspect, ..Crop::default() };
            let frame = crop.frame_size(IMAGE);
            assert!(frame.width <= IMAGE.width + 1e-3 && frame.height <= IMAGE.height + 1e-3);
            // It touches the image on at least one axis.
            assert!(frame.width >= IMAGE.width - 1e-3 || frame.height >= IMAGE.height - 1e-3);
            if let Aspect::Ratio(w, h) = aspect {
                assert!((frame.width / frame.height - f32::from(w) / f32::from(h)).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn zoom_keeps_the_point_under_the_cursor() {
        let crop = Crop::default().tilt_by(0.4).pan_by(Vector::new(30.0, -12.0), 0.1);
        let scale = crop.scale(Size::new(1200.0, 800.0), IMAGE);
        let anchor = Vector::new(-150.0, 90.0);
        let frame_point = |c: &Crop, s: f32| {
            let f = c.frame_size(IMAGE);
            c.to_image(Point::new(f.width / 2.0 + anchor.x / s, f.height / 2.0 + anchor.y / s), IMAGE)
        };
        let zoomed = crop.zoom_by(1.7, anchor, scale);
        close(frame_point(&crop, scale), frame_point(&zoomed, scale * 1.7));
    }

    #[test]
    fn pan_moves_the_image_with_the_cursor() {
        let crop = Crop::default().tilt_by(0.7);
        let scale = 0.2;
        let delta = Vector::new(40.0, 25.0);
        let panned = crop.pan_by(delta, scale);
        // The image point that was at the frame centre is now `delta / scale`
        // (screen) away from it.
        let f = crop.frame_size(IMAGE);
        let centre = Point::new(f.width / 2.0, f.height / 2.0);
        let moved = Point::new(centre.x + delta.x / scale, centre.y + delta.y / scale);
        close(panned.to_image(moved, IMAGE), crop.to_image(centre, IMAGE));
    }

    #[test]
    fn quarter_turns_agree_with_the_crop() {
        let size = Size::new(6000.0, 4000.0);
        for turn in [Rotation::CLOCKWISE, Rotation::COUNTER_CLOCKWISE, Rotation::CLOCKWISE.then(Rotation::CLOCKWISE)] {
            let p = Point::new(1234.0, 567.0);
            close(turn.unapply(turn.apply(p, size), size), p);
            let v = Vector::new(100.0, -30.0);
            let moved = turn.apply(Point::new(size.width / 2.0 + v.x, size.height / 2.0 + v.y), size);
            let rotated = turn.size(size);
            let w = turn.vector(v);
            close(moved, Point::new(rotated.width / 2.0 + w.x, rotated.height / 2.0 + w.y));
        }
    }

    #[test]
    fn shader_is_valid() {
        let source = format!("{}\n{PREVIEW_SHADER}", develop::SHADER);
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap();
        // The Rust uniform must fill the WGSL struct exactly.
        let view = module.types.iter().find(|(_, t)| t.name.as_deref() == Some("View")).unwrap().1;
        assert_eq!(view.inner.size(module.to_ctx()) as usize, std::mem::size_of::<ViewUniform>());
    }
}
