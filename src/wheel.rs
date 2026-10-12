//! A colour wheel: a ring of 12 hue fields with a puck, for tinting one
//! tonal range. The puck's angle is the tint's hue and its distance from
//! the centre the strength; the centre is neutral. Drag to move it, double
//! click to reset it.

use crate::styling;
use iced::advanced::mouse::{Click, click::Kind};
use iced::widget::Action;
use iced::widget::canvas::{self, Frame, Geometry, Path};
use iced::{Color, Event, Point, Rectangle, Renderer, Theme, mouse};

/// Room around the disc for the puck.
const MARGIN: f32 = 5.0;
/// Colour fields around the ring, the gap between them (radians) and the
/// ring's inner radius as a fraction of the outer.
const FIELDS: usize = 12;
const GAP: f32 = 0.02;
/// Rotation of the fields (radians): half a field back, so one is centred
/// on north.
const OFFSET: f32 = -std::f32::consts::TAU / FIELDS as f32 / 2.0;
/// Opacity of the colour fields: 0 invisible, 1 solid.
const FIELD_ALPHA: f32 = 0.5;
/// Shades the ring: −1 towards black, 0 as is, 1 towards white.
const BRIGHTNESS: f32 = 0.25;
/// Size of the ring relative to the widget, 1 = fills it.
const SCALE: f32 = 0.85;
const RING_INNER: f32 = 0.776;
const ARC_STEPS: usize = 6;

/// Where the colours sit on the wheel: (degrees clockwise from north, HSV
/// hue in degrees), interpolated linearly between entries. Hue falls as the
/// wheel angle rises, and the last entry repeats the first a turn later.
/// Yellow 0°, orange 60°, red 120°, purple 180°, blue 240°, green 300°.
const HUE_STOPS: [(f32, f32); 5] = [
    (0.0, 60.0),
    (120.0, 0.0),
    (240.0, -120.0),
    (300.0, -240.0),
    (360.0, -300.0),
];

/// Fully saturated RGB for a wheel angle in radians. Used for both the
/// ring's colours and the tint, so they always agree.
pub fn hue_rgb(angle: f32) -> [f32; 3] {
    let compass = (90.0 - angle.to_degrees()).rem_euclid(360.0);
    let i = HUE_STOPS
        .windows(2)
        .position(|w| compass < w[1].0)
        .unwrap_or(HUE_STOPS.len() - 2);
    let ((c0, h0), (c1, h1)) = (HUE_STOPS[i], HUE_STOPS[i + 1]);
    let hue = h0 + (h1 - h0) * (compass - c0) / (c1 - c0);
    let sector = hue.rem_euclid(360.0) / 60.0;
    let ramp =
        |offset: f32| (2.0 - ((sector + offset).rem_euclid(6.0) - 3.0).abs()).clamp(0.0, 1.0);
    [ramp(3.0), ramp(1.0), ramp(5.0)]
}

/// Puck position (right, down; unit disc) → hue angle and strength.
pub fn polar(position: [f32; 2]) -> (f32, f32) {
    (
        (-position[1]).atan2(position[0]),
        position[0].hypot(position[1]).min(1.0),
    )
}

pub struct View<Message> {
    pub position: [f32; 2],
    /// Which wheel this is, handed back to `on_change`.
    pub index: usize,
    pub on_change: fn(usize, [f32; 2]) -> Message,
}

#[derive(Default)]
pub struct State {
    dragging: bool,
    last_click: Option<Click>,
}

/// Centre and radius of the ring, `SCALE` times the largest that fits.
fn geometry(bounds: Rectangle) -> (Point, f32) {
    (
        Point::new(bounds.width / 2.0, bounds.height / 2.0),
        ((bounds.width.min(bounds.height) / 2.0 - MARGIN) * SCALE).max(1.0),
    )
}

impl<Message> View<Message> {
    /// Cursor (relative to the widget) → puck position, kept on the disc.
    fn position_at(&self, bounds: Rectangle, point: Point) -> [f32; 2] {
        let (center, radius) = geometry(bounds);
        let (dx, dy) = ((point.x - center.x) / radius, (point.y - center.y) / radius);
        let length = dx.hypot(dy).max(1.0);
        [dx / length, dy / length]
    }
}

impl<Message> canvas::Program<Message> for View<Message> {
    type State = State;

    fn update(
        &self,
        state: &mut State,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<Action<Message>> {
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let point = cursor.position_in(bounds)?;
                let (center, radius) = geometry(bounds);
                if point.distance(center) > radius + MARGIN {
                    return None;
                }
                let click = Click::new(point, mouse::Button::Left, state.last_click);
                state.last_click = Some(click);
                let position = if click.kind() == Kind::Single {
                    state.dragging = true;
                    self.position_at(bounds, point)
                } else {
                    state.dragging = false;
                    [0.0, 0.0]
                };
                Some(Action::publish((self.on_change)(self.index, position)).and_capture())
            }
            Event::Mouse(mouse::Event::CursorMoved { position }) if state.dragging => {
                let point = Point::new(position.x - bounds.x, position.y - bounds.y);
                Some(
                    Action::publish((self.on_change)(
                        self.index,
                        self.position_at(bounds, point),
                    ))
                    .and_capture(),
                )
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) if state.dragging => {
                state.dragging = false;
                Some(Action::capture())
            }
            _ => None,
        }
    }

    fn mouse_interaction(
        &self,
        state: &State,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        let (center, radius) = geometry(bounds);
        let over = cursor
            .position_in(bounds)
            .is_some_and(|p| p.distance(center) <= radius + MARGIN);
        if state.dragging || over {
            mouse::Interaction::Crosshair
        } else {
            mouse::Interaction::None
        }
    }

    fn draw(
        &self,
        state: &State,
        renderer: &Renderer,
        theme: &Theme,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let (center, radius) = geometry(bounds);
        let inner = radius * RING_INNER;
        let at =
            |angle: f32, r: f32| Point::new(center.x + r * angle.cos(), center.y - r * angle.sin());

        let mut frame = Frame::new(renderer, bounds.size());
        {
            for field in 0..FIELDS {
                let a0 = (field as f32 / FIELDS as f32 * std::f32::consts::TAU) + OFFSET + GAP;
                let a1 =
                    ((field + 1) as f32 / FIELDS as f32 * std::f32::consts::TAU) + OFFSET - GAP;
                let [r, g, b] = std::array::from_fn(|c| {
                    let base = hue_rgb((a0 + a1) / 2.0)[c] * 0.85 + 0.1;
                    let shaded = if BRIGHTNESS < 0.0 {
                        base * (1.0 + BRIGHTNESS)
                    } else {
                        base + (1.0 - base) * BRIGHTNESS
                    };
                    shaded.clamp(0.0, 1.0)
                });
                let path = Path::new(|builder| {
                    builder.move_to(at(a0, radius));
                    for step in 1..=ARC_STEPS {
                        builder
                            .line_to(at(a0 + (a1 - a0) * step as f32 / ARC_STEPS as f32, radius));
                    }
                    for step in (0..=ARC_STEPS).rev() {
                        builder.line_to(at(a0 + (a1 - a0) * step as f32 / ARC_STEPS as f32, inner));
                    }
                    builder.close();
                });
                frame.fill(&path, Color::from_rgba(r, g, b, FIELD_ALPHA));
            }
        }

        let puck = Point::new(
            center.x + self.position[0] * radius,
            center.y + self.position[1] * radius,
        );
        let over = cursor
            .position_in(bounds)
            .is_some_and(|p| p.distance(center) <= radius + MARGIN);
        let (fill, ring) = styling::knob_colors(theme, state.dragging || over);
        frame.fill(&Path::circle(puck, styling::KNOB_RADIUS), ring);
        frame.fill(
            &Path::circle(puck, styling::KNOB_RADIUS - styling::KNOB_RING),
            fill,
        );
        vec![frame.into_geometry()]
    }
}
