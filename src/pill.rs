//! The pill slider: a rounded rectangle with the control's label at its left and
//! its value at its right, filled from the left up to the value. A slider
//! that is zero in the middle (`centered`) has a short vertical line there
//! and fills from it towards the value, with a translucent overlay. There is
//! no handle. A slider whose bar is painted with what it does (`Track`: the
//! hues, the colour shift of temperature and tint) has the same overlay over
//! its colours, except the hue bar, which has a line at its value instead.
//!
//! A press and drag moves the value from where it was taken (a press never
//! makes it jump to the cursor, as a handle's does), a click on the value
//! asks for it to be typed (`on_value`), and a second click soon after the
//! first resets the control (`on_reset`).

use crate::styling::{BAR_ALPHA, BAR_HOVER_ALPHA, FILL_ALPHA, TICK_ALPHA, TRACK_ALPHA};
use crate::{FONT, FONT_SIZE_STANDARD, color_edit, styling};
use iced::widget::canvas::{self, Frame, Geometry, Path};
use iced::widget::text::Alignment;
use iced::widget::{Action, canvas as canvas_widget, container, stack};
use iced::{
    Color, Element, Event, Length, Padding, Point, Rectangle, Renderer, Size, Theme, alignment,
    keyboard, mouse,
};
use std::ops::RangeInclusive;
use std::rc::Rc;
use std::time::{Duration, Instant};

/// Height of the bar: a little over the text's.
pub const HEIGHT: f32 = FONT_SIZE_STANDARD * 2.0;
/// Radius of the bar's corners.
pub const RADIUS: f32 = 6.0;
/// Space between the bar's end and its text.
pub const TEXT_INSET: f32 = 9.0;
/// A press this close to the right end is on the value.
const VALUE_HIT: f32 = 84.0;
/// How far the cursor must move before a press is a drag.
const DRAG_SLOP: f32 = 3.0;
const DOUBLE_CLICK: Duration = Duration::from_millis(400);

/// What the bar is painted with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Track {
    Plain,
    /// Every hue, left to right.
    Hue,
    /// The gains temperature applies, blue to yellow.
    Temperature,
    /// The gains tint applies, green to magenta.
    Tint,
    /// The hues within a colour's shift of this one (OkLCh degrees).
    HueRange(f32),
    /// A grey to the full colour of this hue.
    Saturation(f32),
}

impl Track {
    /// Gradient stops over the bar, `(offset, colour)`.
    fn stops(self) -> Vec<(f32, Color)> {
        const GREY: f32 = 0.25;
        let encode = |v: f32| {
            let v = v.clamp(0.0, 1.0);
            if v <= 0.003_130_8 {
                12.92 * v
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            }
        };
        let shift = |gains: [f32; 3]| {
            Color::from_rgba(
                encode(GREY * gains[0]),
                encode(GREY * gains[1]),
                encode(GREY * gains[2]),
                TRACK_ALPHA,
            )
        };
        let tracked = |[r, g, b]: [f32; 3]| Color::from_rgba(r, g, b, TRACK_ALPHA);
        match self {
            Track::Plain => Vec::new(),
            Track::Hue => (0..8)
                .map(|i| {
                    let t = i as f32 / 7.0;
                    let [r, g, b] = styling::ui_oklch(0.75, 0.12, t * 360.0);
                    (t, Color::from_rgba(r, g, b, TRACK_ALPHA))
                })
                .collect(),
            Track::HueRange(centre) => (0..5)
                .map(|i| {
                    let t = i as f32 / 4.0;
                    let hue = centre + (t * 2.0 - 1.0) * color_edit::HUE_SHIFT_MAX;
                    (t, tracked(styling::ui_oklch(0.75, 0.12, hue)))
                })
                .collect(),
            Track::Saturation(hue) => [0.0_f32, 0.5, 1.0]
                .map(|t| (t, tracked(styling::ui_oklch(0.7, t * 0.16, hue))))
                .to_vec(),
            Track::Temperature => [0.0_f32, 0.25, 0.5, 0.75, 1.0]
                .map(|u| {
                    let t = u * 2.0 - 1.0;
                    (u, shift([t.exp2(), 1.0, (-t).exp2()]))
                })
                .to_vec(),
            Track::Tint => [0.0_f32, 0.25, 0.5, 0.75, 1.0]
                .map(|u| {
                    let t = u * 2.0 - 1.0;
                    (u, shift([1.0, (-t).exp2(), 1.0]))
                })
                .to_vec(),
        }
    }
}

/// The opacity of a disabled slider: its bar, text and dots.
pub const DISABLED_ALPHA: f32 = 0.4;

/// How far each grey dot behind a stack's front one is above the next, px, and
/// how many of them are drawn.
const STACK_OFFSET: f32 = 4.0;
const STACK_BEHIND_MAX: usize = 2;

/// Dots closer than this, in px, snap together while one is dragged.
const SNAP_PX: f32 = 5.0;

/// How far from a dot's centre a press still picks it, in px.
const MARKER_GRAB: f32 = 9.0;

pub struct Pill<'a, Message> {
    label: &'a str,
    /// Whether the bar draws the label (else widgets laid over it do).
    draw_label: bool,
    /// The value as it is shown, with its unit.
    text: String,
    value: f32,
    range: RangeInclusive<f32>,
    step: f32,
    centered: bool,
    track: Track,
    /// The value is being typed over: its text is left out.
    editing: bool,
    on_change: Rc<dyn Fn(f32) -> Message + 'a>,
    on_release: Option<Message>,
    on_value: Option<Message>,
    on_reset: Option<Message>,
    /// Drawn dimmed and ignoring the pointer.
    disabled: bool,
    /// Dots drawn on the bar (instead of its fill) at these values, in
    /// these colours, each with an id. Dragging a dot moves it (and the dots
    /// on top of it); dragging the bar moves every dot by the same amount.
    markers: Vec<(usize, f32, Color)>,
    on_markers: Option<Rc<dyn Fn(Vec<(usize, f32)>) -> Message + 'a>>,
    /// The dots are drawn (and picked) by a layer above the label.
    knobs_above: bool,
    /// This is that layer: it draws only the dots and takes only presses on
    /// them, so they come before whatever is under them.
    knobs_only: bool,
}

impl<'a, Message: Clone + 'a> Pill<'a, Message> {
    pub fn new(
        value: f32,
        range: RangeInclusive<f32>,
        on_change: impl Fn(f32) -> Message + 'a,
    ) -> Self {
        Self {
            label: "",
            draw_label: true,
            text: String::new(),
            value,
            range,
            step: 0.01,
            centered: false,
            track: Track::Plain,
            editing: false,
            on_change: Rc::new(on_change),
            on_release: None,
            on_value: None,
            on_reset: None,
            disabled: false,
            markers: Vec::new(),
            on_markers: None,
            knobs_above: false,
            knobs_only: false,
        }
    }

    pub fn text(mut self, text: String) -> Self {
        self.text = text;
        self
    }

    pub fn step(mut self, step: f32) -> Self {
        self.step = step;
        self
    }

    /// Zero (the middle of the range) is marked, and the fill starts there.
    pub fn centered(mut self) -> Self {
        self.centered = true;
        self
    }

    pub fn track(mut self, track: Track) -> Self {
        self.track = track;
        self
    }

    pub fn editing(mut self, editing: bool) -> Self {
        self.editing = editing;
        self
    }

    /// Published when a drag ends.
    pub fn on_release(mut self, message: Message) -> Self {
        self.on_release = Some(message);
        self
    }

    /// Published by a click on the value.
    /// Shows dots instead of the fill; `on_move` gets the ids and new values
    /// of the dots a drag moves.
    pub fn markers(
        mut self,
        markers: Vec<(usize, f32, Color)>,
        on_move: impl Fn(Vec<(usize, f32)>) -> Message + 'a,
    ) -> Self {
        self.markers = markers;
        self.on_markers = Some(Rc::new(on_move));
        self
    }

    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    pub fn on_value(mut self, message: Option<Message>) -> Self {
        self.on_value = message;
        self
    }

    /// Published by a second click soon after a first.
    pub fn on_reset(mut self, message: Option<Message>) -> Self {
        self.on_reset = message;
        self
    }

    pub fn into_element(self) -> Element<'a, Message> {
        canvas_widget(self)
            .width(Length::Fill)
            .height(HEIGHT)
            .into()
    }

    /// The bar with `left` laid over its left end, in place of the label the
    /// bar would draw: the label and the icons that go after it. Whatever
    /// of `left` is not a button lets presses through to the bar.
    pub fn into_element_with(mut self, left: Element<'a, Message>) -> Element<'a, Message> {
        self.draw_label = false;
        // The dots go in a layer of their own over the label and its icons.
        let knobs = self.on_markers.is_some().then(|| {
            self.knobs_above = true;
            Pill {
                label: "",
                draw_label: false,
                text: String::new(),
                value: self.value,
                range: self.range.clone(),
                step: self.step,
                centered: self.centered,
                track: self.track,
                editing: false,
                on_change: self.on_change.clone(),
                on_release: None,
                on_value: None,
                on_reset: None,
                disabled: self.disabled,
                markers: self.markers.clone(),
                on_markers: self.on_markers.clone(),
                knobs_above: true,
                knobs_only: true,
            }
            .into_element()
        });
        stack![
            self.into_element(),
            container(left)
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(Padding {
                    left: TEXT_INSET,
                    ..Padding::ZERO
                })
                .align_y(alignment::Vertical::Center),
        ]
        .extend(knobs)
        .height(HEIGHT)
        .into()
    }

    /// The dots under `x` (within reach of it), with the ones stacked on the
    /// nearest: what dragging there moves. `None` if no dot is near.
    fn knob_group(&self, x: f32, width: f32) -> Option<Vec<(usize, f32)>> {
        let x_of = |value: f32| self.fraction(value) * width;
        let nearest = self
            .markers
            .iter()
            .map(|&(_, value, _)| x_of(value))
            .filter(|dot| (dot - x).abs() <= MARKER_GRAB)
            .min_by(|a, b| (a - x).abs().total_cmp(&(b - x).abs()))?;
        Some(
            self.markers
                .iter()
                .filter(|&&(_, value, _)| (x_of(value) - nearest).abs() <= 1.0)
                .map(|&(id, value, _)| (id, value))
                .collect(),
        )
    }

    /// The dots of `group` moved by `dx` px of the bar, kept in range. A dot
    /// that would land within `SNAP_PX` of one that is not being moved lands
    /// on it instead (and the rest of the group moves with it); the other
    /// dot stays put, and the snap is worked out afresh from where the
    /// pointer is, so dragging on lets go of it.
    fn moved_group(&self, group: &[(usize, f32)], dx: f32, width: f32) -> Vec<(usize, f32)> {
        let (lo, hi) = (*self.range.start(), *self.range.end());
        let least = group.iter().map(|g| g.1).fold(f32::MAX, f32::min);
        let most = group.iter().map(|g| g.1).fold(f32::MIN, f32::max);
        let clamp = |delta: f32| delta.clamp(lo - least, hi - most);
        let mut delta = clamp(dx / width * self.span());
        // The nearest approach of a moving dot to a still one, as a change
        // of `delta` and the dot that makes it.
        let px = width / self.span();
        let mut snap: Option<(f32, usize, f32)> = None;
        for &(id, v) in group {
            for &(other, value, _) in &self.markers {
                if group.iter().any(|g| g.0 == other) {
                    continue;
                }
                let change = value - (v + delta);
                if change.abs() * px < SNAP_PX && snap.is_none_or(|s| change.abs() < s.0.abs()) {
                    snap = Some((change, id, value));
                }
            }
        }
        if let Some((change, _, _)) = snap {
            delta = clamp(delta + change);
        }
        group
            .iter()
            .map(|&(id, v)| match snap {
                // Exactly the other dot's value, not the nearest step.
                Some((_, snapped_id, value)) if snapped_id == id => (id, value),
                _ => (id, self.snapped(v + delta)),
            })
            .collect()
    }

    fn span(&self) -> f32 {
        self.range.end() - self.range.start()
    }

    /// Where `value` is along the bar, 0 to 1.
    fn fraction(&self, value: f32) -> f32 {
        ((value - self.range.start()) / self.span()).clamp(0.0, 1.0)
    }

    /// The range's value nearest `value` on the step grid.
    fn snapped(&self, value: f32) -> f32 {
        ((value / self.step).round() * self.step).clamp(*self.range.start(), *self.range.end())
    }
}

#[derive(Default)]
pub struct State {
    drag: Option<Drag>,
    last_click: Option<Instant>,
    /// Shift is down: a drag moves [`FINE`] as far.
    fine: bool,
}

/// How far a drag with Shift held moves, against a plain one.
const FINE: f32 = 0.1;

struct Drag {
    /// The cursor's x and the value when the button went down.
    x: f32,
    value: f32,
    /// The cursor's x at the last move, and the drag so far, in px, each
    /// step counted at the speed (fine or not) it was made at.
    last: f32,
    travel: f32,
    moved: bool,
    on_value: bool,
    /// The dots this drag moves, with their values at the press.
    group: Option<Vec<(usize, f32)>>,
}

impl<Message: Clone> canvas::Program<Message> for Pill<'_, Message> {
    type State = State;

    fn update(
        &self,
        state: &mut State,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<Action<Message>> {
        if self.disabled {
            state.drag = None;
            return None;
        }
        match event {
            Event::Keyboard(keyboard::Event::ModifiersChanged(modifiers)) => {
                state.fine = modifiers.shift();
                None
            }
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let inside = cursor.position_in(bounds)?;
                let group = if self.knobs_only {
                    // Only a press on a dot; anything else goes through.
                    Some(self.knob_group(inside.x, bounds.width)?)
                } else {
                    self.on_markers.as_ref().map(|_| {
                        self.knob_group(inside.x, bounds.width).unwrap_or_else(|| {
                            self.markers.iter().map(|&(id, v, _)| (id, v)).collect()
                        })
                    })
                };
                state.drag = Some(Drag {
                    group,
                    x: cursor.position()?.x,
                    value: self.value,
                    last: cursor.position()?.x,
                    travel: 0.0,
                    moved: false,
                    on_value: inside.x >= bounds.width - VALUE_HIT,
                });
                Some(Action::capture())
            }
            Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                let drag = state.drag.as_mut()?;
                // The cursor, not the event's position: the panel scrolls,
                // and only the cursor is moved with it.
                let x = cursor.position()?.x;
                drag.travel += (x - drag.last) * if state.fine { FINE } else { 1.0 };
                drag.last = x;
                if !drag.moved && (x - drag.x).abs() < DRAG_SLOP {
                    return Some(Action::capture());
                }
                let dx = drag.travel;
                drag.moved = true;
                if let (Some(group), Some(on_markers)) = (&drag.group, &self.on_markers) {
                    let moved = self.moved_group(group, dx, bounds.width);
                    return Some(Action::publish(on_markers(moved)).and_capture());
                }
                let value = self.snapped(drag.value + dx / bounds.width * self.span());
                Some(Action::publish((self.on_change)(value)).and_capture())
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                let drag = state.drag.take()?;
                if drag.moved {
                    return Some(match &self.on_release {
                        Some(message) => Action::publish(message.clone()).and_capture(),
                        None => Action::capture(),
                    });
                }
                let now = Instant::now();
                let double = state
                    .last_click
                    .is_some_and(|t| now.duration_since(t) < DOUBLE_CLICK);
                state.last_click = (!double).then_some(now);
                let message = if double {
                    self.on_reset.clone()
                } else if drag.on_value {
                    self.on_value.clone()
                } else {
                    None
                };
                Some(match message {
                    Some(message) => Action::publish(message).and_capture(),
                    None => Action::capture(),
                })
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
        if self.disabled {
            return mouse::Interaction::None;
        }
        if self.knobs_only {
            // Only over a dot (or while one is dragged), so the layers
            // below see the pointer everywhere else.
            return match cursor.position_in(bounds) {
                _ if state.drag.is_some() => mouse::Interaction::ResizingHorizontally,
                Some(p) if self.knob_group(p.x, bounds.width).is_some() => {
                    mouse::Interaction::ResizingHorizontally
                }
                _ => mouse::Interaction::None,
            };
        }
        if state.drag.as_ref().is_some_and(|d| d.moved) {
            return mouse::Interaction::ResizingHorizontally;
        }
        match cursor.position_in(bounds) {
            Some(p) if self.on_value.is_some() && p.x >= bounds.width - VALUE_HIT => {
                mouse::Interaction::Text
            }
            Some(_) => mouse::Interaction::ResizingHorizontally,
            None => mouse::Interaction::None,
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
        let mut frame = Frame::new(renderer, bounds.size());
        let (w, h) = (bounds.width, bounds.height);
        let ink = theme.palette().text;
        let dim = if self.disabled { DISABLED_ALPHA } else { 1.0 };
        let faded = |color: Color, a: f32| Color { a: a * dim, ..color };
        let knobs = |frame: &mut Frame| {
            // Dots on top of each other (as dragging them treats them) are
            // drawn as a short vertical stack: grey ones behind, above the
            // front one, which has the colour.
            let mut stacks: Vec<(f32, Vec<Color>)> = Vec::new();
            for &(_, value, color) in &self.markers {
                let x = self.fraction(value) * w;
                match stacks.iter_mut().find(|(sx, _)| (sx - x).abs() <= 1.0) {
                    Some((_, colors)) => colors.push(color),
                    None => stacks.push((x, vec![color])),
                }
            }
            let draw = |frame: &mut Frame, at: Point, fill: Color| {
                let dot = Path::circle(at, 4.5);
                frame.fill(&dot, fill);
                frame.stroke(
                    &dot,
                    canvas::Stroke::default()
                        .with_width(1.5)
                        .with_color(faded(ink, 0.9)),
                );
            };
            for (x, colors) in stacks {
                let behind = (colors.len() - 1).min(STACK_BEHIND_MAX);
                // Centred as a whole: the front dot a little below the middle.
                let front = h / 2.0 + STACK_OFFSET * behind as f32 / 2.0;
                for k in (1..=behind).rev() {
                    let at = Point::new(x, front - STACK_OFFSET * k as f32);
                    draw(frame, at, faded(Color::from_rgb(0.5, 0.5, 0.5), 1.0));
                }
                draw(frame, Point::new(x, front), colors[colors.len() - 1]);
            }
        };
        if self.knobs_only {
            knobs(&mut frame);
            return vec![frame.into_geometry()];
        }
        let hovered =
            !self.disabled && (state.drag.is_some() || cursor.position_in(bounds).is_some());
        let pill = Path::rounded_rectangle(Point::ORIGIN, Size::new(w, h), RADIUS.into());

        if self.track == Track::Plain {
            frame.fill(
                &pill,
                faded(
                    ink,
                    if hovered {
                        BAR_HOVER_ALPHA
                    } else {
                        BAR_ALPHA
                    },
                ),
            );
        } else {
            let mut linear =
                canvas::gradient::Linear::new(Point::new(0.0, 0.0), Point::new(w, 0.0));
            for (offset, color) in self.track.stops() {
                linear = linear.add_stop(offset, faded(color, color.a));
            }
            frame.fill(
                &pill,
                canvas::Fill {
                    style: canvas::Style::Gradient(canvas::Gradient::Linear(linear)),
                    ..canvas::Fill::default()
                },
            );
        }

        let at = self.fraction(self.value) * w;
        let origin = if self.centered { w / 2.0 } else { 0.0 };
        // The bar again, clipped to the part between the origin and the
        // value, so its corners at the bar's end stay round. (The clip
        // keeps the frame's coordinates.)
        let (x0, x1) = (origin.min(at), origin.max(at));
        if x1 - x0 >= 0.5 && self.on_markers.is_none() {
            frame.with_clip(
                Rectangle::new(Point::new(x0, 0.0), Size::new(x1 - x0, h)),
                |frame| {
                    frame.fill(&pill, faded(ink, FILL_ALPHA));
                },
            );
        }
        if self.centered {
            frame.fill_rectangle(
                Point::new(origin - 0.5, h * 0.25),
                Size::new(1.0, h * 0.5),
                faded(ink, TICK_ALPHA),
            );
        }
        if !self.knobs_above {
            knobs(&mut frame);
        }
        let text = |content: String, x: f32, align: Alignment, color: Color| canvas::Text {
            content,
            position: Point::new(x, h / 2.0),
            color,
            size: FONT_SIZE_STANDARD.into(),
            font: FONT,
            align_x: align,
            align_y: alignment::Vertical::Center,
            ..canvas::Text::default()
        };
        if self.draw_label {
            frame.fill_text(text(
                self.label.to_string(),
                TEXT_INSET,
                Alignment::Left,
                faded(styling::ADJUSTMENT_GRAY, styling::ADJUSTMENT_GRAY.a),
            ));
        }
        if !self.editing {
            frame.fill_text(text(
                self.text.clone(),
                w - TEXT_INSET,
                Alignment::Right,
                faded(ink, 1.0),
            ));
        }
        vec![frame.into_geometry()]
    }
}
