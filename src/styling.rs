//! Look and feel, ported from iced-material-icon-browser to iced 0.14's
//! closure-based styles.

use iced::widget::{button, container, pane_grid, rule, scrollable, text, text_input};
use iced::{Background, Border, Color, Theme, border};

pub fn theme() -> Theme {
    Theme::custom(
        "Starfish",
        iced::theme::Palette {
            // [apple color guidelines](https://developer.apple.com/design/human-interface-guidelines/color)
            background: Color::from_rgb8(30, 30, 30),
            text: Color::from_rgb8(221, 221, 221),
            primary: Color::from_rgb8(10, 132, 255),
            success: Color::from_rgb8(48, 209, 81),
            warning: Color::from_rgb8(255, 214, 10),
            danger: Color::from_rgb8(255, 69, 58),
        },
    )
}

fn lighten(color: Color, amount: f32) -> Color {
    Color {
        r: color.r + amount,
        g: color.g + amount,
        b: color.b + amount,
        ..color
    }
}

fn faded(color: Color, alpha: f32) -> Color {
    Color { a: alpha, ..color }
}

fn background(theme: &Theme, lift: f32) -> container::Style {
    container::Style {
        background: Some(lighten(theme.palette().background, lift).into()),
        ..Default::default()
    }
}

/// `radius` rounds the corners that sit on the window's outer corners.
pub fn sidebar(theme: &Theme) -> container::Style {
    background(theme, PREVIEW_LIFT)
}

/// The colour of [`sidebar`]'s background, for what is drawn to look cut out
/// of it (the histogram's rounded corners).
pub fn panel_color(theme: &Theme) -> Color {
    lighten(theme.palette().background, PREVIEW_LIFT)
}

/// The square a thumbnail picture is inscribed in (no fill, but a faint one while
/// `loading`), with a white outline when `selected`.
/// The white outline drawn over a thumbnail's picture when it is selected.
pub fn thumbnail_outline(selected: bool) -> container::Style {
    container::Style {
        border: Border {
            radius: 0.0.into(),
            color: Color::WHITE,
            width: if selected { 2.0 } else { 0.0 },
        },
        ..Default::default()
    }
}

pub fn thumbnail_picture(theme: &Theme, loading: bool, selected: bool) -> container::Style {
    container::Style {
        background: loading.then(|| faded(theme.palette().text, 0.06).into()),
        border: Border {
            radius: THUMBNAIL_INNER_RADIUS.into(),
            color: Color::WHITE,
            width: if selected { 2.0 } else { 0.0 },
        },
        ..Default::default()
    }
}

pub fn preview(theme: &Theme) -> container::Style {
    background(theme, PREVIEW_LIFT)
}

/// How much lighter than the window's background the preview is, and the
/// library and adjustments panels with it.
const PREVIEW_LIFT: f32 = -0.03;

/// The preview pane in crop mode: black around the image.
pub fn preview_cropping(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::BLACK)),
        ..Default::default()
    }
}

/// Borderless icon button in the toolbar.
pub fn toolbar_button(theme: &Theme, status: button::Status) -> button::Style {
    let text = theme.palette().text;
    button::Style {
        background: None,
        text_color: match status {
            button::Status::Hovered => theme.palette().primary,
            button::Status::Pressed => theme.extended_palette().primary.strong.color,
            button::Status::Disabled => faded(text, 0.25),
            button::Status::Active => text,
        },
        ..Default::default()
    }
}

/// Sidebar row button; `selected` rows are filled with the primary colour.
/// Corner radius of the thumbnail picture and its caption strip.
pub const THUMBNAIL_INNER_RADIUS: f32 = 0.0;
const THUMBNAIL_BUTTON_RADIUS: f32 = 6.0;
const ROW_BUTTON_RADIUS: f32 = 6.0;

/// `row_button` with rounder corners, no border and no selected fill: the
/// selection is outlined on the picture itself (`thumbnail_picture`).
pub fn thumbnail_button() -> impl Fn(&Theme, button::Status) -> button::Style {
    let style = row_button(false);
    move |theme, status| {
        let mut style = style(theme, status);
        style.border.radius = THUMBNAIL_BUTTON_RADIUS.into();
        style.border.color = Color::TRANSPARENT;
        style
    }
}

pub fn row_button(selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |theme, status| {
        let palette = theme.extended_palette();
        let (background, border_color) = match (selected, status) {
            (_, button::Status::Disabled) => {
                (Color::TRANSPARENT, faded(theme.palette().text, 0.05))
            }
            (true, button::Status::Hovered) => (
                lighten(palette.primary.base.color, 0.05),
                palette.primary.weak.color,
            ),
            (true, _) => (palette.primary.base.color, palette.primary.weak.color),
            (false, button::Status::Hovered) => (
                faded(theme.palette().text, 0.05),
                faded(theme.palette().text, 0.1),
            ),
            (false, _) => (Color::TRANSPARENT, faded(theme.palette().text, 0.1)),
        };
        button::Style {
            background: Some(background.into()),
            text_color: match status {
                button::Status::Disabled => faded(theme.palette().text, 0.4),
                _ => theme.palette().text,
            },
            border: Border {
                color: border_color,
                width: 1.0,
                radius: ROW_BUTTON_RADIUS.into(),
            },
            ..Default::default()
        }
    }
}

/// The pill behind a segmented control (the sidebar's tab switcher).
pub fn segmented(theme: &Theme) -> container::Style {
    container::Style {
        background: Some(faded(theme.palette().text, BAR_ALPHA).into()),
        border: border::rounded(SEGMENT_RADIUS + 2.0),
        ..Default::default()
    }
}

/// One segment of [`segmented`]; the selected one is a raised pill.
pub fn segment(selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |theme, status| {
        let text = theme.palette().text;
        let (background, text_color) = match (selected, status) {
            (true, _) => (faded(text, 0.16), text),
            (false, button::Status::Hovered | button::Status::Pressed) => {
                (faded(text, 0.06), faded(text, 0.85))
            }
            (false, _) => (Color::TRANSPARENT, faded(text, 0.6)),
        };
        button::Style {
            background: Some(background.into()),
            text_color,
            border: border::rounded(SEGMENT_RADIUS),
            ..Default::default()
        }
    }
}

const SEGMENT_RADIUS: f32 = 6.0;

/// A toggle button with the pill slider's shape and colours: a faint bar
/// when off, with the slider's fill over it when on.
pub fn toggle_button(on: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |theme, status| {
        let text = theme.palette().text;
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        let background = match (on, hovered) {
            (false, false) => faded(text, BAR_ALPHA),
            (false, true) => faded(text, BAR_HOVER_ALPHA),
            (true, false) => faded(text, BAR_ALPHA + FILL_ALPHA),
            (true, true) => faded(text, BAR_HOVER_ALPHA + FILL_ALPHA),
        };
        button::Style {
            background: Some(background.into()),
            text_color: if on { text } else { ADJUSTMENT_GRAY },
            border: border::rounded(crate::pill::RADIUS),
            ..Default::default()
        }
    }
}

/// A collapsible section's heading: faint, brighter under the cursor.
pub fn section_header(theme: &Theme, status: button::Status) -> button::Style {
    let text = theme.palette().text;
    button::Style {
        background: None,
        text_color: match status {
            button::Status::Hovered | button::Status::Pressed => faded(text, 0.7),
            _ => faded(text, 0.35),
        },
        ..Default::default()
    }
}

/// The field a value turns into while it is typed over.
pub fn value_input(theme: &Theme, _status: text_input::Status) -> text_input::Style {
    let text = theme.palette().text;
    text_input::Style {
        background: faded(Color::BLACK, 0.3).into(),
        border: Border {
            color: theme.palette().primary,
            width: 1.0,
            radius: 4.0.into(),
        },
        icon: text,
        placeholder: faded(text, 0.4),
        value: text,
        selection: faded(theme.palette().primary, 0.4),
    }
}

/// A rounded dark capsule floating over the picture (progress, hints).
pub fn pill(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Color::from_rgba(0.0, 0.0, 0.0, 0.65).into()),
        text_color: Some(Color::WHITE),
        border: border::rounded(f32::MAX),
        ..Default::default()
    }
}

/// The floating controls over the preview: dark and translucent, so their
/// white icons and text read over any picture, with 6 px corners.
pub fn floating(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Color::from_rgba(0.0, 0.0, 0.0, 0.6).into()),
        text_color: Some(Color::WHITE),
        border: border::rounded(6),
        ..Default::default()
    }
}

/// A button on [`floating`]: white, the accent colour under the cursor.
pub fn floating_button(theme: &Theme, status: button::Status) -> button::Style {
    let mut style = toolbar_button(theme, status);
    if status == button::Status::Active {
        style.text_color = Color::WHITE;
    } else if status == button::Status::Disabled {
        style.text_color = faded(Color::WHITE, 0.3);
    }
    style
}

/// Marks the preview as a drop target while a file is dragged over it.
pub fn drop_target(theme: &Theme) -> container::Style {
    let primary = theme.palette().primary;
    container::Style {
        background: Some(faded(primary, 0.12).into()),
        border: Border {
            color: primary,
            width: 2.0,
            radius: 0.0.into(),
        },
        ..Default::default()
    }
}

/// How colourful the UI's hue colours are (the mixer's swatches, the hue and
/// saturation tracks, the histogram's lines): 1 as designed, below 1 more
/// pastel (0 is grey), above 1 more vivid.
pub const UI_CHROMA_SCALE: f32 = 1.3;

/// A colour for the UI from Oklab lightness, chroma and hue (degrees),
/// with the chroma scaled by [`UI_CHROMA_SCALE`].
pub fn ui_oklch(lightness: f32, chroma: f32, hue: f32) -> [f32; 3] {
    crate::color_edit::oklch_to_display(lightness, chroma * UI_CHROMA_SCALE, hue)
}

/// Opacity of a colour mixer swatch at rest, and selected or under the cursor.
pub const SWATCH_ALPHA: f32 = 0.6;
pub const SWATCH_ACTIVE_ALPHA: f32 = 1.0;

/// The small round reset badge on an edited colour swatch.
pub fn badge(_theme: &Theme, status: button::Status) -> button::Style {
    let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
    button::Style {
        background: Some(faded(Color::BLACK, if hovered { 1.0 } else { 0.8 }).into()),
        text_color: Color::WHITE,
        border: Border {
            radius: f32::MAX.into(),
            width: 1.0,
            color: faded(Color::WHITE, 0.8),
        },
        ..Default::default()
    }
}

pub fn tooltip(theme: &Theme) -> container::Style {
    container::Style {
        background: Some(lighten(theme.palette().background, 0.12).into()),
        text_color: Some(theme.palette().text),
        border: Border {
            color: faded(theme.palette().text, 0.15),
            width: 1.0,
            radius: 4.0.into(),
        },
        ..Default::default()
    }
}

pub fn divider(_theme: &Theme) -> rule::Style {
    rule::Style {
        color: Color::BLACK,
        radius: 0.0.into(),
        fill_mode: rule::FillMode::Full,
        snap: true,
    }
}

pub fn separator(theme: &Theme) -> rule::Style {
    rule::Style {
        color: faded(theme.palette().text, 0.1),
        ..divider(theme)
    }
}

/// Faint bold section heading ("Categories" in the template).
pub fn heading(theme: &Theme) -> text::Style {
    text::Style {
        color: Some(faded(theme.palette().text, 0.35)),
    }
}

pub fn muted(theme: &Theme) -> text::Style {
    text::Style {
        color: Some(faded(theme.palette().text, 0.6)),
    }
}

pub fn error(theme: &Theme) -> text::Style {
    text::Style {
        color: Some(theme.palette().danger),
    }
}

// The shared palette: every control and the histogram are the text colour at
// these opacities, over the panel, plus the painted tracks' pastel hues.

/// A slider's bar at rest, and the segmented control behind its segments.
pub const BAR_ALPHA: f32 = 0.07;
/// The bar under the pointer.
pub const BAR_HOVER_ALPHA: f32 = 0.11;
/// What is laid over the part of a bar that is filled or on.
pub const FILL_ALPHA: f32 = 0.2;
/// A painted track (and a histogram channel) under its marker.
pub const TRACK_ALPHA: f32 = 0.6;
/// Marks on a bar or plot: the centre tick, the histogram's curve.
pub const TICK_ALPHA: f32 = 0.45;

/// Bright gray for slider fills and knobs and the adjustment titles.
pub const ADJUSTMENT_GRAY: Color = Color::from_rgb(0.8, 0.8, 0.8);

/// Backdrop behind the panes, showing through the gaps as divider lines.
pub fn pane_dividers(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Color::BLACK.into()),
        ..Default::default()
    }
}

/// Highlights the pane divider under the cursor and while it's dragged.
/// `button_held`: a mouse button went down elsewhere (panning the preview,
/// dragging a scrollbar), so a divider the pointer passes over is not lit.
/// A drag of the divider itself is still shown (`picked_split`).
pub fn panes(theme: &Theme, button_held: bool) -> pane_grid::Style {
    let split = pane_grid::Line {
        color: theme.palette().primary,
        width: 2.0,
    };
    pane_grid::Style {
        hovered_region: pane_grid::Highlight {
            background: faded(theme.palette().primary, 0.1).into(),
            border: border::rounded(0),
        },
        picked_split: split,
        hovered_split: if button_held {
            pane_grid::Line {
                color: Color::TRANSPARENT,
                width: 0.0,
            }
        } else {
            split
        },
    }
}

/// A colour chip of the colour mixer's selector: its colour, with a ring in
/// the text colour while it is the selected one.
pub fn swatch(color: Color, selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |theme, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        button::Style {
            background: Some(
                faded(
                    color,
                    if selected || hovered {
                        SWATCH_ACTIVE_ALPHA
                    } else {
                        SWATCH_ALPHA
                    },
                )
                .into(),
            ),
            border: Border {
                radius: f32::MAX.into(),
                width: if selected { 2.0 } else { 0.0 },
                color: theme.palette().text,
            },
            ..Default::default()
        }
    }
}

/// The default scrollbar without its track, so only the handle is painted,
/// and the handle only while the pointer is over the scrollable. The handle
/// is `SCROLLBAR_WIDTH` wide and its area is too, but it is drawn
/// `SCROLLBAR_IDLE_WIDTH` wide (the rest a transparent border) until the
/// pointer is over that area, or it is dragged, when it fills it.
pub fn scrollbar(theme: &Theme, status: scrollable::Status) -> scrollable::Style {
    let mut style = scrollable::default(theme, status);
    let (idle, full) = match status {
        scrollable::Status::Active { .. } => (true, false),
        scrollable::Status::Hovered {
            is_vertical_scrollbar_hovered,
            ..
        } => (false, is_vertical_scrollbar_hovered),
        scrollable::Status::Dragged {
            is_vertical_scrollbar_dragged,
            ..
        } => (false, is_vertical_scrollbar_dragged),
    };
    for rail in [&mut style.vertical_rail, &mut style.horizontal_rail] {
        rail.background = None;
        if idle {
            rail.scroller.background = Background::Color(Color::TRANSPARENT);
        }
        rail.scroller.border = Border {
            color: Color::TRANSPARENT,
            width: if full {
                0.0
            } else {
                (SCROLLBAR_WIDTH - SCROLLBAR_IDLE_WIDTH) / 2.0
            },
            radius: (SCROLLBAR_WIDTH / 2.0).into(),
        };
    }
    style
}

/// Width of the scrollbar and its handle: iced's default.
const SCROLLBAR_WIDTH: f32 = 10.0;
/// How wide the handle is drawn until the pointer is over the scrollbar.
const SCROLLBAR_IDLE_WIDTH: f32 = 4.0;

/// A vertical scrollbar of `SCROLLBAR_WIDTH`.
pub fn scrollbar_direction() -> scrollable::Direction {
    scrollable::Direction::Vertical(
        scrollable::Scrollbar::new()
            .width(SCROLLBAR_WIDTH)
            .scroller_width(SCROLLBAR_WIDTH),
    )
}
