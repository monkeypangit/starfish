//! Look and feel, ported from iced-material-icon-browser to iced 0.14's
//! closure-based styles.

use iced::widget::{button, container, pane_grid, pick_list, rule, slider, text, text_input};
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
    background(theme, 0.05)
}

/// The black square a thumbnail picture is inscribed in.
pub fn thumbnail_picture(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Color::BLACK.into()),
        border: Border {
            radius: THUMBNAIL_INNER_RADIUS.into(),
            ..Border::default()
        },
        ..Default::default()
    }
}

pub fn toolbar(theme: &Theme) -> container::Style {
    background(theme, 0.03)
}

pub fn preview(theme: &Theme) -> container::Style {
    background(theme, -0.03)
}

/// The preview pane in crop mode: black around the image.
pub fn preview_cropping(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::BLACK)),
        ..Default::default()
    }
}

/// Soft white for the crop bar's text and icons.
pub const CROP_TEXT: Color = Color::from_rgba(1.0, 1.0, 1.0, 0.65);
/// Width of each of the crop bar's two lines, as in the viewfinder (`LINE_WIDTH` in preview.wgsl).
pub const CROP_LINE: f32 = 1.0;
const CROP_LINE_ALPHA: f32 = 0.2;
const CROP_BAR_RADIUS: f32 = 10.0;

/// The crop bar's outer, black line.
pub fn crop_bar_outer(_theme: &Theme) -> container::Style {
    container::Style {
        border: Border {
            color: Color::from_rgba(0.0, 0.0, 0.0, CROP_LINE_ALPHA),
            width: CROP_LINE,
            radius: CROP_BAR_RADIUS.into(),
        },
        ..Default::default()
    }
}

/// The crop bar's inner, white line, with a faint dark fill behind the text.
pub fn crop_bar_inner(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Color::from_rgba(0.0, 0.0, 0.0, 0.35).into()),
        border: Border {
            color: Color::from_rgba(1.0, 1.0, 1.0, CROP_LINE_ALPHA),
            width: CROP_LINE,
            radius: (CROP_BAR_RADIUS - CROP_LINE).into(),
        },
        ..Default::default()
    }
}

/// The crop bar's aspect ratio picker: soft white text, nothing else.
pub fn crop_pick_list(_theme: &Theme, status: pick_list::Status) -> pick_list::Style {
    let alpha = match status {
        pick_list::Status::Active => 0.65,
        pick_list::Status::Hovered | pick_list::Status::Opened { .. } => 1.0,
    };
    let color = Color::from_rgba(1.0, 1.0, 1.0, alpha);
    pick_list::Style {
        text_color: color,
        placeholder_color: color,
        handle_color: color,
        background: Background::Color(Color::TRANSPARENT),
        border: Border::default(),
    }
}

/// Icon button in the crop bar: soft white, brighter on hover.
pub fn crop_button(_theme: &Theme, status: button::Status) -> button::Style {
    let alpha = match status {
        button::Status::Active => 0.65,
        button::Status::Hovered | button::Status::Pressed => 1.0,
        button::Status::Disabled => 0.2,
    };
    button::Style {
        background: None,
        text_color: Color::from_rgba(1.0, 1.0, 1.0, alpha),
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
pub const THUMBNAIL_INNER_RADIUS: f32 = 8.0;
const THUMBNAIL_BUTTON_RADIUS: f32 = 12.0;
const ROW_BUTTON_RADIUS: f32 = 6.0;

/// `row_button` with rounder corners, and no border on thumbnails that are not selected.
pub fn thumbnail_button(selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    let style = row_button(selected);
    move |theme, status| {
        let mut style = style(theme, status);
        style.border.radius = THUMBNAIL_BUTTON_RADIUS.into();
        if !selected {
            style.border.color = Color::TRANSPARENT;
        }
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
        background: Some(faded(theme.palette().text, 0.07).into()),
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

/// The value next to an adjustment's name: plain text that shows a field
/// outline on hover to say it can be clicked and typed over.
pub fn value_button(theme: &Theme, status: button::Status) -> button::Style {
    let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
    button::Style {
        background: hovered.then(|| faded(theme.palette().text, 0.08).into()),
        text_color: ADJUSTMENT_GRAY,
        border: border::rounded(4),
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

/// Stands in for a thumbnail that has not been decoded yet.
pub fn thumbnail_loading(theme: &Theme) -> container::Style {
    container::Style {
        background: Some(faded(theme.palette().text, 0.06).into()),
        border: Border {
            radius: THUMBNAIL_INNER_RADIUS.into(),
            ..Border::default()
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

/// Bright gray for slider fills and knobs and the adjustment titles.
pub const ADJUSTMENT_GRAY: Color = Color::from_rgb(0.8, 0.8, 0.8);

/// Outer radius and ring width of the knob on sliders and the colour wheel.
pub const KNOB_RADIUS: f32 = 7.0;
pub const KNOB_RING: f32 = 4.0;

/// The knob's (fill, ring) colours. The fill is the sidebar's colour, to hide
/// the rail behind the ring.
pub fn knob_colors(theme: &Theme, active: bool) -> (Color, Color) {
    let ring = if active {
        lighten(ADJUSTMENT_GRAY, 0.1)
    } else {
        ADJUSTMENT_GRAY
    };
    (lighten(theme.palette().background, 0.05), ring)
}

pub fn adjustment_slider(theme: &Theme, status: slider::Status) -> slider::Style {
    let primary = ADJUSTMENT_GRAY;
    let (fill, handle) = knob_colors(theme, status != slider::Status::Active);
    slider::Style {
        rail: slider::Rail {
            backgrounds: (
                Background::Color(primary),
                Background::Color(faded(theme.palette().text, 0.15)),
            ),
            width: 4.0,
            border: border::rounded(2.0),
        },
        handle: slider::Handle {
            shape: slider::HandleShape::Circle {
                radius: KNOB_RADIUS,
            },
            background: Background::Color(fill),
            border_width: KNOB_RING,
            border_color: handle,
        },
    }
}

/// For a slider laid over [`crate::centered_track`], which draws the rail:
/// only the handle is left.
pub fn centered_slider(theme: &Theme, status: slider::Status) -> slider::Style {
    let mut style = adjustment_slider(theme, status);
    style.rail.backgrounds = (
        Background::Color(Color::TRANSPARENT),
        Background::Color(Color::TRANSPARENT),
    );
    style
}

/// Rail colours for [`crate::centered_track`]: the part filled from the
/// centre to the handle, and the rest.
pub fn centered_rail(theme: &Theme) -> (Color, Color) {
    (ADJUSTMENT_GRAY, faded(theme.palette().text, 0.15))
}

/// Backdrop behind the panes, showing through the gaps as divider lines.
pub fn pane_dividers(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Color::BLACK.into()),
        ..Default::default()
    }
}

/// Highlights the pane divider under the cursor and while it's dragged.
pub fn panes(theme: &Theme) -> pane_grid::Style {
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
        hovered_split: split,
    }
}
