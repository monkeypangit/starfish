// Release builds on Windows open no console window next to the app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
// The Debug tab, the only user of some messages and controls, is left out of release builds.
#![cfg_attr(not(debug_assertions), allow(dead_code))]

mod auto;
mod color_edit;
mod develop;
mod export;
mod histogram;
mod lens;
mod library;
mod local;
mod log;
mod pill;
mod preview;
mod processing;
mod raw;
mod source;
mod styling;
mod tone;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use iced::widget::image::Handle;
use iced::widget::{
    button, canvas, center, checkbox, column, container, mouse_area, pane_grid, pick_list,
    responsive, row, rule, scrollable, shader, space, stack, text, text_input, tooltip,
};
use iced::{
    Alignment, ContentFit, Element, Font, Length, Padding, Size, Subscription, Task, alignment,
    keyboard, mouse, task, window,
};

use demosaic_run::noise::{A_RANGE, B_RANGE};
use export::{Compression, Format};
use histogram::Histogram;
use lens::Corrections;
use pill::{Pill, Track};
use preview::{
    Adjustments, Aspect, Crop, Interaction, Markers, Preview, Rotation, ViewState, Zoom,
};
use processing::{Engine, Image, Method, Noise};
use raw::RawFrame;

const APP_TITLE: &str = "Starfish";
/// Linux application id; Wayland shows the icon of the `.desktop` file with
/// this basename, as it ignores window icons.
#[cfg(target_os = "linux")]
const APP_ID: &str = "starfish";
/// The placeholder logo (`resources/icon/starfish.svg`) as the window icon.
const ICON_BYTES: &[u8] = include_bytes!("../resources/icon/starfish-256.png");

const FONT_BYTES_REGULAR: &[u8] = include_bytes!("../resources/Roboto/Roboto-Regular.ttf");
const FONT_BYTES_BOLD: &[u8] = include_bytes!("../resources/Roboto/Roboto-Bold.ttf");
const ICONS_FONT_BYTES: &[u8] = include_bytes!("../resources/MaterialIcons-Regular.ttf");
/// Monospaced digits for the crop bar (Roboto Mono, SIL OFL 1.1, variable weight).
const MONO_FONT_BYTES: &[u8] = include_bytes!("../resources/RobotoMono/RobotoMono-Variable.ttf");

const FONT: Font = Font::with_name("Roboto");
const BOLD_FONT: Font = Font {
    weight: iced::font::Weight::Bold,
    ..FONT
};
const ICONS_FONT: Font = Font::with_name("Material Icons");
const MONO_FONT: Font = Font::with_name("Roboto Mono");

const SPACING_SMALL: f32 = 5.0;
const SPACING_NORMAL: f32 = 10.0;
const SPACING_LARGE: f32 = 20.0;

const FONT_SIZE_SMALLER: f32 = 12.0;
const FONT_SIZE_SMALL: f32 = 13.0;
const FONT_SIZE_STANDARD: f32 = 14.0;
/// The crop bar's readings.

const ICON_SIZE_TINY: f32 = 14.0;
const RESET_SLOT: f32 = 18.0;
const ICON_SIZE_TOOLBAR: f32 = 22.0;
const ICON_SIZE_PLACEHOLDER: f32 = 64.0;

const WINDOW_WIDTH: f32 = 1400.0;
const LEFT_SIDEBAR_WIDTH: f32 = 220.0;
const RIGHT_SIDEBAR_WIDTH: f32 = 290.0;
/// Histogram height as a fraction of its width.
const HISTOGRAM_RATIO: f32 = 0.40;
const HISTOGRAM_MIN_RATIO: f32 = 0.2;
const HISTOGRAM_MAX_RATIO: f32 = 1.5;
/// Slope of the tone curve at the midpoint (tone::Curve); 1 = no curve.
const CONTRAST_RANGE: std::ops::RangeInclusive<f32> = 1.0..=10.0;
/// Black point slider end, in percent of white (Adjustments::black_point).
const BLACK_POINT_PERCENT: f32 = 5.0;
/// The lens corrections a newly opened file gets, where its profile has them
/// and (see [`lens_default`]) its numbers are confirmed. All of them are
/// drawn: distortion and vignetting from the file's profile, CA as measured
/// in the image (`lens::measure`; the profiles' CA tables are still read,
/// and `LensPass` can still apply them, for reference).
const LENS_DEFAULT: Corrections = Corrections::ALL;
/// Height of the panels' header strips.
const HEADER_HEIGHT: f32 = 47.0;
/// Narrowest a thumbnail cell gets; the grid adds columns as the pane widens.
const THUMBNAIL_MIN_WIDTH: f32 = 160.0;
const THUMBNAIL_GAP: f32 = 0.0;
/// Gap between the button border and the thumbnail.
const THUMBNAIL_PADDING: f32 = 8.0;
/// Diameter of the colour mixer's round swatches.
const SWATCH_SIZE: f32 = 31.0;
/// Narrowest the preview and the adjustments panel go.
/// The mixer's parameters, in the order the sliders have them.
const MIX_PARAMS: [color_edit::MixParam; 3] = [
    color_edit::MixParam::Hue,
    color_edit::MixParam::Saturation,
    color_edit::MixParam::Brightness,
];
/// The grading's parameters, in the order the sliders have them.
const GRADE_PARAMS: [color_edit::GradeParam; 2] =
    [color_edit::GradeParam::Hue, color_edit::GradeParam::Amount];
/// A row of round buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Round {
    Mix,
    Grade,
}

/// `content`, reporting when the pointer is over it, for [`Round`] press timing.
fn round_hover<'a>(
    content: impl Into<Element<'a, Message>>,
    group: Round,
    index: usize,
) -> Element<'a, Message> {
    mouse_area(content)
        .on_enter(Message::RoundHover(Some((group, index))))
        .on_exit(Message::RoundHover(None))
        .into()
}

/// How long a round button is held to select it alone.
const LONG_PRESS: Duration = Duration::from_millis(500);
/// The widest gap between the round buttons of a row of them.
const BUTTON_GAP_MAX: f32 = 12.0;
/// The reset badge on an edited mixer swatch, and its icon.
/// How far the badge sticks out past the swatch's corner, each way.
const BADGE_OVERHANG: f32 = 6.0;
const BADGE_SIZE: f32 = RESET_SLOT;
/// The grey each range's button and slider knob is filled with: dark for the
/// shadows, light for the highlights.
const BAND_GREYS: [f32; color_edit::RANGES] = [0.25, 0.50, 0.75];
const BADGE_ICON_SIZE: f32 = ICON_SIZE_TINY;

const PANE_MIN_WIDTH: f32 = 380.0;
/// Narrowest the library panel goes.
const LIBRARY_MIN_WIDTH: f32 = 200.0;
/// Dragging the library's edge below this width collapses it.
const LIBRARY_COLLAPSE_WIDTH: f32 = 140.0;
/// Dragging the collapsed library's edge out this far from where the drag
/// folded it (or from the window's edge) opens it.
const LIBRARY_EXPAND_WIDTH: f32 = 24.0;
/// A library being pulled out folds again only once the drag has come back
/// this far from the furthest it went.
const LIBRARY_REFOLD_DISTANCE: f32 = 12.0;
/// The strip left of the window when the library is collapsed.
const LIBRARY_STRIP_WIDTH: f32 = 8.0;
/// Width of the names column in the Metadata tab.
const METADATA_LABEL_WIDTH: f32 = 86.0;
/// Width of the field a value is typed into.
const VALUE_INPUT_WIDTH: f32 = 72.0;
/// Extra grabbable width either side of a pane divider.
const PANE_RESIZE_LEEWAY: f32 = 6.0;
/// Slider steps per wheel notch when scrolling over an adjustment with Ctrl held.
const NUDGE_STEPS: f32 = 5.0;
/// Touchpad pixels that count as one wheel notch.
const PIXELS_PER_NOTCH: f32 = 40.0;

/// Id of the field a value is typed into; only one exists at a time.
const VALUE_INPUT_ID: &str = "value-input";

const CODEPOINT_FOLDER_OPEN: char = '\u{e2c8}';
const CODEPOINT_DOWNLOAD: char = '\u{e2c4}';
/// Material's first_page / last_page: a bar with an arrow towards it, the
/// pair that folds the library away and brings it back.
const CODEPOINT_COLLAPSE_LIBRARY: char = '\u{e5dc}';
const CODEPOINT_EXPAND_LIBRARY: char = '\u{e5dd}';
const CODEPOINT_PHOTO: char = '\u{e410}';
const CODEPOINT_HOURGLASS: char = '\u{e88b}';
const CODEPOINT_ERROR: char = '\u{e001}';
const CODEPOINT_ROTATE_LEFT: char = '\u{e419}';
const CODEPOINT_ROTATE_RIGHT: char = '\u{e41a}';
const CODEPOINT_COLORIZE: char = '\u{e3b8}';
const CODEPOINT_RESTART: char = '\u{f053}';
const CODEPOINT_AUTO: char = '\u{e663}';
const CODEPOINT_VISIBILITY: char = '\u{e8f4}';
const CODEPOINT_VISIBILITY_OFF: char = '\u{e8f5}';
const CODEPOINT_EXPAND_MORE: char = '\u{e5cf}';
const CODEPOINT_CHEVRON_RIGHT: char = '\u{e5cc}';

fn main() -> iced::Result {
    log::init();
    iced::application(App::new, App::update, App::view)
        .title(App::title)
        .theme(|_: &App| styling::theme())
        .subscription(App::subscription)
        .font(FONT_BYTES_REGULAR)
        .font(FONT_BYTES_BOLD)
        .font(ICONS_FONT_BYTES)
        .font(MONO_FONT_BYTES)
        .default_font(FONT)
        .window(iced::window::Settings {
            size: Size::new(WINDOW_WIDTH, 860.0),
            icon: window::icon::from_file_data(ICON_BYTES, None).ok(),
            #[cfg(target_os = "linux")]
            platform_specific: window::settings::PlatformSpecific {
                application_id: APP_ID.to_owned(),
                ..Default::default()
            },
            min_size: Some(Size::new(1000.0, 600.0)),
            ..Default::default()
        })
        .run()
}

/// What a histogram is of: the image's id and how it is drawn.
type HistogramKey = (u64, export::View);
type EstimateKey = (u64, export::View, u8);

#[derive(Debug, Clone)]
struct Loaded {
    image: Image,
    /// Kept for re-demosaicing with a different noise level.
    raw: Arc<RawFrame>,
    /// The demosaic stage's own noise estimate, if it denoises.
    estimate: Option<Noise>,
    decode_time: Duration,
    develop_time: Duration,
    /// The quick binned stand-in; the real demosaic is still running.
    preview: bool,
    /// The tone fitted to the camera's embedded JPEG; only the quick preview
    /// has it, and `None` if the file has no JPEG.
    fit: Option<auto::Fit>,
}

#[derive(Debug, Default)]
enum Status {
    #[default]
    Idle,
    Loading(String),
    Error(String),
}

/// What the right-hand panel shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SidebarTab {
    Adjustments,
    /// Crop, rotate and straighten. The preview is in crop mode while it is open.
    Crop,
    /// Hot and dead pixels. Picking is started from here.
    Repair,
    Metadata,
    /// File format, compression and the export itself.
    Export,
    /// The experimental switches, in debug builds only.
    #[cfg(debug_assertions)]
    Debug,
}

impl SidebarTab {
    const ALL: &'static [SidebarTab] = &[
        SidebarTab::Adjustments,
        SidebarTab::Crop,
        SidebarTab::Repair,
        SidebarTab::Metadata,
        SidebarTab::Export,
        #[cfg(debug_assertions)]
        SidebarTab::Debug,
    ];

    fn label(self) -> &'static str {
        match self {
            SidebarTab::Adjustments => "Adjust",
            SidebarTab::Crop => "Crop",
            SidebarTab::Repair => "Repair",
            SidebarTab::Metadata => "Info",
            SidebarTab::Export => "Export",
            #[cfg(debug_assertions)]
            SidebarTab::Debug => "Debug",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PaneKind {
    FileBrowser,
    Preview,
    Tools,
}

enum Thumbnail {
    Loading,
    /// The file has no embedded preview.
    Missing,
    /// The picture and its size in pixels.
    Ready(Handle, (u32, u32)),
}

/// A raw file in the browsed folder.
struct Entry {
    path: PathBuf,
    thumbnail: Thumbnail,
}

/// A collapsible group of controls in the adjustments panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Section {
    Exposure,
    Tone,
    Color,
    ColorMixer,
    ColorGrading,
    Enhancements,
    LensCorrections,
}

impl Section {
    fn label(self) -> &'static str {
        match self {
            Section::Exposure => "Exposure",
            Section::Tone => "Tone",
            Section::Enhancements => "Detail",
            Section::LensCorrections => "Lens corrections",
            Section::Color => "Color",
            Section::ColorMixer => "Color mixer",
            Section::ColorGrading => "Color grading",
        }
    }
}

struct App {
    panes: pane_grid::State<PaneKind>,
    /// The sidebars' widths and the window's, in logical pixels. The panes'
    /// split ratios are derived from them (`apply_pane_widths`), so a window
    /// resize changes only the preview, and dragging one divider only the
    /// pane next to it.
    left_width: f32,
    /// Whether the library pane is shown (not persisted).
    library_visible: bool,
    /// While the library is being pulled out (by a press on its strip or by
    /// dragging the collapsed edge): the furthest the drag has gone. It
    /// folds again only when the drag turns back towards the edge.
    library_opening: Option<f32>,
    /// While a drag has the library folded: the furthest in it has gone.
    library_trough: Option<f32>,
    right_width: f32,
    /// The widths last written to the settings.
    saved_widths: (f32, f32),
    /// The left mouse button is down (from a press anywhere in the window).
    mouse_down: bool,
    /// The round button the pointer is over, if any.
    round_hover: Option<(Round, usize)>,
    /// Counts presses on round buttons, so a long press timer that is out
    /// of date can tell.
    round_press: u64,
    /// A long press just selected a button alone: the click that ends it
    /// must not toggle it.
    long_pressed: bool,
    window_width: f32,
    /// The folder shown in the file browser.
    folder: Option<PathBuf>,
    entries: Vec<Entry>,
    /// Counts opened folders, so thumbnails of an earlier one are dropped.
    folder_generation: u64,
    /// Stops the thumbnail loading of the previous folder when replaced.
    thumbnail_task: Option<task::Handle>,
    /// The file that is open (or loading).
    current: Option<PathBuf>,
    loaded: Option<Loaded>,
    status: Status,
    view: ViewState,
    /// The view and the shown image's size when another file was chosen,
    /// carried over to it once its size and orientation are known.
    carried_view: Option<(ViewState, Size)>,
    rotation: Rotation,
    /// The export frame against the image.
    crop: Crop,
    /// Crop mode: the image moves, zooms and tilts under a fixed frame.
    cropping: bool,
    /// Picking defective pixels: the preview shows the raw mosaic
    /// (`pick_view`, once made) and a click marks or unmarks a site.
    picking: bool,
    /// Preview overlay while the exposure (clipped highlights drawn black) or
    /// black point (clipped shadows white) slider is being dragged.
    clip_highlights: bool,
    clip_shadows: bool,
    /// The midpoint slider is being dragged: the preview marks the crossover.
    midpoint_dragging: bool,
    /// The rule-of-thirds grid over the frame (Crop tab; not persisted).
    grid: bool,
    /// The crop frame is kept inside the image (the Crop tab's toggle).
    crop_fit: bool,
    /// The tone the auto icons go back to: what was fitted to the open
    /// file's embedded JPEG (the defaults if it had none).
    auto_tone: Option<auto::Fit>,
    /// The eye icons: the overlays kept on beyond the drags (highlights,
    /// shadows, crossover; not persisted).
    overlays: [bool; 3],
    pick_view: Option<Image>,
    /// The marked sites `(y, x)` of the open file, kept per camera and size.
    picks: Vec<(usize, usize)>,
    preview_size: Size,
    adjustments: Adjustments,
    /// The colour the mixer's sliders show (the one last clicked).
    mix_color: usize,
    /// The colours the mixer's sliders edit (a click toggles one, a long press selects one alone); with
    /// none, the sliders are disabled.
    mix_selected: [bool; color_edit::MIX_COLORS],
    /// The range the grading's sliders are for.
    grade_range: usize,
    /// The ranges the grading's sliders edit, as `mix_selected` is for the mixer.
    grade_selected: [bool; color_edit::RANGES],
    /// Of what the export of the loaded image would hold.
    histogram: Option<Histogram>,
    /// What the histogram run in flight, or else the last one, was for.
    /// After every update a run starts if this is not what is shown now.
    histogram_for: Option<HistogramKey>,
    /// The JPEG size estimate for (image, view, quality), once made.
    jpeg_estimate: Option<(EstimateKey, u64)>,
    /// The estimate being made.
    jpeg_estimating: Option<EstimateKey>,
    /// A histogram run is in flight; when it finishes another starts if the
    /// image or view has moved on, so a drag costs one run at a time with
    /// the latest adjustments instead of one per mouse move.
    histogram_running: bool,
    /// Dragged by the histogram's lower border.
    histogram_ratio: f32,
    /// The ratio last written to the settings.
    saved_histogram_ratio: f32,
    /// The lens corrections to apply, where the raw file's profile has them.
    lens: Corrections,
    /// Debugging: the demosaic stage chosen in the picker. `None` until one
    /// is: the engine then picks (`Engine::automatic`: the light network on
    /// an integrated GPU, else the full one).
    method: Option<Method>,
    /// The noise the network is told where it differs from the image's
    /// estimate (the ISO slider, and a and b in Debug). Cleared when a file
    /// opens.
    noise_set: NoiseSet,
    /// The re-demosaic in flight, if any. A change that makes its settings
    /// out of date cancels it; when it ends it starts another if they have
    /// moved on.
    redeveloping: Option<Redevelop>,
    /// Counts opened files, so a re-demosaic of an earlier file is dropped.
    generation: u64,
    /// Set when another file is opened, to stop the earlier file's demosaic.
    cancel: Arc<AtomicBool>,
    tab: SidebarTab,
    /// Sections of the adjustments panel folded away.
    collapsed: HashSet<Section>,
    /// The adjustment whose value is being typed over, and the text so far.
    editing: Option<(Control, String)>,
    /// Ctrl (Cmd) is held: scrolling over an adjustment nudges it.
    ctrl: bool,
    /// A file is being dragged over the window.
    drop_hover: bool,
    export_format: Format,
    export_compression: Compression,
    /// JPEG quality, 1..=100.
    export_quality: u8,
    /// An export is being written.
    exporting: bool,
}

#[derive(Debug, Clone)]
enum Message {
    OpenFile,
    FileChosen(Option<PathBuf>),
    OpenFolder,
    FolderChosen(Option<PathBuf>),
    Thumbnail(u64, usize, Option<(Handle, (u32, u32))>),
    Previewed(u64, Result<Loaded, String>),
    Developed(u64, Result<Loaded, String>),
    GrainAmount(f32),
    /// log10 of the noise model's a (0) or b (1) the network is told.
    Noise(usize, f32),
    /// The ISO slider: stops from the file's ISO (`NoiseSet::stops`).
    NoiseIso(f32),
    LensCorrection(lens::Kind, bool),
    Method(Method),
    FilterGrain(bool),
    CaFix(bool),
    Highlights(bool),
    HotPixels(bool),
    Grid(bool),
    /// The eyedropper: start or finish picking defective pixels.
    TogglePick,
    /// The clipping overlays in the toolbar.
    ExposureDrag(f32),
    BlackPointDrag(f32),
    ClipReleased,
    /// The auto icon: back to the tone fitted to the embedded JPEG.
    Auto(Control),
    /// An eye icon: keep overlay `.0` (`Overlay`) on or off.
    ToggleOverlay(Overlay),
    /// The raw mosaic to pick in is ready (for this generation's file).
    PickView(u64, Image),
    Redeveloped(u64, Result<Redeveloped, String>),
    HistogramReady(HistogramKey, Result<Option<Histogram>, String>),
    EstimateReady(EstimateKey, Result<Option<u64>, String>),
    Preview(Interaction),
    ZoomFit,
    ToggleLibrary,
    RotateLeft,
    RotateRight,
    CropReset,
    CropFit,
    /// The histogram's crossover dot dragged to this x (sRGB-encoded, 0..1).
    HistogramMidpoint(f32),
    /// The histogram's contrast handle dragged to this y (sRGB-encoded, 0..1).
    HistogramContrast(f32),
    /// The straighten slider, in degrees.
    CropAngle(f32),
    CropAspect(Aspect),
    /// Percent of white (Adjustments::black_point × 100).
    BlackPoint(f32),
    Exposure(f32),
    Contrast(f32),
    LocalContrast(f32),
    Sharpness(f32),
    AddedGrain(f32),
    ToeKnee(f32),
    ShoulderKnee(f32),
    /// Slider position, 0..1 (tone::midpoint_position).
    Midpoint(f32),
    /// The midpoint slider moved by the mouse; released ends the overlay.
    MidpointDrag(f32),
    MidpointReleased,
    Saturation(f32),
    Vibrance(f32),
    Temperature(f32),
    Tint(f32),
    ResetAdjustments,
    /// Every auto tuned value back to what the auto tone gave.
    AutoAll,
    Tab(SidebarTab),
    ResetControl(Control),
    /// Puts every control of a section back to its default.
    ResetSection(Section),
    /// Toggles a colour in the mixer's selection; a long press selects it alone.
    MixColor(usize),
    /// The pointer is over a round button (or left it).
    RoundHover(Option<(Round, usize)>),
    /// A round button was held for `LONG_PRESS`: press number, button.
    RoundLongPress(u64, Round, usize),
    /// Colours' values for one parameter, as `(colour, value)`, from dragging
    /// their dots on its slider.
    MixMove(color_edit::MixParam, Vec<(usize, f32)>),
    /// Puts one colour's hue, saturation and brightness back.
    MixReset(usize),
    /// A colour's hue, saturation or brightness.
    Mix(usize, color_edit::MixParam, f32),
    /// Selects the range the grading's sliders are for.
    GradeRange(usize),
    /// Ranges' values for one parameter, as `(range, value)`, from dragging
    /// their dots on its slider.
    GradeMove(color_edit::GradeParam, Vec<(usize, f32)>),
    /// Puts one range's target hue and hue shift back.
    GradeReset(usize),
    /// A range's target hue (degrees) or amount.
    Grade(usize, color_edit::GradeParam, f32),
    HistogramRatio(f32),
    PaneResized(pane_grid::ResizeEvent),
    /// The mouse button went up: a divider drag may have ended.
    MousePressed,
    MouseReleased,
    WindowResized(Size),
    ToggleSection(Section),
    /// Starts typing over a control's value.
    EditValue(Control),
    EditInput(String),
    EditCommit,
    EditCancel,
    /// Scrolls a control by wheel notches (positive = up).
    Nudge(Control, f32),
    Ctrl(bool),
    DropHover(bool),
    Dropped(PathBuf),
    ExportFormat(Format),
    ExportCompression(Compression),
    ExportQuality(f32),
    Export,
    ExportPath(Option<PathBuf>),
    Exported(Result<PathBuf, String>),
}

/// One adjustment control, for its own reset button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Control {
    Grain,
    /// The noise model's a (0) or b (1).
    Noise(usize),
    /// The noise as an ISO, relative to the file's (`NoiseSet::stops`).
    Iso,
    Temperature,
    Tint,
    BlackPoint,
    Exposure,
    Saturation,
    Vibrance,
    Contrast,
    Midpoint,
    LocalContrast,
    Sharpness,
    AddedGrain,
    ToeKnee,
    ShoulderKnee,
    /// One of the colour mixer's numbers, for colour `.0`.
    Mix(usize, color_edit::MixParam),
    /// One of the grading's, for range `.0`.
    Grade(usize, color_edit::GradeParam),
}

impl Control {
    /// Whether the slider is switched off: a mixer or grading slider with
    /// nothing selected.
    fn disabled(self, app: &App) -> bool {
        match self {
            Control::Mix(..) => !app.mix_selected.iter().any(|&on| on),
            Control::Grade(..) => !app.grade_selected.iter().any(|&on| on),
            _ => false,
        }
    }

    /// Whether `self` differs from its default.
    fn changed(self, app: &App) -> bool {
        let (a, d) = (&app.adjustments, Adjustments::default());
        match self {
            Control::Grain => a.grain != d.grain,
            Control::Temperature => a.temperature != d.temperature,
            Control::Tint => a.tint != d.tint,
            Control::BlackPoint => a.black_point != d.black_point,
            Control::Exposure => a.exposure != d.exposure,
            Control::Saturation => a.saturation != d.saturation,
            Control::Vibrance => a.vibrance != d.vibrance,
            Control::Contrast => a.contrast != d.contrast,
            Control::Midpoint => a.midpoint != d.midpoint,
            Control::LocalContrast => a.local_contrast != d.local_contrast,
            Control::Sharpness => a.sharpness != d.sharpness,
            Control::AddedGrain => a.added_grain != d.added_grain,
            Control::ToeKnee => a.toe_power != d.toe_power,
            Control::ShoulderKnee => a.shoulder_power != d.shoulder_power,
            Control::Noise(i) => app.noise_set.ab[i].is_some(),
            Control::Iso => app.noise_set.stops.is_some(),
            // Any selected colour edited: the reset covers all of them.
            Control::Mix(_, p) => (0..color_edit::MIX_COLORS)
                .any(|i| app.mix_selected[i] && a.color_mix.get(i, p) != 0.0),
            Control::Grade(_, p) => (0..color_edit::RANGES).any(|i| {
                app.grade_selected[i] && a.grading.get(i, p) != d.grading.get(i, p)
            }),
        }
    }

    /// The slider behind `self`.
    fn slider(self, app: &App) -> Option<SliderSpec> {
        let a = &app.adjustments;
        let spec = |x, range, message: fn(f32) -> Message| SliderSpec {
            x,
            range,
            step: 0.01,
            message: Box::new(message),
        };
        Some(match self {
            Control::Grain => spec(a.grain, 0.0..=1.0, Message::GrainAmount),
            Control::Temperature => spec(a.temperature, -1.0..=1.0, Message::Temperature),
            Control::Tint => spec(a.tint, -1.0..=1.0, Message::Tint),
            Control::BlackPoint => spec(
                a.black_point * 100.0,
                -BLACK_POINT_PERCENT..=BLACK_POINT_PERCENT,
                Message::BlackPoint,
            ),
            Control::Exposure => spec(a.exposure, -5.0..=5.0, Message::Exposure),
            Control::Saturation => spec(a.saturation, 0.0..=2.0, Message::Saturation),
            Control::Vibrance => spec(a.vibrance, 0.0..=2.0, Message::Vibrance),
            Control::Contrast => spec(a.contrast, CONTRAST_RANGE, Message::Contrast),
            Control::LocalContrast => spec(a.local_contrast, 0.0..=1.0, Message::LocalContrast),
            Control::Sharpness => spec(a.sharpness, 0.0..=1.0, Message::Sharpness),
            Control::AddedGrain => spec(a.added_grain, 0.0..=1.0, Message::AddedGrain),
            Control::ToeKnee => spec(a.toe_power, tone::POWER_RANGE, Message::ToeKnee),
            Control::ShoulderKnee => {
                spec(a.shoulder_power, tone::POWER_RANGE, Message::ShoulderKnee)
            }
            Control::Midpoint => SliderSpec {
                step: 0.005,
                ..spec(
                    tone::midpoint_position(a.midpoint),
                    0.0..=1.0,
                    Message::Midpoint,
                )
            },
            Control::Noise(i) => {
                let message: fn(f32) -> Message =
                    [|x| Message::Noise(0, x), |x| Message::Noise(1, x)][i];
                spec(app.noise_log10(i)?, noise_range(i), message)
            }
            Control::Iso => {
                app.loaded.as_ref()?.estimate?;
                SliderSpec {
                    x: app.noise_set.stops.unwrap_or(0.0),
                    range: ISO_STOPS,
                    step: 1.0 / 3.0,
                    message: Box::new(Message::NoiseIso),
                }
            }
            Control::Mix(i, p) => SliderSpec {
                x: a.color_mix.get(i, p),
                range: -1.0..=1.0,
                step: 0.01,
                message: Box::new(move |x| Message::Mix(i, p, x)),
            },
            Control::Grade(i, p) => SliderSpec {
                x: a.grading.get(i, p),
                range: match p {
                    color_edit::GradeParam::Hue => 0.0..=360.0,
                    color_edit::GradeParam::Amount => 0.0..=1.0,
                },
                step: if p == color_edit::GradeParam::Hue {
                    1.0
                } else {
                    0.01
                },
                message: Box::new(move |x| Message::Grade(i, p, x)),
            },
        })
    }

    /// The number the control's value label shows, without its unit; `None`
    /// if there is nothing to edit (no image, or no noise estimate).
    fn shown(self, app: &App) -> Option<f32> {
        let x = self.slider(app)?.x;
        app.loaded.as_ref()?;
        if matches!(self, Control::Mix(i, _) if !app.mix_selected[i])
            || matches!(self, Control::Grade(i, _) if !app.grade_selected[i])
        {
            return None;
        }
        Some(match self {
            Control::Grain if !app.denoised() => return None,
            Control::Grain
            | Control::Tint
            | Control::Saturation
            | Control::Vibrance
            | Control::LocalContrast
            | Control::Sharpness
            | Control::AddedGrain
            | Control::Mix(
                _,
                color_edit::MixParam::Saturation | color_edit::MixParam::Brightness,
            )
            | Control::Grade(_, color_edit::GradeParam::Amount) => x * 100.0,
            Control::Mix(_, color_edit::MixParam::Hue) => x * color_edit::HUE_SHIFT_MAX,
            Control::Temperature => match app.white() {
                Some(white) => white.kelvin(x),
                None => x * 100.0,
            },
            Control::Midpoint => app.adjustments.midpoint * 100.0,
            Control::Noise(i) => noise_value(i, x) as f32,
            // without the file's ISO there is nothing to show it relative to
            Control::Iso => app.loaded.as_ref()?.raw.iso? * x.exp2(),
            _ => x,
        })
    }

    /// The slider position that makes the label show `shown`.
    fn position_for(self, app: &App, shown: f32) -> f32 {
        match self {
            Control::Grain
            | Control::Tint
            | Control::Saturation
            | Control::Vibrance
            | Control::LocalContrast
            | Control::Sharpness
            | Control::AddedGrain
            | Control::Mix(
                _,
                color_edit::MixParam::Saturation | color_edit::MixParam::Brightness,
            )
            | Control::Grade(_, color_edit::GradeParam::Amount) => shown / 100.0,
            Control::Mix(_, color_edit::MixParam::Hue) => shown / color_edit::HUE_SHIFT_MAX,
            Control::Temperature => match app.white() {
                Some(white) => white.temperature_for(shown.max(1.0)),
                None => shown / 100.0,
            },
            Control::Midpoint => tone::midpoint_position(shown / 100.0),
            // 0 (or less) is the slider's left end
            Control::Noise(i) if shown <= 0.0 => *noise_range(i).start(),
            Control::Noise(_) => shown.log10(),
            Control::Iso => match app.loaded.as_ref().and_then(|l| l.raw.iso) {
                Some(iso) if shown > 0.0 => (shown / iso).log2(),
                _ => f32::NAN,
            },
            _ => shown,
        }
    }

    /// `shown` as the value label writes it, and typing starts from.
    fn format(self, shown: f32) -> String {
        match self {
            Control::Noise(_) if shown == 0.0 => String::from("0"),
            Control::Noise(_) => format!("{shown:.2e}"),
            Control::Iso => format!("{:.0}", nominal_iso(shown)),
            _ => format!("{shown:.*}", self.decimals()),
        }
    }

    /// Decimals the label shows.
    fn decimals(self) -> usize {
        match self {
            Control::Midpoint => 2,
            Control::BlackPoint
            | Control::Exposure
            | Control::Contrast
            | Control::ToeKnee
            | Control::ShoulderKnee => 2,
            _ => 0,
        }
    }

    /// The message setting the control to what was typed, clamped to its range.
    fn typed(self, app: &App, input: &str) -> Option<Message> {
        let SliderSpec { range, message, .. } = self.slider(app)?;
        let number = input
            .trim()
            .trim_end_matches(|c: char| c.is_alphabetic() || c == '%' || c.is_whitespace());
        let shown: f32 = number.parse().ok().filter(|v: &f32| v.is_finite())?;
        let x = self.position_for(app, shown);
        x.is_finite()
            .then(|| message(x.clamp(*range.start(), *range.end())))
    }

    /// The message moving the control by `notches` wheel notches.
    fn nudged(self, app: &App, notches: f32) -> Option<Message> {
        self.shown(app)?;
        let SliderSpec {
            x,
            range,
            step,
            message,
        } = self.slider(app)?;
        // Back onto the slider's grid, so a return to the default is exact.
        let x = ((x + notches * NUDGE_STEPS * step) / step).round() * step;
        Some(message(x.clamp(*range.start(), *range.end())))
    }
}

/// The preview overlays the eye icons keep on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Overlay {
    /// Clipped highlights (exposure).
    Highlights,
    /// Clipped shadows (black point).
    Shadows,
    /// The tone curve's crossover (midpoint).
    Crossover,
}

/// What a numeric control's slider is: position, range, step and message.
struct SliderSpec {
    x: f32,
    range: std::ops::RangeInclusive<f32>,
    step: f32,
    message: Box<dyn Fn(f32) -> Message>,
}

impl App {
    fn new() -> (Self, Task<Message>) {
        let mut app = Self {
            panes: initial_panes(),
            left_width: LEFT_SIDEBAR_WIDTH,
            library_visible: true,
            library_opening: None,
            library_trough: None,
            right_width: RIGHT_SIDEBAR_WIDTH,
            saved_widths: (LEFT_SIDEBAR_WIDTH, RIGHT_SIDEBAR_WIDTH),
            mouse_down: false,
            round_hover: None,
            round_press: 0,
            long_pressed: false,
            window_width: WINDOW_WIDTH,
            folder: None,
            entries: Vec::new(),
            folder_generation: 0,
            thumbnail_task: None,
            current: None,
            loaded: None,
            status: Status::default(),
            export_format: Format::Jpeg,
            export_compression: Compression::Low,
            export_quality: 95,
            exporting: false,
            view: ViewState::default(),
            carried_view: None,
            rotation: Rotation::default(),
            crop: Crop::default(),
            cropping: false,
            picking: false,
            clip_highlights: false,
            clip_shadows: false,
            midpoint_dragging: false,
            grid: false,
            crop_fit: true,
            auto_tone: None,
            overlays: [false; 3],
            pick_view: None,
            picks: Vec::new(),
            preview_size: Size::ZERO,
            adjustments: Adjustments::default(),
            mix_color: 0,
            mix_selected: [false; color_edit::MIX_COLORS],
            grade_range: 0,
            grade_selected: [false; color_edit::RANGES],
            histogram: None,
            histogram_for: None,
            jpeg_estimate: None,
            jpeg_estimating: None,
            histogram_running: false,
            histogram_ratio: HISTOGRAM_RATIO,
            saved_histogram_ratio: HISTOGRAM_RATIO,
            lens: LENS_DEFAULT,
            method: None,
            noise_set: NoiseSet::default(),
            redeveloping: None,
            generation: 0,
            cancel: Arc::new(AtomicBool::new(false)),
            tab: SidebarTab::Adjustments,
            collapsed: HashSet::new(),
            editing: None,
            ctrl: false,
            drop_hover: false,
        };
        if let Some((left, right)) = library::remembered_pane_widths() {
            app.left_width = left;
            app.right_width = right;
            app.saved_widths = (left, right);
            app.apply_pane_widths();
        }
        if let Some(ratio) = library::remembered_histogram_ratio() {
            app.histogram_ratio = ratio.clamp(HISTOGRAM_MIN_RATIO, HISTOGRAM_MAX_RATIO);
            app.saved_histogram_ratio = app.histogram_ratio;
        }
        // `starfish <file>` opens a file straight away (and its folder);
        // otherwise the photo (or else the folder) from the last run comes back.
        let task = match std::env::args_os().nth(1) {
            Some(path) => {
                // Absolute, so the file's parent is a real folder, not "".
                let path = PathBuf::from(path);
                let path = std::fs::canonicalize(&path).unwrap_or(path);
                Task::done(Message::FileChosen(Some(path)))
            }
            None => match (library::remembered_photo(), library::remembered_folder()) {
                (Some(photo), _) => Task::done(Message::FileChosen(Some(photo))),
                (None, Some(folder)) => app.open_folder(folder),
                (None, None) => Task::none(),
            },
        };
        let size = window::oldest()
            .and_then(window::size)
            .map(Message::WindowResized);
        (app, Task::batch([task, size]))
    }

    /// The library's width: its own, or the strip a collapsed one leaves.
    fn library_width(&self) -> f32 {
        if self.library_visible {
            self.left_width
        } else {
            LIBRARY_STRIP_WIDTH
        }
    }

    /// Sets the split ratios so the sidebars have `left_width` and
    /// `right_width` in a window `window_width` wide.
    fn apply_pane_widths(&mut self) {
        let width = self.window_width.max(1.0);
        let min = PANE_MIN_WIDTH;
        // what is left for the preview, if the sidebars give way first
        let sidebars = (width - min).max(0.0);
        let left = if self.library_visible {
            self.left_width
                .clamp(LIBRARY_MIN_WIDTH.min(sidebars), sidebars)
        } else {
            LIBRARY_STRIP_WIDTH
        };
        let right = self
            .right_width
            .clamp(min.min(sidebars - left), sidebars - left);
        let pane_grid::Node::Split {
            id: outer, a: _, b, ..
        } = self.panes.layout().clone()
        else {
            return;
        };
        let pane_grid::Node::Split { id: inner, .. } = *b else {
            return;
        };
        let rest = (width - left).max(1.0);
        self.panes.resize(outer, left / width);
        self.panes
            .resize(inner, ((rest - right) / rest).clamp(0.0, 1.0));
    }

    fn title(&self) -> String {
        match &self.loaded {
            Some(loaded) => format!("{} — {APP_TITLE}", loaded.image.name),
            None => APP_TITLE.to_owned(),
        }
    }

    fn subscription(&self) -> Subscription<Message> {
        use iced::Event;
        use keyboard::key::{Key, Named};

        Subscription::batch([
            window::resize_events().map(|(_, size)| Message::WindowResized(size)),
            iced::event::listen_with(|event, _status, _window| match event {
                Event::Window(window::Event::FileHovered(_)) => Some(Message::DropHover(true)),
                Event::Window(window::Event::FilesHoveredLeft) => Some(Message::DropHover(false)),
                Event::Window(window::Event::FileDropped(path)) => Some(Message::Dropped(path)),
                Event::Mouse(iced::mouse::Event::ButtonPressed(iced::mouse::Button::Left)) => {
                    Some(Message::MousePressed)
                }
                Event::Mouse(iced::mouse::Event::ButtonReleased(iced::mouse::Button::Left)) => {
                    Some(Message::MouseReleased)
                }
                Event::Keyboard(keyboard::Event::ModifiersChanged(modifiers)) => {
                    Some(Message::Ctrl(modifiers.command()))
                }
                Event::Keyboard(keyboard::Event::KeyPressed {
                    key: Key::Named(Named::Escape),
                    ..
                }) => Some(Message::EditCancel),
                _ => None,
            }),
        ])
    }

    /// The as-shot white balance of the open file.
    fn white(&self) -> Option<raw::WhiteBalance> {
        self.loaded.as_ref().and_then(|l| l.raw.white)
    }

    /// Shows `error` over the preview and writes it to the log.
    fn fail(&mut self, error: String) {
        log::write(&format!("Error: {error}"));
        self.status = Status::Error(error);
    }

    fn is_loading(&self) -> bool {
        matches!(self.status, Status::Loading(_))
    }

    /// While the fit toggle is on, pulls the crop frame back inside the image.
    fn constrain_crop(&mut self) {
        if self.crop_fit {
            if let Some(size) = self.rotated_size() {
                self.crop = self.crop.within(size);
            }
        }
    }

    /// The image's size after rotation, before cropping.
    fn rotated_size(&self) -> Option<Size> {
        self.loaded
            .as_ref()
            .map(|l| {
                let size = Size::new(l.image.width as f32, l.image.height as f32);
                // picking shows the raw frame as it is
                if self.picking {
                    size
                } else {
                    self.rotation.size(size)
                }
            })
            .filter(|_| self.preview_size.width > 0.0 && self.preview_size.height > 0.0)
    }

    /// Picking shows the raw frame uncropped and unrotated. The view is in
    /// the frame's space, which is the raw frame's own only without a turn or
    /// crop; then it is kept, so entering and leaving the mode does not move
    /// the picture, else it is reset.
    fn reset_view_for_pick(&mut self) {
        if self.rotation != Rotation::default() || self.crop != Crop::default() {
            self.view = ViewState::default();
        }
    }

    /// The shown image's size: what is inside the crop frame, after rotation.
    fn image_size(&self) -> Option<Size> {
        self.rotated_size().map(|s| {
            if self.picking {
                s
            } else {
                self.crop.frame_size(s)
            }
        })
    }

    /// Logical pixels per image pixel in crop mode.
    fn crop_scale(&self) -> Option<f32> {
        self.rotated_size()
            .map(|s| self.crop.scale(self.preview_size, s))
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        let task = self.handle(message);
        Task::batch([task, self.refresh_histogram(), self.refresh_estimate()])
    }

    fn handle(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::OpenFile => {
                return Task::perform(
                    rfd::AsyncFileDialog::new()
                        .set_title("Open raw file")
                        .add_filter("Raw images", raw::RAW_EXTENSIONS)
                        .add_filter("All files", &["*"])
                        .pick_file(),
                    |handle| Message::FileChosen(handle.map(|h| h.path().to_path_buf())),
                );
            }
            Message::FileChosen(None) => {}
            Message::OpenFolder => {
                let mut dialog = rfd::AsyncFileDialog::new().set_title("Open folder");
                if let Some(folder) = &self.folder {
                    dialog = dialog.set_directory(folder);
                }
                return Task::perform(dialog.pick_folder(), |handle| {
                    Message::FolderChosen(handle.map(|h| h.path().to_path_buf()))
                });
            }
            Message::FolderChosen(None) => {}
            Message::FolderChosen(Some(folder)) => return self.open_folder(folder),
            Message::Thumbnail(generation, index, thumbnail) => {
                if generation == self.folder_generation {
                    if let Some(entry) = self.entries.get_mut(index) {
                        entry.thumbnail = match thumbnail {
                            Some((handle, size)) => Thumbnail::Ready(handle, size),
                            None => Thumbnail::Missing,
                        };
                    }
                }
            }
            Message::FileChosen(Some(path)) => {
                log::write(&format!("Opening {}", path.display()));
                library::remember_photo(&path);
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                self.status = Status::Loading(name);
                self.generation += 1;
                self.cancel.store(true, Ordering::Relaxed);
                self.cancel = Arc::new(AtomicBool::new(false));
                if let Some(run) = &self.redeveloping {
                    run.cancel.store(true, Ordering::Relaxed);
                }
                // The zoom and pan carry over to the next image.
                self.carried_view = self.image_size().map(|size| (self.view, size));
                // Drop the old image right away.
                self.loaded = None;
                self.histogram = None;
                self.histogram_for = None;
                self.rotation = Rotation::default();
                self.crop = Crop::default();
                self.cropping = self.tab == SidebarTab::Crop;
                self.picking = false;
                self.pick_view = None;
                self.picks.clear();
                processing::set_picked(Vec::new());
                self.noise_set = NoiseSet::default();
                self.lens = LENS_DEFAULT;
                self.editing = None;
                // A file from outside the shown folder switches to its folder.
                let folder = match path.parent() {
                    Some(parent) if self.folder.as_deref() != Some(parent) => {
                        Some(self.open_folder(parent.to_path_buf()))
                    }
                    _ => None,
                };
                self.current = Some(path.clone());
                let generation = self.generation;
                let develop = Task::perform(preview(path), move |result| {
                    Message::Previewed(generation, result)
                });
                return match folder {
                    Some(folder) => Task::batch([folder, develop]),
                    None => develop,
                };
            }
            Message::Previewed(generation, _) | Message::Developed(generation, _)
                if generation != self.generation => {}
            Message::Previewed(generation, Ok(loaded)) => {
                let raw = loaded.raw.clone();
                // the sites marked before for this camera, replaced from the start
                self.picks = library::defects(&raw.body(), raw.width, raw.height);
                processing::set_picked(self.picks.clone());
                self.lens = lens_default(loaded.image.profile.as_deref());
                // A portrait shot opens upright; `FileChosen` reset the view and crop.
                self.rotation = Rotation::from_exif(raw.orientation);
                self.view = ViewState::default();
                self.auto_tone = loaded.fit;
                match loaded.fit {
                    Some(fit) => fit.apply(&mut self.adjustments),
                    None => auto::reset(&mut self.adjustments),
                }
                let decode_time = loaded.decode_time;
                self.loaded = Some(loaded);
                if let (Some((view, old)), Some(new)) =
                    (self.carried_view.take(), self.image_size())
                {
                    // Same zoom; the same relative spot of the image.
                    if view.zoom != Zoom::Fit {
                        self.view = ViewState {
                            zoom: view.zoom,
                            center: iced::Point::new(
                                view.center.x / old.width * new.width,
                                view.center.y / old.height * new.height,
                            ),
                        };
                    }
                }
                return Task::perform(
                    develop(raw, decode_time, self.method, self.cancel.clone()),
                    move |result| Message::Developed(generation, result),
                );
            }
            Message::Developed(_, Ok(loaded)) => {
                self.loaded = Some(loaded);
                self.status = Status::Idle;
                // The slider may have moved while the preview was shown.
                return self.redevelop();
            }
            Message::Previewed(_, Err(error)) | Message::Developed(_, Err(error)) => {
                self.fail(error);
            }
            Message::GrainAmount(value) => {
                // drawn: the network's output holds the noise it removed
                self.adjustments.grain = value;
            }
            Message::Noise(i, x) => {
                self.noise_set.ab[i] = Some(x);
                return self.redevelop();
            }
            Message::NoiseIso(stops) => {
                // the ISO sets both; a or b set in Debug start again from it
                self.noise_set = NoiseSet {
                    stops: Some(stops),
                    ab: [None; 2],
                };
                return self.redevelop();
            }
            Message::LensCorrection(kind, on) => {
                // drawn, all of them
                self.lens.set(kind, on);
            }
            Message::Method(method) => {
                self.method = Some(method);
                return self.redevelop();
            }
            Message::FilterGrain(on) => {
                processing::set_filter_grain(on);
                return self.redevelop();
            }
            Message::CaFix(on) => {
                processing::set_ca_fix(on);
                return self.redevelop();
            }
            Message::Highlights(on) => {
                processing::set_highlights(on);
                return self.redevelop();
            }
            Message::Grid(on) => self.grid = on,
            Message::HotPixels(on) => {
                processing::set_hot_pixels(on);
                return self.redevelop();
            }
            Message::Redeveloped(generation, result) => {
                self.redeveloping = None;
                if generation != self.generation {
                    return Task::none();
                }
                match (result, &mut self.loaded) {
                    (Ok(redeveloped), Some(loaded)) => {
                        loaded.image = redeveloped.image;
                        loaded.develop_time = redeveloped.develop_time;
                        loaded.estimate = redeveloped.estimate;
                        return self.redevelop();
                    }
                    // Stopped because the settings moved on: catch up.
                    (Err(error), _) if error == processing::CANCELLED => {
                        return self.redevelop();
                    }
                    (Err(error), _) => self.fail(error),
                    (Ok(_), None) => {}
                }
            }
            Message::Preview(Interaction::Resized(size)) => {
                self.preview_size = size;
            }
            Message::Preview(Interaction::Zoom { factor, anchor }) if self.cropping => {
                if let Some(scale) = self.crop_scale() {
                    self.crop = self.crop.zoom_by(factor, anchor, scale);
                    self.constrain_crop();
                }
            }
            Message::Preview(Interaction::Pan(delta)) if self.cropping => {
                if let Some(scale) = self.crop_scale() {
                    self.crop = self.crop.pan_by(delta, scale);
                    self.constrain_crop();
                }
            }
            Message::Preview(Interaction::Rotate(delta)) => {
                self.crop = self.crop.tilt_by(delta);
                self.constrain_crop();
            }
            Message::ExposureDrag(value) => {
                self.adjustments.exposure = value;
                self.clip_highlights = true;
            }
            Message::BlackPointDrag(percent) => {
                self.adjustments.black_point = percent / 100.0;
                self.clip_shadows = true;
            }
            Message::Auto(control) => {
                if let Some(value) = self.auto_value(control) {
                    match control {
                        Control::Exposure => self.adjustments.exposure = value,
                        Control::Contrast => self.adjustments.contrast = value,
                        Control::Midpoint => self.adjustments.midpoint = value,
                        _ => {}
                    }
                }
            }
            Message::ToggleOverlay(overlay) => {
                let on = &mut self.overlays[overlay as usize];
                *on = !*on;
            }
            Message::ClipReleased => {
                self.clip_highlights = false;
                self.clip_shadows = false;
            }
            Message::TogglePick => {
                if self.picking {
                    // done: one re-demosaic with the marks, and remember them
                    self.picking = false;
                    self.pick_view = None;
                    self.reset_view_for_pick();
                    if let Some(loaded) = &self.loaded {
                        let raw = &loaded.raw;
                        library::remember_defects(&raw.body(), raw.width, raw.height, &self.picks);
                    }
                    processing::set_picked(self.picks.clone());
                    return self.redevelop();
                }
                let Some(loaded) = &self.loaded else {
                    return Task::none();
                };
                self.cropping = false;
                self.picking = true;
                let (generation, raw, image) =
                    (self.generation, loaded.raw.clone(), loaded.image.clone());
                self.reset_view_for_pick();
                return Task::perform(
                    async move {
                        tokio::task::spawn_blocking(move || image.raw_view(&raw))
                            .await
                            .map_err(|e| format!("Worker failed: {e}"))
                    },
                    move |result| match result {
                        Ok(view) => Message::PickView(generation, view),
                        Err(error) => Message::Developed(generation, Err(error)),
                    },
                );
            }
            Message::PickView(generation, view) => {
                if generation == self.generation && self.picking {
                    self.pick_view = Some(view);
                }
            }
            Message::Preview(Interaction::Click(position)) => {
                if let (true, Some(view)) = (self.picking, &self.pick_view) {
                    let size = Size::new(view.width as f32, view.height as f32);
                    let (scale, center) = self.view.resolve(self.preview_size, size);
                    let x =
                        (center.x + (position.x - self.preview_size.width / 2.0) / scale).floor();
                    let y =
                        (center.y + (position.y - self.preview_size.height / 2.0) / scale).floor();
                    if x >= 0.0 && y >= 0.0 && x < size.width && y < size.height {
                        let site = (y as usize, x as usize);
                        match self.picks.iter().position(|&p| p == site) {
                            Some(i) => {
                                self.picks.remove(i);
                            }
                            None => self.picks.push(site),
                        }
                    }
                }
            }
            Message::CropAngle(degrees) => {
                self.crop = self.crop.with_angle(degrees.to_radians());
                self.constrain_crop();
            }
            Message::CropReset => self.crop = Crop::default(),
            Message::CropFit => {
                self.crop_fit = !self.crop_fit;
                self.constrain_crop();
            }
            Message::CropAspect(aspect) => {
                self.crop.aspect = aspect;
                self.constrain_crop();
            }
            Message::Preview(Interaction::Zoom { factor, anchor }) => {
                if let Some(image) = self.image_size() {
                    self.view = self.view.zoom_by(factor, anchor, self.preview_size, image);
                }
            }
            Message::Preview(Interaction::ToggleZoom(anchor)) => {
                if let Some(image) = self.image_size() {
                    self.view = self.view.toggle_zoom(anchor, self.preview_size, image);
                }
            }
            Message::Preview(Interaction::Pan(delta)) => {
                if let Some(image) = self.image_size() {
                    self.view = self.view.pan_by(delta, self.preview_size, image);
                }
            }
            Message::ZoomFit => self.view = ViewState::default(),
            Message::ToggleLibrary => {
                self.library_visible = !self.library_visible;
                self.library_opening = self.library_visible.then_some(0.0);
                self.apply_pane_widths();
            }
            Message::RotateLeft | Message::RotateRight => {
                let turn = match message {
                    Message::RotateRight => Rotation::CLOCKWISE,
                    _ => Rotation::COUNTER_CLOCKWISE,
                };
                // Keeps the point at the centre of the pane there.
                if let Some(image) = self.image_size() {
                    self.view = self.view.rotated(turn, image);
                }
                self.rotation = self.rotation.then(turn);
                self.crop = self.crop.rotated(turn);
                self.constrain_crop();
            }
            Message::BlackPoint(percent) => {
                self.adjustments.black_point = percent / 100.0;
                self.clip_shadows = false;
            }
            Message::Exposure(value) => {
                self.adjustments.exposure = value;
                self.clip_highlights = false;
            }
            Message::Temperature(value) => {
                self.adjustments.temperature = value;
            }
            Message::Tint(value) => {
                self.adjustments.tint = value;
            }
            Message::Saturation(value) => {
                self.adjustments.saturation = value;
            }
            Message::Vibrance(value) => {
                self.adjustments.vibrance = value;
            }
            Message::Contrast(value) => {
                self.adjustments.contrast = value;
            }
            Message::LocalContrast(value) => {
                self.adjustments.local_contrast = value;
            }
            Message::Sharpness(value) => {
                self.adjustments.sharpness = value;
            }
            Message::AddedGrain(value) => {
                self.adjustments.added_grain = value;
            }
            Message::ToeKnee(value) => self.adjustments.toe_power = value,
            Message::ShoulderKnee(value) => self.adjustments.shoulder_power = value,
            Message::Midpoint(encoded) => {
                self.adjustments.midpoint = tone::midpoint_from_position(encoded);
                self.midpoint_dragging = false;
            }
            Message::MidpointDrag(encoded) => {
                self.adjustments.midpoint = tone::midpoint_from_position(encoded);
                self.midpoint_dragging = true;
            }
            Message::MidpointReleased => self.midpoint_dragging = false,
            Message::HistogramMidpoint(x) => {
                self.adjustments.midpoint = histogram::srgb_decode(x).max(tone::MIDPOINT_MIN);
                self.midpoint_dragging = true;
            }
            Message::HistogramContrast(y) => {
                let target = tone::encode(histogram::srgb_decode(y));
                self.adjustments.contrast =
                    tone::Curve::contrast_for(&self.adjustments, target, CONTRAST_RANGE);
            }
            Message::MixColor(color) => {
                let color = color.min(color_edit::MIX_COLORS - 1);
                if std::mem::take(&mut self.long_pressed) {
                    return Task::none();
                }
                select(&mut self.mix_selected, &mut self.mix_color, color, false);
            }
            Message::MixMove(param, moved) => {
                for (i, value) in moved {
                    self.adjustments
                        .color_mix
                        .set(i.min(color_edit::MIX_COLORS - 1), param, value);
                }
            }
            Message::MixReset(color) => {
                for param in MIX_PARAMS {
                    self.adjustments.color_mix.set(color, param, 0.0);
                }
                // The reset also deselects it.
                select_off(&mut self.mix_selected, &mut self.mix_color, color);
            }
            Message::Mix(color, param, value) => {
                // A slider moved on a selected colour sets them all.
                if self.mix_selected[color.min(color_edit::MIX_COLORS - 1)] {
                    for i in (0..color_edit::MIX_COLORS).filter(|&i| self.mix_selected[i]) {
                        self.adjustments.color_mix.set(i, param, value);
                    }
                }
            }
            Message::GradeRange(range) => {
                let range = range.min(color_edit::RANGES - 1);
                if std::mem::take(&mut self.long_pressed) {
                    return Task::none();
                }
                select(&mut self.grade_selected, &mut self.grade_range, range, false);
            }
            Message::GradeMove(param, moved) => {
                for (i, value) in moved {
                    self.adjustments.grading.set(i, param, value);
                }
            }
            Message::GradeReset(range) => {
                let d = color_edit::Grading::default();
                for param in GRADE_PARAMS {
                    self.adjustments.grading.set(range, param, d.get(range, param));
                }
                select_off(&mut self.grade_selected, &mut self.grade_range, range);
            }
            Message::Grade(range, param, value) => {
                // A slider moved on a selected range sets them all.
                if self.grade_selected[range.min(color_edit::RANGES - 1)] {
                    for i in (0..color_edit::RANGES).filter(|&i| self.grade_selected[i]) {
                        self.adjustments.grading.set(i, param, value);
                    }
                }
            }
            Message::EstimateReady(key, result) => {
                self.jpeg_estimating = None;
                match result {
                    Ok(Some(bytes)) => self.jpeg_estimate = Some((key, bytes)),
                    // The preview has not been drawn yet: try again.
                    Ok(None) => {}
                    Err(error) => {
                        // Not retried for this key until something changes.
                        self.jpeg_estimate = Some((key, 0));
                        log::write(&format!("Error: {error}"));
                    }
                }
            }
            Message::HistogramReady((image_id, _), result) => {
                self.histogram_running = false;
                let current = self.loaded.as_ref().map(|l| l.image.id);
                match result {
                    Ok(Some(histogram)) if current == Some(image_id) => {
                        self.histogram = Some(histogram)
                    }
                    Ok(Some(_)) => {}
                    // The preview has not been drawn yet: try again.
                    Ok(None) => self.histogram_for = None,
                    // Then the preview cannot show the image either.
                    Err(error)
                        if current == Some(image_id) && error.starts_with("Out of GPU memory") =>
                    {
                        self.fail(error)
                    }
                    Err(error) => log::write(&format!("Error: {error}")),
                }
            }
            Message::HistogramRatio(ratio) => {
                self.histogram_ratio = ratio.clamp(HISTOGRAM_MIN_RATIO, HISTOGRAM_MAX_RATIO);
            }
            Message::ResetSection(section) => {
                let d = Adjustments::default();
                match section {
                    Section::ColorMixer => {
                        self.adjustments.color_mix = d.color_mix;
                        self.mix_selected = [false; color_edit::MIX_COLORS];
                    }
                    Section::ColorGrading => {
                        self.adjustments.grading = d.grading;
                        self.grade_selected = [false; color_edit::RANGES];
                    }
                    _ => {}
                }
            }
            Message::ResetControl(control) => {
                let d = Adjustments::default();
                let a = &mut self.adjustments;
                match control {
                    Control::Grain => a.grain = d.grain,
                    Control::Noise(i) => {
                        self.noise_set.ab[i] = None;
                        return self.redevelop();
                    }
                    Control::Iso => {
                        self.noise_set = NoiseSet::default();
                        return self.redevelop();
                    }
                    Control::Temperature => a.temperature = d.temperature,
                    Control::Tint => a.tint = d.tint,
                    Control::BlackPoint => a.black_point = d.black_point,
                    Control::Exposure => a.exposure = d.exposure,
                    Control::Saturation => a.saturation = d.saturation,
                    Control::Vibrance => a.vibrance = d.vibrance,
                    Control::Contrast => a.contrast = d.contrast,
                    Control::Midpoint => a.midpoint = d.midpoint,
                    Control::LocalContrast => a.local_contrast = d.local_contrast,
                    Control::Sharpness => a.sharpness = d.sharpness,
                    Control::AddedGrain => a.added_grain = d.added_grain,
                    Control::ToeKnee => a.toe_power = d.toe_power,
                    Control::ShoulderKnee => a.shoulder_power = d.shoulder_power,
                    Control::Mix(i, p) => {
                        if self.mix_selected[i] {
                            for j in (0..color_edit::MIX_COLORS).filter(|&j| self.mix_selected[j]) {
                                a.color_mix.set(j, p, 0.0);
                            }
                        } else {
                            a.color_mix.set(i, p, 0.0)
                        }
                    }
                    Control::Grade(i, p) => {
                        for j in (0..color_edit::RANGES)
                            .filter(|&j| self.grade_selected[j] || j == i && !self.grade_selected[i])
                        {
                            a.grading.set(j, p, d.grading.get(j, p));
                        }
                    }
                }
            }
            Message::Tab(tab) => {
                let previous = std::mem::replace(&mut self.tab, tab);
                if previous == tab {
                    return Task::none();
                }
                // The preview is in crop mode while the Crop tab is open. Its
                // view is in the frame's space, which the crop changed.
                if previous == SidebarTab::Crop || tab == SidebarTab::Crop {
                    self.cropping = tab == SidebarTab::Crop;
                    self.view = ViewState::default();
                }
                // Leaving Repair finishes picking: one re-demosaic with the marks.
                if previous == SidebarTab::Repair && self.picking {
                    return self.handle(Message::TogglePick);
                }
            }
            Message::AutoAll => self.adjustments = self.auto_adjustments(),
            Message::ResetAdjustments => {
                self.adjustments = Adjustments::default();
                self.mix_selected = [false; color_edit::MIX_COLORS];
                self.grade_selected = [false; color_edit::RANGES];
                self.noise_set = NoiseSet::default();
                return self.redevelop();
            }
            Message::PaneResized(pane_grid::ResizeEvent { split, ratio }) => {
                let pane_grid::Node::Split { id: outer, .. } = self.panes.layout() else {
                    return Task::none();
                };
                let width = self.window_width;
                if split == *outer {
                    // Dragged below the collapse width the library folds to
                    // its strip (keeping the width it reopens at); dragged
                    // back out past it, it opens again.
                    let wanted = ratio * width;
                    if !self.library_visible {
                        let trough = self.library_trough.unwrap_or(0.0).min(wanted);
                        if wanted >= trough + LIBRARY_EXPAND_WIDTH {
                            self.library_visible = true;
                            self.library_trough = None;
                            self.library_opening = Some(wanted);
                        } else {
                            self.library_trough = Some(trough);
                        }
                    } else if wanted >= LIBRARY_COLLAPSE_WIDTH {
                        self.library_opening = None;
                        self.left_width = wanted;
                    } else if let Some(peak) = self.library_opening {
                        // Being pulled out: stay open until the drag turns back.
                        let peak = peak.max(wanted);
                        if wanted < peak - LIBRARY_REFOLD_DISTANCE {
                            self.library_opening = None;
                            self.library_visible = false;
                            self.library_trough = Some(wanted);
                        } else {
                            self.library_opening = Some(peak);
                        }
                    } else {
                        self.library_visible = false;
                        self.library_trough = Some(wanted);
                    }
                } else {
                    self.right_width = (1.0 - ratio) * (width - self.library_width());
                }
                self.apply_pane_widths();
            }
            Message::MousePressed => {
                self.mouse_down = true;
                self.long_pressed = false;
                if let Some((group, index)) = self.round_hover {
                    // Held for `LONG_PRESS`, the button is selected alone.
                    self.round_press += 1;
                    let press = self.round_press;
                    return Task::perform(
                        async {
                            let _ = tokio::task::spawn_blocking(|| std::thread::sleep(LONG_PRESS))
                                .await;
                        },
                        move |()| Message::RoundLongPress(press, group, index),
                    );
                }
            }
            Message::RoundHover(hover) => self.round_hover = hover,
            Message::RoundLongPress(press, group, index) => {
                if press == self.round_press
                    && self.mouse_down
                    && self.round_hover == Some((group, index))
                {
                    self.long_pressed = true;
                    match group {
                        Round::Mix => {
                            select(&mut self.mix_selected, &mut self.mix_color, index, true)
                        }
                        Round::Grade => {
                            select(&mut self.grade_selected, &mut self.grade_range, index, true)
                        }
                    }
                }
            }
            Message::MouseReleased => {
                self.mouse_down = false;
                self.library_trough = None;
                self.library_opening = None;
                let widths = (self.left_width, self.right_width);
                if widths != self.saved_widths {
                    self.saved_widths = widths;
                    library::remember_pane_widths(widths.0, widths.1);
                }
                if self.histogram_ratio != self.saved_histogram_ratio {
                    self.saved_histogram_ratio = self.histogram_ratio;
                    library::remember_histogram_ratio(self.histogram_ratio);
                }
            }
            Message::WindowResized(size) => {
                self.window_width = size.width;
                self.apply_pane_widths();
            }
            Message::ToggleSection(section) => {
                if !self.collapsed.remove(&section) {
                    self.collapsed.insert(section);
                }
            }
            Message::EditValue(control) => {
                let Some(shown) = control.shown(self) else {
                    return Task::none();
                };
                self.editing = Some((control, control.format(shown)));
                return Task::batch([
                    iced::widget::operation::focus(VALUE_INPUT_ID),
                    iced::widget::operation::select_all(VALUE_INPUT_ID),
                ]);
            }
            Message::EditInput(input) => {
                if let Some((_, text)) = &mut self.editing {
                    *text = input;
                }
            }
            Message::EditCommit => {
                if let Some((control, input)) = self.editing.take() {
                    if let Some(message) = control.typed(self, &input) {
                        return self.handle(message);
                    }
                }
            }
            Message::EditCancel => self.editing = None,
            Message::Nudge(control, notches) => {
                if let Some(message) = control.nudged(self, notches) {
                    return self.handle(message);
                }
            }
            Message::Ctrl(on) => self.ctrl = on,
            Message::DropHover(on) => self.drop_hover = on,
            Message::ExportFormat(format) => self.export_format = format,
            Message::ExportCompression(level) => self.export_compression = level,
            Message::ExportQuality(q) => self.export_quality = q.round().clamp(1.0, 100.0) as u8,
            Message::Export => {
                let Some(loaded) = &self.loaded else {
                    return Task::none();
                };
                let name = std::path::Path::new(&loaded.image.name)
                    .with_extension(self.export_format.extension());
                let mut dialog = rfd::AsyncFileDialog::new()
                    .set_title("Export image")
                    .set_file_name(name.to_string_lossy())
                    .add_filter(
                        self.export_format.to_string(),
                        &[self.export_format.extension()],
                    );
                if let Some(dir) = library::remembered_export_folder().or_else(|| {
                    self.current
                        .as_ref()
                        .and_then(|p| p.parent())
                        .map(|p| p.to_path_buf())
                }) {
                    dialog = dialog.set_directory(dir);
                }
                return Task::perform(dialog.save_file(), |handle| {
                    Message::ExportPath(handle.map(|h| h.path().to_path_buf()))
                });
            }
            Message::ExportPath(None) => {}
            Message::ExportPath(Some(mut path)) => {
                let Some(loaded) = &self.loaded else {
                    return Task::none();
                };
                if path.extension().is_none() {
                    path.set_extension(self.export_format.extension());
                }
                if let Some(dir) = path.parent() {
                    library::remember_export_folder(dir);
                }
                self.exporting = true;
                let (image, adjustments, rotation, crop, lens, format, quality, compression) = (
                    loaded.image.clone(),
                    self.adjustments,
                    self.rotation,
                    self.crop,
                    self.drawn_lens(),
                    self.export_format,
                    self.export_quality,
                    self.export_compression,
                );
                return Task::perform(
                    async move {
                        tokio::task::spawn_blocking(move || {
                            let (device, queue) =
                                preview::gpu().ok_or("The preview has not been drawn yet")?;
                            let view = export::View {
                                adjustments,
                                rotation,
                                crop,
                                lens,
                            };
                            export::export(
                                &device,
                                &queue,
                                &image,
                                view,
                                format,
                                quality,
                                compression,
                                &path,
                            )
                            .map(|()| path)
                        })
                        .await
                        .map_err(|e| e.to_string())?
                    },
                    Message::Exported,
                );
            }
            Message::Exported(result) => {
                self.exporting = false;
                if let Err(error) = result {
                    self.fail(format!("Export failed: {error}"));
                }
            }
            Message::Dropped(path) => {
                self.drop_hover = false;
                return if path.is_dir() {
                    self.open_folder(path)
                } else {
                    self.handle(Message::FileChosen(Some(path)))
                };
            }
        }
        Task::none()
    }

    /// Lists `folder` in the file browser and starts loading the thumbnails.
    fn open_folder(&mut self, folder: PathBuf) -> Task<Message> {
        library::remember_folder(&folder);
        let paths = library::list(&folder);
        self.entries = paths
            .iter()
            .map(|path| Entry {
                path: path.clone(),
                thumbnail: Thumbnail::Loading,
            })
            .collect();
        self.folder = Some(folder);
        self.folder_generation += 1;
        let generation = self.folder_generation;
        let (task, handle) = Task::run(library::thumbnails(paths), move |(index, thumbnail)| {
            Message::Thumbnail(generation, index, thumbnail)
        })
        .abortable();
        // Dropping the previous handle aborts that folder's loading.
        self.thumbnail_task = Some(handle.abort_on_drop());
        task
    }

    /// The method the picker shows: the one chosen, else the one the engine
    /// picked for the image shown (none before the first develop).
    fn shown_method(&self) -> Option<Method> {
        let developed = self.loaded.as_ref().filter(|loaded| !loaded.preview);
        self.method.or(developed.map(|loaded| loaded.image.method))
    }

    /// The noise to denoise for: the estimate, if the demosaic stage
    /// denoises. The slider puts some of it back where the image is drawn.
    fn wanted_noise(&self) -> Option<Noise> {
        if self.method.is_some_and(|method| !method.denoises()) {
            return None;
        }
        Some(set_noise(self.loaded.as_ref()?.estimate?, self.noise_set))
    }

    /// log10 of the noise model's a (0) or b (1) the network is told: the
    /// one set, else the estimate's moved to the ISO slider's ISO. `None`
    /// without an estimate.
    fn noise_log10(&self, i: usize) -> Option<f32> {
        let estimate = self.loaded.as_ref()?.estimate?;
        let noise = set_noise(estimate, self.noise_set);
        let x = [noise.a, noise.b][i];
        Some(if x > 0.0 {
            x.log10() as f32
        } else {
            *noise_range(i).start()
        })
    }

    /// Whether the image shown was denoised, so the grain slider has
    /// something to put back.
    fn denoised(&self) -> bool {
        self.loaded
            .as_ref()
            .is_some_and(|loaded| loaded.image.noise.is_some())
    }

    /// The lens corrections switched on that the image's profile has
    /// (`Image::profile`: the file's distortion and vignetting, the measured
    /// CA). All are applied where the image is drawn (preview and export),
    /// so toggling one is instant.
    fn drawn_lens(&self) -> Corrections {
        let available = self
            .loaded
            .as_ref()
            .and_then(|l| l.image.profile.as_deref());
        available.map_or(Corrections::NONE, |l| self.lens.and(l.available()))
    }

    /// Re-demosaics for the noise estimate and with the lens checkbox unless
    /// that is already shown. A run in flight for other settings is
    /// cancelled (it calls this again when it stops), so a slow GPU never
    /// finishes a run nobody wants any more. The quick preview is left
    /// alone: the full demosaic is on its way and catches up when it
    /// arrives.
    fn redevelop(&mut self) -> Task<Message> {
        let Some(loaded) = &self.loaded else {
            return Task::none();
        };
        let settings = Settings {
            noise: self.wanted_noise(),
            method: self.method,
            filter_grain: processing::filter_grain(),
            ca_fix: processing::ca_fix(),
            highlights: processing::highlights(),
            hot_pixels: processing::hot_pixels_fix(),
            picked: processing::picked().0,
        };
        if let Some(run) = &self.redeveloping {
            if run.settings != settings {
                run.cancel.store(true, Ordering::Relaxed);
            }
            return Task::none();
        }
        let noise = settings.noise;
        if loaded.preview
            || (loaded.image.noise == noise
                && loaded.image.filter_grain == settings.filter_grain
                && loaded.image.ca_fix == settings.ca_fix
                && loaded.image.highlights == settings.highlights
                && loaded.image.hot_pixel_fix == settings.hot_pixels
                && loaded.image.picked_version == settings.picked
                && self
                    .method
                    .is_none_or(|method| method == loaded.image.method))
        {
            return Task::none();
        }
        // Its own flag, so cancelling it leaves the file's develop alone;
        // opening another file sets it too.
        let cancel = Arc::new(AtomicBool::new(false));
        self.redeveloping = Some(Redevelop {
            settings,
            cancel: cancel.clone(),
        });
        let generation = self.generation;
        Task::perform(
            redevelop(
                loaded.raw.clone(),
                loaded.estimate,
                self.noise_set,
                self.method,
                cancel,
            ),
            move |result| Message::Redeveloped(generation, result),
        )
    }

    /// Starts a JPEG size estimate if the Export tab shows one that is out of
    /// date and none is in flight.
    fn refresh_estimate(&mut self) -> Task<Message> {
        let Some(loaded) = &self.loaded else {
            self.jpeg_estimate = None;
            return Task::none();
        };
        if self.tab != SidebarTab::Export
            || self.export_format != Format::Jpeg
            || self.jpeg_estimating.is_some()
        {
            return Task::none();
        }
        let view = export::View {
            adjustments: self.adjustments,
            rotation: self.rotation,
            crop: self.crop,
            lens: self.drawn_lens(),
        };
        let key = (loaded.image.id, view, self.export_quality);
        if self.jpeg_estimate.map(|(k, _)| k) == Some(key) {
            return Task::none();
        }
        self.jpeg_estimating = Some(key);
        let image = loaded.image.clone();
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || {
                    let Some((device, queue)) = preview::gpu() else {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        return Ok(None);
                    };
                    export::estimate_jpeg_size(&device, &queue, &image, view, key.2).map(Some)
                })
                .await
                .map_err(|e| format!("Worker failed: {e}"))?
            },
            move |result| Message::EstimateReady(key, result),
        )
    }

    /// Starts a histogram run on a worker thread if the image or the view
    /// has moved on since the last one and none is in flight.
    fn refresh_histogram(&mut self) -> Task<Message> {
        if self.histogram_running {
            return Task::none();
        }
        let Some(loaded) = &self.loaded else {
            self.histogram_for = None;
            self.histogram = None;
            return Task::none();
        };
        let view = export::View {
            adjustments: self.adjustments,
            rotation: self.rotation,
            crop: self.crop,
            lens: self.drawn_lens(),
        };
        let key = (loaded.image.id, view);
        if self.histogram_for == Some(key) {
            return Task::none();
        }
        self.histogram_for = Some(key);
        self.histogram_running = true;
        let image = loaded.image.clone();
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || Histogram::compute(&image, view))
                    .await
                    .map_err(|e| format!("Worker failed: {e}"))?
            },
            move |result| Message::HistogramReady(key, result),
        )
    }

    fn view(&self) -> Element<'_, Message> {
        if OVERLAY_LAYOUT {
            return self.view_center();
        }
        let button_held = self.mouse_down;
        let panes = pane_grid(&self.panes, |_pane, kind, _maximized| {
            pane_grid::Content::new(match kind {
                PaneKind::FileBrowser if self.library_visible => self.view_file_browser(),
                PaneKind::FileBrowser => self.view_collapsed_library(),
                PaneKind::Preview => self.view_center(),
                PaneKind::Tools => match self.tab {
                    SidebarTab::Adjustments => {
                        container(self.view_tools()).style(styling::sidebar).into()
                    }
                    SidebarTab::Crop => self.view_crop(),
                    SidebarTab::Repair => self.view_repair(),
                    SidebarTab::Metadata => self.view_metadata(),
                    SidebarTab::Export => self.view_export(),
                    #[cfg(debug_assertions)]
                    SidebarTab::Debug => self.view_debug(),
                },
            })
        })
        .spacing(1)
        // iced has one minimum for every pane; `apply_pane_widths` holds the
        // wider ones to `PANE_MIN_WIDTH` after each resize.
        .min_size(if self.library_visible {
            LIBRARY_COLLAPSE_WIDTH * 0.7
        } else {
            LIBRARY_STRIP_WIDTH
        })
        .on_resize(PANE_RESIZE_LEEWAY, Message::PaneResized)
        .style(move |theme| styling::panes(theme, button_held));

        // The 1 px gaps between panes show this background as divider lines.
        let content = container(panes).style(styling::pane_dividers);

        content.into()
    }

    // Left pane.

    /// What a collapsed library leaves: an empty strip. Pressing it opens the
    /// library, and the pane grid takes the same press as the start of a drag
    /// on the divider, so the panel can be dragged straight out.
    fn view_collapsed_library(&self) -> Element<'_, Message> {
        mouse_area(
            container(space::horizontal())
                .width(Length::Fill)
                .height(Length::Fill)
                .style(styling::sidebar),
        )
        .on_press(Message::ToggleLibrary)
        .interaction(iced::mouse::Interaction::ResizingHorizontally)
        .into()
    }

    fn view_file_browser(&self) -> Element<'_, Message> {
        let header_button = |glyph, hint: &'static str, message| {
            with_tooltip(
                button(icon(glyph, ICON_SIZE_TOOLBAR))
                    .on_press_maybe((!self.is_loading()).then_some(message))
                    .style(styling::toolbar_button)
                    .padding(SPACING_SMALL),
                hint,
            )
        };
        let folder = self.folder.clone();
        // The path, or the end of it behind ".../" when it does not fit.
        let title = responsive(move |size| {
            let label = match &folder {
                Some(folder) => text(shorten_path(folder, size.width))
                    .size(FONT_SIZE_STANDARD)
                    .wrapping(text::Wrapping::None),
                None => text("Library").size(FONT_SIZE_STANDARD),
            };
            // The responsive area fills its container; centre the text in it.
            container(label)
                .height(Length::Fill)
                .align_y(Alignment::Center)
                .into()
        });
        let header = row![
            container(title)
                .width(Length::Fill)
                .height(FONT_SIZE_STANDARD * 1.5)
                .align_y(Alignment::Center),
            header_button(CODEPOINT_FOLDER_OPEN, "Open folder…", Message::OpenFolder),
            with_tooltip(
                button(icon(CODEPOINT_COLLAPSE_LIBRARY, ICON_SIZE_TOOLBAR))
                    .on_press(Message::ToggleLibrary)
                    .style(styling::toolbar_button)
                    .padding(SPACING_SMALL),
                "Collapse library",
            ),
        ]
        .align_y(Alignment::Center);

        let body: Element<'_, Message> = if self.entries.is_empty() {
            let hint = match self.folder {
                Some(_) => "No raw files in this folder",
                None if drop_supported() => "Open a folder, or drop one here, to browse its photos",
                None => "Open a folder to browse its photos",
            };
            center(
                column![
                    icon(CODEPOINT_FOLDER_OPEN, ICON_SIZE_PLACEHOLDER * 0.6)
                        .style(styling::heading),
                    text(hint)
                        .size(FONT_SIZE_SMALL)
                        .style(styling::muted)
                        .align_x(Alignment::Center),
                    button(text("Open folder…").size(FONT_SIZE_STANDARD))
                        .on_press_maybe((!self.is_loading()).then_some(Message::OpenFolder))
                        .style(styling::row_button(false))
                        .padding([6.0, SPACING_LARGE]),
                ]
                .spacing(SPACING_NORMAL)
                .align_x(Alignment::Center),
            )
            .padding(SPACING_LARGE)
            .into()
        } else {
            column![]
                .push(space().height(SPACING_NORMAL))
                .push(responsive(|size| self.view_thumbnails(size.width)))
                .height(Length::Fill)
                .into()
        };

        sidebar(header, container(body).height(Length::Fill), space())
            .style(styling::sidebar)
            .into()
    }

    /// A scrollable grid of thumbnails with as many columns as fit in `width`.
    fn view_thumbnails(&self, width: f32) -> Element<'_, Message> {
        let available = (width - 2.0 * SPACING_NORMAL).max(THUMBNAIL_MIN_WIDTH);
        let columns =
            (((available + THUMBNAIL_GAP) / (THUMBNAIL_MIN_WIDTH + THUMBNAIL_GAP)) as usize).max(1);
        let cell = (available - THUMBNAIL_GAP * (columns - 1) as f32) / columns as f32;

        let cells: Vec<Element<'_, Message>> = self
            .entries
            .iter()
            .map(|entry| {
                let selected = self.current.as_ref() == Some(&entry.path);
                let side = cell - 2.0 * THUMBNAIL_PADDING;
                let loading = matches!(entry.thumbnail, Thumbnail::Loading);
                // Only a missing or loading thumbnail outlines the square;
                // a picture is outlined itself.
                let outline_square = selected && !matches!(entry.thumbnail, Thumbnail::Ready(..));
                let picture: Element<'_, Message> = match &entry.thumbnail {
                    Thumbnail::Ready(handle, (w, h)) => {
                        let fit = (side / *w as f32).min(side / *h as f32);
                        let (w, h) = ((*w as f32 * fit).round(), (*h as f32 * fit).round());
                        let image = iced::widget::image(handle.clone())
                            .width(w)
                            .height(h)
                            .content_fit(ContentFit::Fill);
                        let outline = container(space())
                            .width(Length::Fill)
                            .height(Length::Fill)
                            .style(move |_: &iced::Theme| styling::thumbnail_outline(selected));
                        center(stack![image, outline].width(w).height(h)).into()
                    }
                    Thumbnail::Missing => {
                        center(icon(CODEPOINT_PHOTO, ICON_SIZE_TOOLBAR).style(styling::muted))
                            .into()
                    }
                    Thumbnail::Loading => space().into(),
                };
                let picture = container(picture)
                    .width(side)
                    .height(side)
                    .clip(true)
                    .style(move |theme: &iced::Theme| {
                        styling::thumbnail_picture(theme, loading, outline_square)
                    });
                let stem = entry
                    .path
                    .file_stem()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let kind = entry
                    .path
                    .extension()
                    .map(|e| e.to_string_lossy().to_uppercase())
                    .unwrap_or_default();
                let label = row![
                    text(stem)
                        .size(FONT_SIZE_SMALLER)
                        .wrapping(text::Wrapping::None),
                    space::horizontal(),
                    text(kind)
                        .size(FONT_SIZE_SMALLER)
                        .wrapping(text::Wrapping::None)
                        .style(styling::muted),
                ]
                .spacing(SPACING_SMALL)
                .width(Length::Fill);
                let content = column![label, picture]
                    .spacing(SPACING_SMALL)
                    .width(Length::Fill);
                button(content)
                    .on_press(Message::FileChosen(Some(entry.path.clone())))
                    .style(styling::thumbnail_button())
                    .padding(THUMBNAIL_PADDING)
                    .width(cell)
                    .clip(true)
                    .into()
            })
            .collect();

        let mut grid = column![].spacing(THUMBNAIL_GAP);
        let mut cells = cells.into_iter().peekable();
        while cells.peek().is_some() {
            grid = grid.push(row(cells.by_ref().take(columns)).spacing(THUMBNAIL_GAP));
        }

        scrollable(container(grid).padding(Padding::from([0.0, SPACING_NORMAL])))
            .width(Length::Fill)
            .height(Length::Fill)
            .direction(styling::scrollbar_direction())
            .style(styling::scrollbar)
            .into()
    }

    // Center pane.

    fn view_center(&self) -> Element<'_, Message> {
        // The adjustments are laid over the preview's right side, with no
        // background of their own.
        let tools = OVERLAY_LAYOUT.then(|| {
            container(
                container(self.view_tools())
                    .width(RIGHT_SIDEBAR_WIDTH)
                    .height(Length::Fill),
            )
            .align_x(alignment::Horizontal::Right)
            .width(Length::Fill)
            .height(Length::Fill)
        });
        let preview = stack![self.view_preview()].extend(tools.map(Element::from));
        preview.into()
    }

    /// The floating view controls in the preview's bottom right corner.
    fn view_controls(&self) -> Option<Element<'_, Message>> {
        // Only while zoomed in or out of the fit view; the zoom is the
        // crop's in crop mode.
        let Zoom::Scale(scale) = self.view.zoom else {
            return None;
        };
        if self.loaded.is_none() || self.cropping {
            return None;
        }
        let label = button(
            row![
                text(format!("{:.0}%", scale * 100.0)).size(FONT_SIZE_STANDARD),
                icon(CODEPOINT_RESTART, ICON_SIZE_TOOLBAR),
            ]
            .spacing(SPACING_SMALL)
            .align_y(Alignment::Center),
        )
        .on_press(Message::ZoomFit)
        .style(styling::floating_button)
        .padding([SPACING_SMALL, SPACING_NORMAL]);

        Some(
            container(container(label).style(styling::floating))
                .width(Length::Fill)
                .height(Length::Fill)
                .align_x(Alignment::Center)
                .align_y(Alignment::End)
                .padding(SPACING_LARGE)
                .into(),
        )
    }

    fn view_preview(&self) -> Element<'_, Message> {
        // picking: the raw mosaic as it is, whatever is set for the image
        let picking = self.pick_view.as_ref().filter(|_| self.picking);
        let picture = shader(Preview {
            image: picking.or(self.loaded.as_ref().map(|l| &l.image)),
            view: self.view,
            adjustments: if picking.is_some() {
                Adjustments::default()
            } else {
                self.adjustments
            },
            rotation: if picking.is_some() {
                Rotation::default()
            } else {
                self.rotation
            },
            crop: if picking.is_some() {
                Crop::default()
            } else {
                self.crop
            },
            cropping: self.cropping && picking.is_none(),
            lens: if picking.is_some() {
                Corrections::NONE
            } else {
                self.drawn_lens()
            },
            picking: picking.is_some(),
            // not over the raw mosaic of the eyedropper
            clip_highlights: (self.clip_highlights || self.overlays[Overlay::Highlights as usize])
                && picking.is_none(),
            clip_shadows: (self.clip_shadows || self.overlays[Overlay::Shadows as usize])
                && picking.is_none(),
            crossover: (self.midpoint_dragging || self.overlays[Overlay::Crossover as usize])
                && picking.is_none(),
            grid: self.grid,
            on_interaction: Message::Preview,
        })
        .width(Length::Fill)
        .height(Length::Fill);

        let placeholder = |codepoint, message: String, style: fn(&iced::Theme) -> text::Style| {
            center(
                column![
                    icon(codepoint, ICON_SIZE_PLACEHOLDER).style(style),
                    text(message).size(FONT_SIZE_STANDARD).style(style),
                ]
                .spacing(SPACING_NORMAL)
                .align_x(Alignment::Center),
            )
        };

        let overlay = match &self.status {
            Status::Loading(_) if self.loaded.is_some() => None,
            Status::Loading(name) => Some(placeholder(
                CODEPOINT_HOURGLASS,
                format!("Developing {name}…"),
                styling::muted,
            )),
            Status::Error(error) => {
                Some(placeholder(CODEPOINT_ERROR, error.clone(), styling::error))
            }
            Status::Idle if self.loaded.is_none() => Some(center(
                column![
                    icon(CODEPOINT_PHOTO, ICON_SIZE_PLACEHOLDER).style(styling::heading),
                    text("Open a raw file to start")
                        .size(FONT_SIZE_STANDARD)
                        .style(styling::muted),
                    button(text("Open raw file…").size(FONT_SIZE_STANDARD))
                        .on_press(Message::OpenFile)
                        .style(styling::row_button(true))
                        .padding([6.0, SPACING_LARGE]),
                ]
                .spacing(SPACING_NORMAL)
                .align_x(Alignment::Center),
            )),
            Status::Idle => None,
        };

        // The picture stays up while the slow stages catch up; say so.
        let busy = self.loaded.as_ref().and_then(|loaded| {
            if loaded.preview {
                Some("Developing…")
            } else if self.redeveloping.is_some() {
                Some("Updating…")
            } else {
                None
            }
        });
        let busy = busy.map(|label| {
            container(
                container(
                    row![
                        icon(CODEPOINT_HOURGLASS, ICON_SIZE_TINY),
                        text(label).size(FONT_SIZE_SMALL)
                    ]
                    .spacing(SPACING_SMALL)
                    .align_y(Alignment::Center),
                )
                .padding([5.0, 12.0])
                .style(styling::pill),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Alignment::End)
            .align_y(Alignment::Start)
            .padding(SPACING_NORMAL)
        });

        // Only while the library is collapsed.
        let library_button = (!self.library_visible).then(|| {
            container(
                container(
                    button(icon(CODEPOINT_EXPAND_LIBRARY, ICON_SIZE_TOOLBAR))
                        .on_press(Message::ToggleLibrary)
                        .style(styling::floating_button)
                        .padding(SPACING_SMALL),
                )
                .style(styling::floating),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Alignment::Start)
            .align_y(Alignment::Start)
            .padding(SPACING_LARGE)
        });

        let drop = self.drop_hover.then(|| {
            container(center(
                container(text("Drop to open").size(FONT_SIZE_STANDARD))
                    .padding([8.0, SPACING_LARGE])
                    .style(styling::pill),
            ))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(styling::drop_target)
        });

        let markers = picking.map(|view| {
            canvas(Markers {
                sites: &self.picks,
                view: self.view,
                image: Size::new(view.width as f32, view.height as f32),
            })
            .width(Length::Fill)
            .height(Length::Fill)
        });
        let pick_pill = self.picking.then(|| {
            let label = if picking.is_some() {
                format!(
                    "Click a hot or dead pixel to mark it, click again to unmark · {} marked",
                    self.picks.len()
                )
            } else {
                String::from("Preparing the raw mosaic…")
            };
            container(
                container(text(label).size(FONT_SIZE_STANDARD))
                    .padding([6.0, SPACING_LARGE])
                    .style(styling::pill),
            )
            .width(Length::Fill)
            .padding(SPACING_NORMAL)
            .align_x(Alignment::Center)
        });

        container(
            stack![picture]
                .extend(markers.map(Element::from))
                .extend(pick_pill.map(Element::from))
                .extend(overlay.map(Element::from))
                .extend(busy.map(Element::from))
                .extend(library_button.map(Element::from))
                .extend(self.view_controls())
                .extend(drop.map(Element::from)),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .style(if self.cropping {
            styling::preview_cropping
        } else {
            styling::preview
        })
        .into()
    }

    // Right pane.

    /// The message resetting `control`, if it has been changed.
    fn reset_for(&self, control: Control) -> Option<Message> {
        control
            .changed(self)
            .then_some(Message::ResetControl(control))
    }

    /// A segmented control choosing what the right-hand panel shows.
    fn tab_switcher(&self) -> Element<'_, Message> {
        let tab = |label, tab| {
            button(text(label).size(FONT_SIZE_SMALL))
                .on_press(Message::Tab(tab))
                .style(styling::segment(self.tab == tab))
                .padding([4.0, 7.0])
        };
        container(
            row![]
                .extend(
                    SidebarTab::ALL
                        .iter()
                        .map(|&t| Element::from(tab(t.label(), t))),
                )
                .spacing(2),
        )
        .padding(2)
        .style(styling::segmented)
        .into()
    }

    /// The right panel's header: the tabs and `extra` controls.
    fn panel_header<'a>(&'a self, extra: Option<Element<'a, Message>>) -> Element<'a, Message> {
        // The metadata tab is hidden in the overlay layout.
        row![]
            .extend((!OVERLAY_LAYOUT).then(|| self.tab_switcher()))
            .push(space::horizontal())
            .extend(extra)
            .spacing(SPACING_SMALL)
            .align_y(Alignment::Center)
            .into()
    }

    fn view_metadata(&self) -> Element<'_, Message> {
        let rows: Element<'_, Message> = match &self.loaded {
            Some(loaded) => {
                let image = &loaded.image;
                let mut processing = vec![
                    ("Sensor", image.sensor.to_string()),
                    ("Demosaic", image.demosaic.to_string()),
                ];
                if let Some(Noise { a, b }) = loaded.estimate {
                    processing.push(("Noise fit", format!("a {a:.2e}, b {b:.2e}")));
                }
                if let Some(Noise { a, b }) = image.noise.filter(|&n| Some(n) != loaded.estimate) {
                    processing.push(("Noise set", format!("a {a:.2e}, b {b:.2e}")));
                }
                // mid-frame / corner
                if let Some(ca) = &image.corrected_ca {
                    processing.push(("Lateral CA", format!("{}, in the mosaic", ca.summary())));
                } else if let Some(ca) = &image.ca {
                    processing.push(("Lateral CA", format!("{}, drawn", ca.summary())));
                }
                processing.push(("Decode", format!("{} ms", loaded.decode_time.as_millis())));
                processing.push(("Develop", format!("{} ms", loaded.develop_time.as_millis())));
                // The engine exists once an image is open.
                if let Ok(engine) = Engine::get() {
                    processing.push(("GPU", engine.gpu_summary()));
                }
                if let Some(log) = log::path() {
                    processing.push(("Log", log.display().to_string()));
                }

                let mut groups = column![].spacing(SPACING_LARGE);
                for group in ["Camera", "Exposure", "File"] {
                    let rows: Vec<_> = loaded
                        .raw
                        .metadata
                        .iter()
                        .filter(|(label, _)| metadata_group(label) == group)
                        .map(|(label, value)| (*label, value.clone()))
                        .collect();
                    if !rows.is_empty() {
                        groups = groups.push(metadata_table(group, rows));
                    }
                }
                groups.push(metadata_table("Processing", processing)).into()
            }
            None => text("No image open")
                .size(FONT_SIZE_STANDARD)
                .style(styling::muted)
                .into(),
        };

        sidebar(
            self.panel_header(None),
            rule::horizontal(1).style(styling::separator),
            column![
                space().height(SPACING_NORMAL),
                scrollable(container(rows).padding(PANEL_INSET))
                    .width(Length::Fill)
                    .direction(styling::scrollbar_direction())
                    .style(styling::scrollbar)
            ],
        )
        .style(styling::sidebar)
        .into()
    }

    /// A heading that folds `body` away when clicked.
    fn section<'a>(
        &self,
        section: Section,
        body: impl Into<Element<'a, Message>>,
    ) -> Element<'a, Message> {
        let collapsed = self.collapsed.contains(&section);
        let header = button(
            row![
                icon(
                    if collapsed {
                        CODEPOINT_CHEVRON_RIGHT
                    } else {
                        CODEPOINT_EXPAND_MORE
                    },
                    ICON_SIZE_TINY + 2.0,
                ),
                text(section.label())
                    .font(BOLD_FONT)
                    .size(FONT_SIZE_SMALLER),
            ]
            .spacing(SPACING_SMALL)
            .align_y(Alignment::Center),
        )
        .on_press(Message::ToggleSection(section))
        .style(styling::section_header)
        .padding([2.0, 0.0])
        .width(Length::Fill);

        // The reset icon of the sections that have one, right aligned and only
        // while something in the section differs from its default.
        let d = Adjustments::default();
        let a = &self.adjustments;
        let changed = match section {
            Section::ColorMixer => a.color_mix != d.color_mix,
            Section::ColorGrading => a.grading != d.grading,
            _ => false,
        };
        let reset = matches!(section, Section::ColorMixer | Section::ColorGrading).then(|| {
            icon_slot(changed.then_some((CODEPOINT_RESTART, Message::ResetSection(section), false)))
        });
        let header = row![header]
            .extend(reset)
            .spacing(SPACING_SMALL)
            .align_y(Alignment::Center);

        let mut panel = column![header].spacing(SPACING_NORMAL);
        if !collapsed {
            panel = panel.push(body);
        }
        panel.into()
    }

    /// A tab's page: the header (the tabs and `extra`) over scrolling `body`.
    fn tab_page<'a>(
        &'a self,
        extra: Option<Element<'a, Message>>,
        body: impl Into<Element<'a, Message>>,
    ) -> Element<'a, Message> {
        sidebar(
            self.panel_header(extra),
            rule::horizontal(1).style(styling::separator),
            column![
                space().height(SPACING_NORMAL),
                scrollable(container(body).padding(PANEL_INSET))
                    .width(Length::Fill)
                    .direction(styling::scrollbar_direction())
                    .style(styling::scrollbar)
            ],
        )
        .style(styling::sidebar)
        .into()
    }

    /// Crop, rotate and straighten. The preview is in crop mode (the frame,
    /// drag to move, zoom and tilt) while this tab is open.
    fn view_crop(&self) -> Element<'_, Message> {
        let has_image = self.loaded.is_some();
        let label = |label| {
            text(label)
                .size(FONT_SIZE_STANDARD)
                .color(styling::ADJUSTMENT_GRAY)
        };
        let action = |codepoint, tip, message: Message| {
            with_tooltip(
                button(icon(codepoint, ICON_SIZE_TOOLBAR))
                    .on_press_maybe(has_image.then_some(message))
                    .style(styling::toolbar_button)
                    .padding(SPACING_SMALL),
                tip,
            )
        };

        let aspect = row![
            label("Aspect").width(Length::Fill),
            pick_list(Aspect::ALL, Some(self.crop.aspect), Message::CropAspect)
                .text_size(FONT_SIZE_STANDARD),
        ]
        .spacing(SPACING_NORMAL)
        .align_y(Alignment::Center);

        let degrees = self.crop.angle.to_degrees();
        let straighten = Pill::new(degrees, -45.0..=45.0, Message::CropAngle)
            .centered()
            .step(0.1)
            .text(format!("{degrees:+.1}°"))
            .on_reset((self.crop.angle != 0.0).then_some(Message::CropAngle(0.0)))
            .into_element_with(label("Straighten").into());

        let grid = checkbox(self.grid)
            .label("Composition grid")
            .on_toggle(Message::Grid)
            .size(16)
            .text_size(FONT_SIZE_STANDARD);

        let fit = button(text("Keep frame inside image").size(FONT_SIZE_STANDARD))
            .on_press_maybe(has_image.then_some(Message::CropFit))
            .style(styling::row_button(self.crop_fit))
            .padding([6.0, SPACING_LARGE]);

        let turns = row![
            action(CODEPOINT_ROTATE_LEFT, "Rotate left", Message::RotateLeft),
            action(CODEPOINT_ROTATE_RIGHT, "Rotate right", Message::RotateRight),
        ]
        .align_y(Alignment::Center);

        // As the adjustments' reset-all: only while the crop differs from its default.
        let reset = with_tooltip(
            button(icon(CODEPOINT_RESTART, ICON_SIZE_TOOLBAR - 4.0))
                .on_press_maybe((self.crop != Crop::default()).then_some(Message::CropReset))
                .style(styling::toolbar_button)
                .padding(4),
            "Reset crop",
        );

        // The frame's top left corner and size, in pixels of the unrotated image.
        let readout = self.loaded.as_ref().map(|loaded| {
            let source = Size::new(loaded.image.width as f32, loaded.image.height as f32);
            let rotated = self.rotation.size(source);
            let frame = self.crop.frame_size(rotated);
            let top_left = self
                .rotation
                .unapply(self.crop.to_image(iced::Point::ORIGIN, rotated), source);
            let size = self.rotation.size(frame);
            let line = |name: &'static str, value: String| {
                row![
                    label(name).width(Length::Fill),
                    text(value).font(MONO_FONT).size(FONT_SIZE_SMALL),
                ]
            };
            column![
                line("Zoom", format!("{:.0}%", self.crop.zoom * 100.0)),
                line("Position", format!("{:.0}, {:.0}", top_left.x, top_left.y)),
                line("Size", format!("{:.0} × {:.0}", size.width, size.height)),
            ]
            .spacing(SPACING_SMALL)
        });

        let hint = text(
            "Drag the image to move it, the wheel zooms, drag beside the frame to zoom or tilt.",
        )
        .size(FONT_SIZE_SMALL)
        .style(styling::muted);

        self.tab_page(
            Some(reset),
            column![aspect, straighten, fit, grid, turns]
                .extend(readout.map(Element::from))
                .push(hint)
                .spacing(SPACING_LARGE),
        )
    }

    /// File format and compression, and the export button.
    fn view_export(&self) -> Element<'_, Message> {
        let label = |label| {
            text(label)
                .size(FONT_SIZE_STANDARD)
                .color(styling::ADJUSTMENT_GRAY)
        };
        let format = row![
            label("Format").width(Length::Fill),
            pick_list(Format::ALL, Some(self.export_format), Message::ExportFormat)
                .text_size(FONT_SIZE_STANDARD),
        ]
        .spacing(SPACING_NORMAL)
        .align_y(Alignment::Center);
        let setting: Element<'_, Message> = if self.export_format == Format::Jpeg {
            Pill::new(
                self.export_quality as f32,
                1.0..=100.0,
                Message::ExportQuality,
            )
            .step(1.0)
            .text(self.export_quality.to_string())
            .into_element_with(label("Quality").into())
        } else {
            row![
                label("Compression").width(Length::Fill),
                pick_list(
                    Compression::ALL,
                    Some(self.export_compression),
                    Message::ExportCompression,
                )
                .text_size(FONT_SIZE_STANDARD),
            ]
            .spacing(SPACING_NORMAL)
            .align_y(Alignment::Center)
            .into()
        };
        let size = (self.export_format == Format::Jpeg && self.loaded.is_some()).then(|| {
            let line = match self.jpeg_estimate {
                Some((_, bytes)) if bytes > 0 => format!(
                    "Estimated size ≈ {:.1} MB{}",
                    bytes as f64 / 1e6,
                    if self.jpeg_estimating.is_some() {
                        "…"
                    } else {
                        ""
                    }
                ),
                _ if self.jpeg_estimating.is_some() => "Estimating size…".to_string(),
                _ => String::new(),
            };
            Element::from(text(line).size(FONT_SIZE_SMALL).style(styling::muted))
        });
        let enabled = self.loaded.is_some() && !self.exporting;
        let export = button(
            row![
                icon(CODEPOINT_DOWNLOAD, ICON_SIZE_TOOLBAR - 4.0),
                text(if self.exporting {
                    "Exporting…"
                } else {
                    "Export"
                })
                .size(FONT_SIZE_STANDARD),
            ]
            .spacing(SPACING_SMALL)
            .align_y(Alignment::Center),
        )
        .on_press_maybe(enabled.then_some(Message::Export))
        .style(styling::segment(false))
        .padding([SPACING_SMALL, SPACING_NORMAL]);

        self.tab_page(
            None,
            column![format, setting]
                .extend(size)
                .push(export)
                .spacing(SPACING_LARGE),
        )
    }

    /// Hot and dead pixels: the automatic fix and the eyedropper.
    fn view_repair(&self) -> Element<'_, Message> {
        let has_image = self.loaded.is_some();
        // stuck sites replaced by their neighbours' median before anything
        // else, with how many it found (processing::set_hot_pixels)
        let fix = row![
            checkbox(processing::hot_pixels_fix())
                .label("Hot and dead pixels")
                .on_toggle(Message::HotPixels)
                .size(16)
                .text_size(FONT_SIZE_STANDARD),
            space::horizontal(),
            text(match self.loaded.as_ref().map(|loaded| &loaded.image) {
                Some(image) => match (image.hot_pixels, image.picked) {
                    (None, _) => String::new(),
                    (Some(n), 0) => format!("{n} fixed"),
                    (Some(n), m) => format!("{n} fixed ({m} marked)"),
                },
                None => String::new(),
            })
            .size(FONT_SIZE_SMALL)
            .style(styling::muted),
        ]
        .spacing(SPACING_NORMAL)
        .align_y(Alignment::Center);

        let pick = button(
            row![
                icon(CODEPOINT_COLORIZE, ICON_SIZE_TOOLBAR - 4.0),
                text(if self.picking { "Done" } else { "Pick pixels" }).size(FONT_SIZE_STANDARD),
            ]
            .spacing(SPACING_SMALL)
            .align_y(Alignment::Center),
        )
        .on_press_maybe(has_image.then_some(Message::TogglePick))
        .style(styling::segment(self.picking))
        .padding([SPACING_SMALL, SPACING_NORMAL]);

        let hint = text(
            "Pick shows the raw mosaic: click a stuck pixel to mark it, click again to \
             unmark. The marks are kept for this camera body.",
        )
        .size(FONT_SIZE_SMALL)
        .style(styling::muted);

        self.tab_page(None, column![fix, pick, hint].spacing(SPACING_LARGE))
    }

    /// The experimental switches and the noise model, in debug builds.
    #[cfg(debug_assertions)]
    fn view_debug(&self) -> Element<'_, Message> {
        let noise = self.adjustment_with(
            "Demosaic residual grain",
            Control::Grain,
            match self.denoised() {
                true => format!("{:.0}%", self.adjustments.grain * 100.0),
                false => String::from("—"),
            },
            Pill::new(self.adjustments.grain, 0.0..=1.0, Message::GrainAmount),
        );

        let noise_level = |i: usize, label| {
            let control = Control::Noise(i);
            // Without an estimate (the quick stage) there is nothing to set.
            let x = self.noise_log10(i).unwrap_or(*noise_range(i).start());
            let value = match control.shown(self) {
                Some(shown) => control.format(shown),
                None => String::from("—"),
            };
            let slider = Pill::new(x, noise_range(i), move |x| Message::Noise(i, x));
            self.adjustment_with(label, control, value, slider)
        };
        let controls = column![
            row![
                text("Demosaic")
                    .size(FONT_SIZE_STANDARD)
                    .width(Length::Fill),
                pick_list(
                    Method::available(self.loaded.as_ref().map(|loaded| loaded.image.sensor)),
                    self.shown_method(),
                    Message::Method,
                )
                .text_size(FONT_SIZE_STANDARD),
            ]
            .spacing(SPACING_NORMAL)
            .align_y(Alignment::Center),
            // what the grain slider puts back (processing::set_filter_grain)
            checkbox(processing::filter_grain())
                .label("3x3 Middle 5 filter grain")
                .on_toggle(Message::FilterGrain)
                .size(16)
                .text_size(FONT_SIZE_STANDARD),
            // the CA taken out of the mosaic before a network, instead of
            // drawn (processing::set_ca_fix)
            checkbox(processing::ca_fix())
                .label("Lateral CA in the mosaic")
                .on_toggle(Message::CaFix)
                .size(16)
                .text_size(FONT_SIZE_STANDARD),
            // the clipped samples of the mosaic rebuilt from the unclipped
            // ones, after the CA and before the demosaic (processing::set_highlights)
            checkbox(processing::highlights())
                .label("Highlight reconstruction in the mosaic")
                .on_toggle(Message::Highlights)
                .size(16)
                .text_size(FONT_SIZE_STANDARD),
            noise,
            // the noise model itself, which the ISO slider sets
            noise_level(0, "Shot noise (a)"),
            noise_level(1, "Read noise (b)"),
        ]
        .spacing(SPACING_NORMAL);
        self.tab_page(None, controls)
    }

    fn view_tools(&self) -> Element<'_, Message> {
        let adjustments = self.adjustments;

        let temperature = self.adjustment_with(
            "Temperature",
            Control::Temperature,
            match self.white() {
                Some(white) => format!("{:.0} K", white.kelvin(adjustments.temperature)),
                None => format!("{:+.0}", adjustments.temperature * 100.0),
            },
            Pill::new(adjustments.temperature, -1.0..=1.0, Message::Temperature)
                .centered()
                .track(Track::Temperature),
        );
        let tint = self.adjustment_with(
            "Tint",
            Control::Tint,
            format!("{:+.0}", adjustments.tint * 100.0),
            Pill::new(adjustments.tint, -1.0..=1.0, Message::Tint)
                .centered()
                .track(Track::Tint),
        );
        let black_point = self.adjustment_eye(
            "Black point",
            Control::BlackPoint,
            format!("{:+.2}%", adjustments.black_point * 100.0),
            Pill::new(
                adjustments.black_point * 100.0,
                -BLACK_POINT_PERCENT..=BLACK_POINT_PERCENT,
                Message::BlackPointDrag,
            )
            .centered()
            .on_release(Message::ClipReleased),
            Overlay::Shadows,
        );
        let exposure = self.adjustment_eye(
            "Exposure",
            Control::Exposure,
            format!("{:+.2} EV", adjustments.exposure),
            Pill::new(adjustments.exposure, -5.0..=5.0, Message::ExposureDrag)
                .centered()
                .on_release(Message::ClipReleased),
            Overlay::Highlights,
        );
        let saturation = self.adjustment_with(
            "Saturation",
            Control::Saturation,
            format!("{:.0}%", adjustments.saturation * 100.0),
            Pill::new(adjustments.saturation, 0.0..=2.0, Message::Saturation),
        );
        let vibrance = self.adjustment_with(
            "Vibrance",
            Control::Vibrance,
            format!("{:.0}%", adjustments.vibrance * 100.0),
            Pill::new(adjustments.vibrance, 0.0..=2.0, Message::Vibrance),
        );
        let contrast = self.adjustment_with(
            "Contrast",
            Control::Contrast,
            format!("{:.2}", adjustments.contrast),
            Pill::new(adjustments.contrast, CONTRAST_RANGE, Message::Contrast),
        );
        let shoulder_knee = self.adjustment_with(
            "Highlight tone",
            Control::ShoulderKnee,
            format!("{:.2}", adjustments.shoulder_power),
            Pill::new(
                adjustments.shoulder_power,
                tone::POWER_RANGE,
                Message::ShoulderKnee,
            ),
        );
        let toe_knee = self.adjustment_with(
            "Shadow tone",
            Control::ToeKnee,
            format!("{:.2}", adjustments.toe_power),
            Pill::new(adjustments.toe_power, tone::POWER_RANGE, Message::ToeKnee),
        );
        // Along the curve's own axis, so the crossover in the graph follows
        // the handle; the label is linear.
        let midpoint = self.adjustment_eye(
            "Midpoint",
            Control::Midpoint,
            format!("{:.2}%", adjustments.midpoint * 100.0),
            Pill::new(
                tone::midpoint_position(adjustments.midpoint),
                0.0..=1.0,
                Message::MidpointDrag,
            )
            .on_release(Message::MidpointReleased)
            .step(0.005),
            Overlay::Crossover,
        );

        let local_contrast = self.adjustment_with(
            "Local contrast",
            Control::LocalContrast,
            format!("{:.0}", adjustments.local_contrast * 100.0),
            Pill::new(
                adjustments.local_contrast,
                0.0..=1.0,
                Message::LocalContrast,
            ),
        );

        let sharpness = self.adjustment_with(
            "Sharpness",
            Control::Sharpness,
            format!("{:.0}", adjustments.sharpness * 100.0),
            Pill::new(adjustments.sharpness, 0.0..=1.0, Message::Sharpness),
        );

        let added_grain = self.adjustment_with(
            "Grain",
            Control::AddedGrain,
            format!("{:.0}", adjustments.added_grain * 100.0),
            Pill::new(adjustments.added_grain, 0.0..=1.0, Message::AddedGrain),
        );

        // One checkbox per correction the file's lens profile has and the
        // app offers.
        let lens_corrections = self
            .loaded
            .as_ref()
            .and_then(|l| l.image.profile.as_deref())
            .map(|profile| profile.available())
            .filter(|available| available.any())
            .map(|available| {
                lens::Kind::ALL
                    .into_iter()
                    .filter(|&kind| available.get(kind))
                    .fold(row![], |buttons, kind| {
                        let on = self.lens.get(kind);
                        buttons.push(
                            button(
                                text(kind.label())
                                    .size(FONT_SIZE_STANDARD)
                                    .align_y(alignment::Vertical::Center),
                            )
                            .on_press(Message::LensCorrection(kind, !on))
                            .style(styling::toggle_button(on))
                            .height(pill::HEIGHT)
                            .padding([0.0, SPACING_NORMAL]),
                        )
                    })
                    .spacing(SPACING_SMALL)
                    .wrap()
            });
        let iso_value = match Control::Iso.shown(self) {
            Some(shown) => format!("ISO {}", Control::Iso.format(shown)),
            // the quick stage, or a file without an ISO (the slider still works)
            None => match (
                self.noise_set.stops,
                self.loaded.as_ref().and_then(|l| l.estimate),
            ) {
                (Some(stops), Some(_)) => format!("{stops:+.1} EV"),
                _ => String::from("—"),
            },
        };
        let iso_slider = Pill::new(
            self.noise_set.stops.unwrap_or(0.0),
            ISO_STOPS,
            Message::NoiseIso,
        )
        .step(1.0 / 3.0);
        let noise_level =
            self.adjustment_with("Noise level", Control::Iso, iso_value, iso_slider);
        let enhancements = column![
            local_contrast,
            sharpness,
            added_grain
        ]
        .spacing(SPACING_NORMAL);

        let histogram = canvas(histogram::View {
            histogram: self.histogram.as_ref(),
            curve: tone::Curve::new(&adjustments),
            on_resize: Message::HistogramRatio,
            on_midpoint: Message::HistogramMidpoint,
            on_contrast: Message::HistogramContrast,
            on_release: || Message::MidpointReleased,
        });
        // Inset like the sliders, whose corners it shares.
        let mut histogram = column![histogram::Aspect::new(histogram, self.histogram_ratio)];
        if let Some(loaded) = &self.loaded {
            let line = |s: String| text(s).size(FONT_SIZE_SMALL).style(styling::muted);
            let mut info = column![].spacing(2);
            for (left, right) in shot_summary(&loaded.raw) {
                if !left.is_empty() || !right.is_empty() {
                    info = info.push(row![line(left), space().width(Length::Fill), line(right)]);
                }
            }
            histogram = histogram.push(info.padding(Padding {
                top: SPACING_SMALL,
                ..Padding::ZERO
            }));
        }
        let histogram = container(histogram)
            .width(Length::Fill)
            .padding(Padding {
                bottom: SPACING_NORMAL,
                ..PANEL_INSET
            });

        let reset = with_tooltip(
            button(icon(CODEPOINT_RESTART, ICON_SIZE_TOOLBAR - 4.0))
                .on_press_maybe(
                    (adjustments != Adjustments::default()
                        || self.noise_set != NoiseSet::default())
                    .then_some(Message::ResetAdjustments),
                )
                .style(styling::toolbar_button)
                .padding(4),
            "Reset all adjustments",
        );

        let auto = with_tooltip(
            button(icon(CODEPOINT_AUTO, ICON_SIZE_TOOLBAR - 4.0))
                .on_press_maybe(
                    (self.loaded.is_some() && self.auto_adjustments() != adjustments)
                        .then_some(Message::AutoAll),
                )
                .style(styling::toolbar_button)
                .padding(4),
            "Restore the auto tuned values",
        );

        sidebar(
            self.panel_header(Some(row![auto, reset].spacing(SPACING_SMALL).into())),
            histogram,
            scrollable(
                container(
                    column![
                        space().height(SPACING_SMALL),
                        lens_corrections.map(|body| self.section(Section::LensCorrections, body)),
                        self.section(
                            Section::Exposure,
                            column![noise_level, black_point, exposure].spacing(SPACING_NORMAL),
                        ),
                        self.section(
                            Section::Tone,
                            column![midpoint, contrast, shoulder_knee, toe_knee]
                                .spacing(SPACING_NORMAL),
                        ),
                        self.section(
                            Section::Color,
                            column![temperature, tint, saturation, vibrance]
                                .spacing(SPACING_NORMAL),
                        ),
                        self.section(Section::Enhancements, enhancements),
                        self.section(Section::ColorMixer, self.color_mixer()),
                        self.section(Section::ColorGrading, self.color_grading()),
                        space().height(SPACING_SMALL),
                    ]
                    .spacing(SPACING_LARGE),
                )
                .padding(PANEL_INSET),
            )
            .width(Length::Fill)
            .direction(styling::scrollbar_direction())
            .style(styling::scrollbar),
        )
        .into()
    }

    /// The colour mixer: a swatch for each of its colours, and the hue,
    /// saturation and brightness sliders of the selected one, painted with
    /// what they do to it.
    fn color_mixer(&self) -> Element<'_, Message> {
        use color_edit::{HUE_SHIFT_MAX, MIX_COLORS, MIX_HUES, MIX_NAMES, MixParam};
        let selected = self.mix_color.min(MIX_COLORS - 1);
        let swatches = button_row(MIX_COLORS, move |color| {
            let [r, g, b] = styling::ui_oklch(0.72, 0.14, MIX_HUES[color]);
            let swatch = with_tooltip(
                button(space())
                    .on_press(Message::MixColor(color))
                    .style(styling::swatch(
                        iced::Color::from_rgb(r, g, b),
                        self.mix_selected[color],
                    ))
                    .width(SWATCH_SIZE)
                    .height(SWATCH_SIZE),
                MIX_NAMES[color],
            );
            let swatch = round_hover(swatch, Round::Mix, color);
            let edited = MIX_PARAMS
                .iter()
                .any(|&param| self.adjustments.color_mix.get(color, param) != 0.0);
            with_reset_badge(swatch, edited, Message::MixReset(color))
        });
        let hue = MIX_HUES[selected];
        let any = self.mix_selected.iter().any(|&on| on);
        let slider = |label: &'static str, param: MixParam, track: Track| {
            let x = self.adjustments.color_mix.get(selected, param);
            let value = match param {
                MixParam::Hue => format!("{:+.0}°", x * HUE_SHIFT_MAX + 0.0),
                _ => format!("{:+.0}", x * 100.0 + 0.0),
            };
            let markers = (0..MIX_COLORS)
                .filter(|&i| self.mix_selected[i])
                .map(|i| {
                    let [r, g, b] = styling::ui_oklch(0.72, 0.14, MIX_HUES[i]);
                    (
                        i,
                        self.adjustments.color_mix.get(i, param),
                        iced::Color::from_rgb(r, g, b),
                    )
                })
                .collect();
            self.adjustment_with(
                label,
                Control::Mix(selected, param),
                value,
                Pill::new(x, -1.0..=1.0, move |x| Message::Mix(selected, param, x))
                    .centered()
                    .track(track)
                    .markers(markers, move |moved| Message::MixMove(param, moved))
                    .disabled(!any),
            )
        };
        column![
            swatches,
            slider("Hue", MixParam::Hue, Track::HueRange(hue)),
            slider("Saturation", MixParam::Saturation, Track::Saturation(hue)),
            slider("Brightness", MixParam::Brightness, Track::Plain),
        ]
        .spacing(SPACING_NORMAL)
        .into()
    }

    /// The colour grading: a round button for each of shadows, midtones and
    /// highlights, and the target hue and hue shift of the selected ones.
    fn color_grading(&self) -> Element<'_, Message> {
        use color_edit::{GradeParam, Grading, RANGE_NAMES, RANGES};
        let selected = self.grade_range.min(RANGES - 1);
        let grading = &self.adjustments.grading;
        let defaults = Grading::default();
        let ranges = button_row(RANGES, move |range| {
            let band = with_tooltip(
                button(space())
                .on_press(Message::GradeRange(range))
                .style(styling::swatch(
                    iced::Color::from_rgb(BAND_GREYS[range], BAND_GREYS[range], BAND_GREYS[range]),
                    self.grade_selected[range],
                ))
                .width(SWATCH_SIZE)
                .height(SWATCH_SIZE),
                RANGE_NAMES[range],
            );
            let band = round_hover(band, Round::Grade, range);
            let edited = GRADE_PARAMS
                .iter()
                .any(|&p| grading.get(range, p) != defaults.get(range, p));
            with_reset_badge(band, edited, Message::GradeReset(range))
        });
        let any = self.grade_selected.iter().any(|&on| on);
        let markers = |param: GradeParam| -> Vec<(usize, f32, iced::Color)> {
            (0..RANGES)
                .filter(|&i| self.grade_selected[i])
                .map(|i| {
                    let v = BAND_GREYS[i];
                    (
                        i,
                        grading.get(i, param),
                        iced::Color::from_rgb(v, v, v),
                    )
                })
                .collect()
        };
        column![
            ranges,
            self.adjustment_with(
                "Target hue",
                Control::Grade(selected, GradeParam::Hue),
                format!("{:.0}°", grading.hue[selected]),
                Pill::new(grading.hue[selected], 0.0..=360.0, move |x| Message::Grade(
                    selected,
                    GradeParam::Hue,
                    x
                ))
                .step(1.0)
                .centered()
                .track(Track::Hue)
                .markers(markers(GradeParam::Hue), |moved| Message::GradeMove(
                    GradeParam::Hue,
                    moved
                ))
                .disabled(!any),
            ),
            self.adjustment_with(
                "Hue shift",
                Control::Grade(selected, GradeParam::Amount),
                format!("{:.0}%", grading.amount[selected] * 100.0),
                Pill::new(grading.amount[selected], 0.0..=1.0, move |x| {
                    Message::Grade(selected, GradeParam::Amount, x)
                })
                // From grey to the target hue.
                .track(Track::Saturation(grading.hue[selected]))
                .markers(markers(GradeParam::Amount), |moved| Message::GradeMove(
                    GradeParam::Amount,
                    moved
                ))
                .disabled(!any),
            ),
        ]
        .spacing(SPACING_NORMAL)
        .into()
    }

    /// The adjustments with every auto tuned value (tone, knees, saturation,
    /// vibrance) as the auto tone gave it, the rest kept.
    fn auto_adjustments(&self) -> Adjustments {
        let mut adjustments = self.adjustments;
        match self.auto_tone {
            Some(fit) => fit.apply(&mut adjustments),
            None => auto::reset(&mut adjustments),
        }
        adjustments
    }

    /// The value the auto icon of `control` goes back to (what the embedded
    /// JPEG gave, else the default): exposure, contrast and midpoint.
    fn auto_value(&self, control: Control) -> Option<f32> {
        let d = Adjustments::default();
        let fit = self.auto_tone;
        match control {
            Control::Exposure => Some(fit.map_or(d.exposure, |f| f.exposure)),
            Control::Contrast => Some(fit.map_or(d.contrast, |f| f.contrast)),
            Control::Midpoint => Some(fit.map_or(d.midpoint, |f| f.midpoint)),
            _ => None,
        }
    }

    /// The auto icon in the slot after the reset icon: only for the tone
    /// controls, and shown only where they differ from their auto value.
    fn auto_slot(&self, control: Control) -> Option<Element<'static, Message>> {
        let auto = self.auto_value(control)?;
        let a = &self.adjustments;
        let (now, tolerance) = match control {
            Control::Exposure => (a.exposure, 0.005),
            Control::Contrast => (a.contrast, 0.005),
            _ => (a.midpoint, 1e-4),
        };
        let differs = self.loaded.is_some() && (now - auto).abs() > tolerance;
        Some(icon_slot(differs.then_some((
            CODEPOINT_AUTO,
            Message::Auto(control),
            false,
        ))))
    }

    /// A control's label and its icons after it, laid over the left end of
    /// its pill: the eye if it has one, the reset icon (only while the
    /// control is not at its default) and the auto icon.
    fn label_and_icons<'a>(
        &self,
        label: &'a str,
        control: Control,
        eye: Option<Element<'a, Message>>,
    ) -> Element<'a, Message> {
        row![
            text(label).size(FONT_SIZE_STANDARD).color(iced::Color {
                a: if control.disabled(self) {
                    pill::DISABLED_ALPHA
                } else {
                    1.0
                },
                ..styling::ADJUSTMENT_GRAY
            })
        ]
        .extend(eye)
        .extend(
            self.reset_for(control)
                .map(|message| icon_slot(Some((CODEPOINT_RESTART, message, false)))),
        )
        .extend(self.auto_slot(control))
        .spacing(SPACING_SMALL)
        .align_y(Alignment::Center)
        .into()
    }

    /// [`Self::adjustment_with`] with an eye icon before the reset icon that
    /// keeps the slider's preview overlay on.
    fn adjustment_eye<'a>(
        &self,
        label: &'a str,
        control: Control,
        value: String,
        pill: Pill<'a, Message>,
        overlay: Overlay,
    ) -> Element<'a, Message> {
        let on = self.overlays[overlay as usize];
        let eye = icon_slot(Some((
            if on {
                CODEPOINT_VISIBILITY
            } else {
                CODEPOINT_VISIBILITY_OFF
            },
            Message::ToggleOverlay(overlay),
            !on,
        )));
        self.adjustment_titled(label, control, value, pill, Some(eye))
    }

    /// A control: its pill slider with the name at its left and the value at
    /// its right (click the value to type a number over it, click twice to
    /// reset), and its icons beside it. With Ctrl held the wheel nudges it.
    fn adjustment_with<'a>(
        &self,
        label: &'a str,
        control: Control,
        value: String,
        pill: Pill<'a, Message>,
    ) -> Element<'a, Message> {
        self.adjustment_titled(label, control, value, pill, None)
    }

    fn adjustment_titled<'a>(
        &self,
        label: &'a str,
        control: Control,
        value: String,
        pill: Pill<'a, Message>,
        eye: Option<Element<'a, Message>>,
    ) -> Element<'a, Message> {
        let typing = matches!(&self.editing, Some((editing, _)) if *editing == control);
        let pill: Element<'a, Message> = pill
            .text(value)
            .editing(typing)
            .on_value(control.shown(self).map(|_| Message::EditValue(control)))
            .on_reset(self.reset_for(control))
            .into_element_with(self.label_and_icons(label, control, eye));
        // The field is laid over the pill where the value is, so the pill
        // stays in the tree and keeps its place while it is typed in.
        let pill: Element<'a, Message> = match &self.editing {
            Some((editing, typed)) if *editing == control => stack![
                pill,
                container(
                    text_input("", typed)
                        .id(VALUE_INPUT_ID)
                        .on_input(Message::EditInput)
                        .on_submit(Message::EditCommit)
                        .size(FONT_SIZE_SMALL)
                        .padding([4.0, 8.0])
                        .width(VALUE_INPUT_WIDTH)
                        .align_x(Alignment::End)
                        .style(styling::value_input),
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(Padding {
                    right: 10.0,
                    ..Padding::ZERO
                })
                .align_x(alignment::Horizontal::Right)
                .align_y(alignment::Vertical::Center),
            ]
            .height(pill::HEIGHT)
            .into(),
            _ => pill,
        };
        // Always wrapped, so the widget tree does not change (and a field
        // being typed in keeps its focus) when Ctrl goes down.
        let area = mouse_area(pill);
        if self.ctrl && control.shown(self).is_some() {
            area.on_scroll(move |delta| Message::Nudge(control, notches(delta)))
        } else {
            area
        }
        .into()
    }
}

/// File browser | preview | tools, with the sidebars at their default widths.
fn initial_panes() -> pane_grid::State<PaneKind> {
    use pane_grid::{Axis, Configuration};

    let pane = |kind| Box::new(Configuration::Pane(kind));
    let rest = WINDOW_WIDTH - LEFT_SIDEBAR_WIDTH;
    pane_grid::State::with_configuration(Configuration::Split {
        axis: Axis::Vertical,
        ratio: LEFT_SIDEBAR_WIDTH / WINDOW_WIDTH,
        a: pane(PaneKind::FileBrowser),
        b: Box::new(Configuration::Split {
            axis: Axis::Vertical,
            ratio: (rest - RIGHT_SIDEBAR_WIDTH) / rest,
            a: pane(PaneKind::Preview),
            b: pane(PaneKind::Tools),
        }),
    })
}

/// The lens corrections an image with `profile` starts with: [`LENS_DEFAULT`]
/// without the ones the profile marks unverified.
fn lens_default(profile: Option<&lens::LensCorrection>) -> Corrections {
    profile.map_or(LENS_DEFAULT, |lens| LENS_DEFAULT.without(lens.unverified()))
}

/// Decodes and demosaics off the UI thread.
/// Decodes `path` and bins it into a quick low-quality image.
async fn preview(path: PathBuf) -> Result<Loaded, String> {
    tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let raw = raw::decode(&path)?;
        let decode_time = start.elapsed();

        let start = Instant::now();
        let image = Engine::get()?.develop_preview(&raw, Corrections::NONE)?;
        let develop_time = start.elapsed();

        let start = Instant::now();
        let fit =
            library::embedded_preview(&path).and_then(|jpeg| auto::fit_embedded(&image, &jpeg));
        match &fit {
            Some(fit) => log::write(&format!(
                "Auto tone in {:.0} ms: exposure {:+.2}, contrast {:.2}, midpoint {:.3}, \
                 clipped {:.2}%; luma quantile rms {:.4} (defaults {:.4})",
                start.elapsed().as_secs_f32() * 1e3,
                fit.exposure,
                fit.contrast,
                fit.midpoint,
                fit.clipped * 100.0,
                fit.rms,
                fit.rms_default,
            )),
            None => log::write("Auto tone: no embedded preview, defaults"),
        }

        Ok(Loaded {
            estimate: None,
            image,
            raw: Arc::new(raw),
            decode_time,
            develop_time,
            preview: true,
            fit,
        })
    })
    .await
    .map_err(|e| format!("Worker failed: {e}"))?
}

/// The full neural demosaic of a decoded frame.
async fn develop(
    raw: Arc<RawFrame>,
    decode_time: Duration,
    method: Option<Method>,
    cancel: Arc<AtomicBool>,
) -> Result<Loaded, String> {
    tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let engine = Engine::get()?;
        let method = method.unwrap_or_else(|| engine.automatic(&raw));
        // None for the quick stage, which does not denoise.
        let estimate = engine.estimate_noise(&raw, method)?;
        let image = engine.develop(&raw, estimate, Corrections::NONE, method, &cancel)?;
        let develop_time = start.elapsed();

        Ok(Loaded {
            estimate,
            image,
            raw,
            decode_time,
            develop_time,
            preview: false,
            fit: None,
        })
    })
    .await
    .map_err(|e| format!("Worker failed: {e}"))?
}

/// What a re-demosaic makes: the noise it denoises for (as far as known
/// when it starts), the lens corrections and the stage (`None`: the engine's
/// choice).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Settings {
    noise: Option<Noise>,
    method: Option<Method>,
    filter_grain: bool,
    ca_fix: bool,
    highlights: bool,
    hot_pixels: bool,
    /// [`processing::set_picked`]'s version.
    picked: u64,
}

/// A re-demosaic in flight.
#[derive(Debug)]
struct Redevelop {
    settings: Settings,
    cancel: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
struct Redeveloped {
    image: Image,
    /// The noise estimate of the stage that made `image`, if it denoises.
    estimate: Option<Noise>,
    develop_time: Duration,
}

/// Demosaics `raw` again with the lens corrections and the chosen stage, off
/// the UI thread, denoising for the estimate. `estimate` is the
/// noise estimate already known, if any: switching from the quick stage back
/// to a network has none yet, so it is measured then. Both networks fit the
/// same noise (same CFA and period), so switching between them keeps it.
/// `set` replaces the estimate's a or b (`App::noise_set`).
async fn redevelop(
    raw: Arc<RawFrame>,
    estimate: Option<Noise>,
    set: NoiseSet,
    method: Option<Method>,
    cancel: Arc<AtomicBool>,
) -> Result<Redeveloped, String> {
    tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let engine = Engine::get()?;
        let method = method.unwrap_or_else(|| engine.automatic(&raw));
        let estimate = match (method, estimate) {
            (method, _) if !method.denoises() => None,
            (_, Some(estimate)) => Some(estimate),
            (_, None) => engine.estimate_noise(&raw, method)?,
        };
        let noise = estimate.map(|estimate| set_noise(estimate, set));
        let image = engine.develop(&raw, noise, Corrections::NONE, method, &cancel)?;
        let develop_time = start.elapsed();
        Ok(Redeveloped {
            image,
            estimate,
            develop_time,
        })
    })
    .await
    .map_err(|e| format!("Worker failed: {e}"))?
}

/// The noise sliders' range, log10 of a (0) or b (1): what the networks were
/// trained on, and a decade below it whose left end is 0 (the networks
/// were trained on noise-free patches too, told a = b = 0).
fn noise_range(i: usize) -> std::ops::RangeInclusive<f32> {
    let (lo, hi) = [A_RANGE, B_RANGE][i];
    lo.log10() as f32 - 1.0..=hi.log10() as f32
}

/// a (0) or b (1) at slider position `x`: 10^x, and 0 at the left end.
fn noise_value(i: usize, x: f32) -> f64 {
    if x <= *noise_range(i).start() {
        0.0
    } else {
        10f64.powf(x as f64)
    }
}

/// The noise the panel sets in place of the image's estimate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct NoiseSet {
    /// The ISO slider: stops above (or below) the file's ISO, `None` = the
    /// estimate as fitted.
    stops: Option<f32>,
    /// log10 of a and b set in Debug, replacing what `stops` gives.
    ab: [Option<f32>; 2],
}

/// The ISO slider's range, in stops from the file's ISO.
const ISO_STOPS: std::ops::RangeInclusive<f32> = -4.0..=6.0;

/// `estimate` (fitted at the file's ISO) as it would be `stops` higher, with
/// no table of the camera: a gain `r = 2^stops` times the file's scales the
/// shot noise's variance (a, relative to white) by `r`, and the read noise's
/// (b) by between 1 (noise after the amplifier) and `r^2` (before it, which
/// dominates on modern sensors above base ISO); `r^2` is taken. Clamped to
/// what the networks were trained on.
fn noise_at_stops(estimate: Noise, stops: f32) -> Noise {
    let r = (stops as f64).exp2();
    Noise {
        a: (estimate.a * r).min(A_RANGE.1),
        b: (estimate.b * r * r).min(B_RANGE.1),
    }
}

/// `estimate` with the noise set in the panel (`App::noise_set`): moved to
/// the ISO slider's ISO, then a or b replaced where set in Debug.
fn set_noise(estimate: Noise, set: NoiseSet) -> Noise {
    let moved = noise_at_stops(estimate, set.stops.unwrap_or(0.0));
    Noise {
        a: set.ab[0].map_or(moved.a, |x| noise_value(0, x)),
        b: set.ab[1].map_or(moved.b, |x| noise_value(1, x)),
    }
}

/// `iso` rounded to the nearest of the cameras' third-stop series (100,
/// 125, 160, 200, 250, 320, 400, 500, 640, 800, ...).
fn nominal_iso(iso: f32) -> f32 {
    const SERIES: [f32; 11] = [1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.4, 8.0, 10.0];
    if !(iso > 0.0) {
        return iso;
    }
    let decade = 10f32.powi(iso.log10().floor() as i32);
    let m = iso / decade;
    let best = SERIES
        .into_iter()
        .min_by(|x, y| (m / x).ln().abs().total_cmp(&(m / y).ln().abs()))
        .unwrap_or(m);
    // rounds the float error of 1.25 and 1.6 times a decade away
    (best * decade).round()
}

/// `count` round buttons (`build` makes the one at an index) centred in a
/// row, `BUTTON_GAP_MAX` apart, or closer if the width is too small for that.
fn button_row<'a>(
    count: usize,
    build: impl Fn(usize) -> Element<'a, Message> + 'a,
) -> Element<'a, Message> {
    responsive(move |size| {
        let gaps = count.saturating_sub(1).max(1) as f32;
        let gap = ((size.width - count as f32 * SWATCH_SIZE) / gaps).clamp(0.0, BUTTON_GAP_MAX);
        let buttons = (0..count).fold(row![].spacing(gap), |row, i| row.push(build(i)));
        container(buttons)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(alignment::Horizontal::Center)
            .align_y(alignment::Vertical::Center)
            .into()
    })
    .height(SWATCH_SIZE)
    .into()
}

/// Toggles `index` of a row of round buttons in the selection, or with
/// `alone` (a long press) selects just it. `shown` is the one whose values
/// the sliders show.
fn select<const N: usize>(selected: &mut [bool; N], shown: &mut usize, index: usize, alone: bool) {
    if alone {
        *selected = [false; N];
        selected[index] = true;
    } else {
        selected[index] = !selected[index];
    }
    if selected[index] {
        *shown = index;
    } else if *shown == index {
        if let Some(other) = selected.iter().position(|&on| on) {
            *shown = other;
        }
    }
}

/// Deselects `index` of a row of round buttons; `shown` moves to another
/// selected one if it was this.
fn select_off<const N: usize>(selected: &mut [bool; N], shown: &mut usize, index: usize) {
    selected[index] = false;
    if *shown == index {
        if let Some(other) = selected.iter().position(|&on| on) {
            *shown = other;
        }
    }
}

/// `content`, with a small reset badge sticking out of its top right corner
/// when `edited`.
fn with_reset_badge<'a>(
    content: impl Into<Element<'a, Message>>,
    edited: bool,
    reset: Message,
) -> Element<'a, Message> {
    let content = content.into();
    if !edited {
        return content;
    }
    let badge = button(center(icon(CODEPOINT_RESTART, BADGE_ICON_SIZE)))
        .on_press(reset)
        .style(styling::badge)
        .padding(0)
        .width(BADGE_SIZE)
        .height(BADGE_SIZE);
    // Nudged out past the corner, so it sticks out.
    let badge = iced::widget::float(badge)
        .translate(|_, _| iced::Vector::new(BADGE_OVERHANG, -BADGE_OVERHANG));
    stack![
        content,
        container(badge)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(alignment::Horizontal::Right)
            .align_y(alignment::Vertical::Top),
    ]
    .width(SWATCH_SIZE)
    .height(SWATCH_SIZE)
    .into()
}

fn icon<'a>(codepoint: char, size: f32) -> text::Text<'a> {
    text(codepoint).font(ICONS_FONT).size(size)
}

fn heading<'a>(label: impl text::IntoFragment<'a>) -> text::Text<'a> {
    text(label)
        .font(BOLD_FONT)
        .size(FONT_SIZE_SMALLER)
        .style(styling::heading)
}

/// A header-bar-height strip holding a panel's header.
fn header_bar<'a>(content: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    container(content)
        .width(Length::Fill)
        .height(HEADER_HEIGHT)
        .align_y(Alignment::Center)
        .into()
}

/// `flush` spans the full sidebar width, between the padded header and content.
/// Whether the adjustments are laid over the preview, without a background,
/// and the file browser and metadata are hidden (the tablet layout), or the
/// three panes are shown.
const OVERLAY_LAYOUT: bool = false;

/// The lines under the histogram, each as a left and a right part: the
/// camera and the lens, then "shutter @ ISO n" and "focal length @ aperture".
/// Each leaves out what the file lacks.
fn shot_summary(raw: &RawFrame) -> [(String, String); 2] {
    let get = |label: &str| {
        raw.metadata
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, v)| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    let join = |parts: [Option<String>; 2]| {
        parts.into_iter().flatten().collect::<Vec<_>>().join(" · ")
    };
    [
        (
            get("Camera").unwrap_or_default(),
            get("Lens").unwrap_or_default(),
        ),
        (
            join([get("ISO").map(|v| format!("ISO {v}")), get("Shutter")]),
            join([get("Focal length"), get("Aperture")]),
        ),
    ]
}

/// Horizontal padding of a sidebar's content. Scrolling content applies it
/// inside the scrollable so the scrollbar sits at the panel's edge.
const PANEL_INSET: Padding = Padding {
    top: 0.0,
    right: SPACING_LARGE,
    bottom: 0.0,
    left: SPACING_LARGE,
};

/// Whether files dropped on the window arrive: winit 0.30 has no drag and
/// drop on Wayland, which it uses whenever `WAYLAND_DISPLAY` is set.
fn drop_supported() -> bool {
    cfg!(any(windows, target_os = "macos")) || std::env::var_os("WAYLAND_DISPLAY").is_none()
}

/// `path` cut from the front, at a folder boundary and marked ".../", to fit
/// `width` pixels (estimated from an average glyph width; the last folder
/// name is always kept).
fn shorten_path(path: &std::path::Path, width: f32) -> String {
    let full = path.to_string_lossy().into_owned();
    let fits = |chars: usize| chars as f32 * FONT_SIZE_STANDARD * 0.58 <= width;
    if fits(full.chars().count()) {
        return full;
    }
    let components: Vec<_> = path
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(name) => Some(name.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    let sep = std::path::MAIN_SEPARATOR;
    let mut shown = components.last().cloned().unwrap_or_default();
    for name in components.iter().rev().skip(1) {
        let longer = format!("{name}{sep}{shown}");
        if !fits(longer.chars().count() + 4) {
            break;
        }
        shown = longer;
    }
    format!("...{sep}{shown}")
}

fn sidebar<'a>(
    header: impl Into<Element<'a, Message>>,
    flush: impl Into<Element<'a, Message>>,
    content: impl Into<Element<'a, Message>>,
) -> container::Container<'a, Message> {
    container(column![
        container(header_bar(header)).padding(PANEL_INSET),
        flush.into(),
        container(content.into()).padding(Padding {
            bottom: SPACING_NORMAL,
            ..Padding::ZERO
        }),
    ])
    .height(Length::Fill)
}

/// A small icon button, or an empty slot of its size.
fn icon_slot<'a>(action: Option<(char, Message, bool)>) -> Element<'a, Message> {
    match action {
        Some((codepoint, message, dim)) => button(center({
            let glyph = icon(codepoint, ICON_SIZE_TINY);
            if dim {
                glyph.style(styling::muted)
            } else {
                glyph
            }
        }))
        .on_press(message)
        .style(styling::toolbar_button)
        .padding(0)
        .width(RESET_SLOT)
        .height(RESET_SLOT)
        .into(),
        None => space().width(RESET_SLOT).height(RESET_SLOT).into(),
    }
}

/// Wheel notches in a scroll event (positive = up).
fn notches(delta: mouse::ScrollDelta) -> f32 {
    match delta {
        mouse::ScrollDelta::Lines { y, .. } => y,
        mouse::ScrollDelta::Pixels { y, .. } => y / PIXELS_PER_NOTCH,
    }
}

/// `content` with a hint that shows when the cursor rests on it.
fn with_tooltip<'a>(
    content: impl Into<Element<'a, Message>>,
    hint: &'a str,
) -> Element<'a, Message> {
    tooltip(
        content,
        container(text(hint).size(FONT_SIZE_SMALLER))
            .padding([3.0, 8.0])
            .style(styling::tooltip),
        tooltip::Position::Bottom,
    )
    .into()
}

/// Which block of the Metadata tab a row belongs in.
fn metadata_group(label: &str) -> &'static str {
    match label {
        "Camera" | "Lens" | "Lens profile" | "Artist" | "Copyright" => "Camera",
        "Taken" | "Focal length" | "Aperture" | "Focus distance" | "Shutter" | "ISO"
        | "Exposure bias" => "Exposure",
        _ => "File",
    }
}

/// A heading over a two-column table of names and values.
fn metadata_table<'a>(group: &'a str, rows: Vec<(&'static str, String)>) -> Element<'a, Message> {
    let table = column(rows.into_iter().map(|(label, value)| {
        row![
            text(label)
                .size(FONT_SIZE_SMALL)
                .style(styling::muted)
                .width(METADATA_LABEL_WIDTH),
            // Paths have no spaces to break at.
            text(value)
                .size(FONT_SIZE_STANDARD)
                .wrapping(text::Wrapping::WordOrGlyph)
                .width(Length::Fill),
        ]
        .spacing(SPACING_NORMAL)
        .into()
    }))
    .spacing(SPACING_SMALL);
    column![heading(group), table]
        .spacing(SPACING_NORMAL)
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_iso_rounds_to_third_stops() {
        for (iso, want) in [
            (100.0, 100.0),
            (126.0, 125.0),
            (252.0, 250.0),
            (635.0, 640.0),
            (3200.0, 3200.0),
            (12800.0 * 1.26, 16000.0),
            (64.0, 64.0),
        ] {
            assert_eq!(nominal_iso(iso), want, "{iso}");
        }
    }

    #[test]
    fn iso_scales_shot_noise_linearly_and_read_noise_squared() {
        let fit = Noise { a: 1e-4, b: 1e-6 };
        assert_eq!(set_noise(fit, NoiseSet::default()), fit);
        let up = noise_at_stops(fit, 1.0);
        assert!((up.a / fit.a - 2.0).abs() < 1e-9 && (up.b / fit.b - 4.0).abs() < 1e-9);
        // never beyond what the networks were trained on
        let far = noise_at_stops(fit, 6.0);
        assert_eq!((far.a, far.b), (64e-4, B_RANGE.1));
        // a or b set in Debug wins over the ISO
        let set = NoiseSet {
            stops: Some(1.0),
            ab: [Some(-3.0), None],
        };
        let n = set_noise(fit, set);
        assert!((n.a - 1e-3).abs() < 1e-12 && (n.b / fit.b - 4.0).abs() < 1e-9);
    }

    /// The window icon is dropped silently if it fails to decode.
    #[test]
    fn icon_decodes() {
        assert!(window::icon::from_file_data(ICON_BYTES, None).is_ok());
    }
}
