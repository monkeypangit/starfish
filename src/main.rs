// Release builds on Windows open no console window next to the app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod agx;
mod develop;
mod export;
mod histogram;
mod lens;
mod library;
mod local;
mod log;
mod preview;
mod processing;
mod raw;
mod source;
mod styling;
mod wheel;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use iced::widget::image::Handle;
use iced::widget::{
    button, canvas, center, checkbox, column, container, mouse_area, pane_grid, responsive,
    pick_list, row, rule, scrollable, shader, slider, space, stack, text, text_input, tooltip,
};
use iced::{
    Alignment, ContentFit, Element, Font, Length, Padding, Size, Subscription, Task,
    keyboard, mouse, task, window,
};

use export::Format;
use histogram::Histogram;
use preview::{Adjustments, Aspect, Crop, Interaction, MID_GRAY, Preview, Rotation, ViewState, Zoom};
use processing::{Engine, Image, Noise};
use lens::Corrections;
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
const FONT_SIZE_CROP: f32 = 14.0;

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
/// Slope of the tone curve at the midpoint (agx::Curve); 1 = no curve.
const CONTRAST_RANGE: std::ops::RangeInclusive<f32> = 1.0..=2.0;
/// Black point slider end, in percent of white (Adjustments::black_point).
const BLACK_POINT_PERCENT: f32 = 5.0;
/// Noise reduction slider position for a newly opened file.
const NOISE_REDUCTION_DEFAULT: f32 = 0.0;
/// The lens corrections a newly opened file gets, where its profile has them
/// and (see [`lens_default`]) its numbers are confirmed.
const LENS_DEFAULT: Corrections = Corrections {
    chromatic: false,
    ..Corrections::ALL
};
/// The lens corrections the app offers. CA is left out for every camera
/// until the demosaic network corrects it: the profiles' CA tables are still
/// read and `LensPass` can still apply them (the `develop_file` test's dump
/// does), for reference when that is built.
const LENS_OFFERED: Corrections = Corrections {
    chromatic: false,
    ..Corrections::ALL
};
/// Height of the panels' header strips.
const HEADER_HEIGHT: f32 = 47.0;
/// Narrowest a thumbnail cell gets; the grid adds columns as the pane widens.
const THUMBNAIL_MIN_WIDTH: f32 = 150.0;
const THUMBNAIL_GAP: f32 = 3.0;
/// Gap between the button border and the thumbnail.
const THUMBNAIL_PADDING: f32 = 4.0;
const PANE_MIN_WIDTH: f32 = 180.0;
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
const CODEPOINT_PHOTO: char = '\u{e410}';
const CODEPOINT_HOURGLASS: char = '\u{e88b}';
const CODEPOINT_ERROR: char = '\u{e001}';
const CODEPOINT_ZOOM_IN: char = '\u{e8ff}';
const CODEPOINT_ROTATE_LEFT: char = '\u{e419}';
const CODEPOINT_ROTATE_RIGHT: char = '\u{e41a}';
const CODEPOINT_ZOOM_OUT: char = '\u{e900}';
const CODEPOINT_FIT_SCREEN: char = '\u{ea10}';
const CODEPOINT_CROP_FREE: char = '\u{e3c2}';
const CODEPOINT_CROP: char = '\u{e3be}';
const CODEPOINT_RESTART: char = '\u{f053}';
const CODEPOINT_CHECK: char = '\u{e5ca}';
const CODEPOINT_LINK: char = '\u{e157}';
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
    Metadata,
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
    Ready(Handle),
}

/// A raw file in the browsed folder.
struct Entry {
    path: PathBuf,
    name: String,
    thumbnail: Thumbnail,
}

/// A collapsible group of controls in the adjustments panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Section {
    WhiteBalance,
    Exposure,
    Tone,
    Color,
    ColorGrading,
    Enhancements,
    LensCorrections,
    Debug,
}

impl Section {
    fn label(self) -> &'static str {
        match self {
            Section::WhiteBalance => "White balance",
            Section::Exposure => "Exposure",
            Section::Tone => "Tone",
            Section::Enhancements => "Enhancements",
            Section::LensCorrections => "Lens corrections",
            Section::Color => "Color",
            Section::ColorGrading => "Color grading",
            Section::Debug => "Debug",
        }
    }
}

struct App {
    panes: pane_grid::State<PaneKind>,
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
    rotation: Rotation,
    /// The export frame against the image.
    crop: Crop,
    /// Crop mode: the image moves, zooms and tilts under a fixed frame.
    cropping: bool,
    preview_size: Size,
    adjustments: Adjustments,
    /// Which colour wheels move together.
    wheels_linked: [bool; 3],
    /// Of what the export of the loaded image would hold.
    histogram: Option<Histogram>,
    /// What the histogram run in flight, or else the last one, was for.
    /// After every update a run starts if this is not what is shown now.
    histogram_for: Option<HistogramKey>,
    /// A histogram run is in flight; when it finishes another starts if the
    /// image or view has moved on, so a drag costs one run at a time with
    /// the latest adjustments instead of one per mouse move.
    histogram_running: bool,
    /// Dragged by the histogram's lower border.
    histogram_ratio: f32,
    /// Noise reduction slider, 0..1; see [`noise_factor`].
    noise_reduction: f32,
    /// The lens corrections to apply, where the raw file's profile has them.
    lens: Corrections,
    /// Debugging: demosaic with the quick binning stage instead of the
    /// neural network.
    quick_demosaic: bool,
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
    /// An export is being written.
    exporting: bool,
}

#[derive(Debug, Clone)]
enum Message {
    OpenFile,
    FileChosen(Option<PathBuf>),
    OpenFolder,
    FolderChosen(Option<PathBuf>),
    Thumbnail(u64, usize, Option<Handle>),
    Previewed(u64, Result<Loaded, String>),
    Developed(u64, Result<Loaded, String>),
    NoiseReduction(f32),
    LensCorrection(lens::Kind, bool),
    QuickDemosaic(bool),
    Redeveloped(u64, Result<Redeveloped, String>),
    HistogramReady(HistogramKey, Result<Option<Histogram>, String>),
    Preview(Interaction),
    ZoomIn,
    ZoomOut,
    ZoomFit,
    ZoomActual,
    RotateLeft,
    RotateRight,
    ToggleCrop,
    CropReset,
    CropAspect(Aspect),
    /// Percent of white (Adjustments::black_point × 100).
    BlackPoint(f32),
    Exposure(f32),
    Contrast(f32),
    LocalContrast(f32),
    /// Display-encoded, 0..1 (agx::encode).
    Midpoint(f32),
    Saturation(f32),
    Temperature(f32),
    Tint(f32),
    ResetAdjustments,
    Tab(SidebarTab),
    ResetControl(Control),
    /// Wheel index (shadows, midtones, highlights) and puck position.
    Grading(usize, [f32; 2]),
    /// Takes a wheel out of the linked group.
    UnlinkWheel(usize),
    /// Joins a wheel to the linked pair, taking their value.
    LinkWheel(usize),
    /// Range index (shadows, midtones, highlights) and brightness, −1..1.
    Level(usize, f32),
    HistogramRatio(f32),
    PaneResized(pane_grid::ResizeEvent),
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
    Export,
    ExportPath(Option<PathBuf>),
    Exported(Result<PathBuf, String>),
}

/// One adjustment control, for its own reset button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Control {
    Noise,
    Temperature,
    Tint,
    BlackPoint,
    Exposure,
    Saturation,
    Contrast,
    Midpoint,
    LocalContrast,
    /// Range index (shadows, midtones, highlights).
    Level(usize),
    Wheel(usize),
}

impl Control {
    /// Whether `self` differs from its default.
    fn changed(self, app: &App) -> bool {
        let (a, d) = (&app.adjustments, Adjustments::default());
        match self {
            Control::Noise => app.noise_reduction != NOISE_REDUCTION_DEFAULT,
            Control::Temperature => a.temperature != d.temperature,
            Control::Tint => a.tint != d.tint,
            Control::BlackPoint => a.black_point != d.black_point,
            Control::Exposure => a.exposure != d.exposure,
            Control::Saturation => a.saturation != d.saturation,
            Control::Contrast => a.contrast != d.contrast,
            Control::Midpoint => a.midpoint != d.midpoint,
            Control::LocalContrast => a.local_contrast != d.local_contrast,
            Control::Level(i) => a.levels[i] != d.levels[i],
            Control::Wheel(i) => a.grading[i] != d.grading[i],
        }
    }

    /// The slider behind `self`. Levels and wheels have no number to type.
    fn slider(self, app: &App) -> Option<SliderSpec> {
        let a = &app.adjustments;
        let spec = |x, range, message| SliderSpec {
            x,
            range,
            step: 0.01,
            message,
        };
        Some(match self {
            Control::Noise => spec(app.noise_reduction, 0.0..=1.0, Message::NoiseReduction),
            Control::Temperature => spec(a.temperature, -1.0..=1.0, Message::Temperature),
            Control::Tint => spec(a.tint, -1.0..=1.0, Message::Tint),
            Control::BlackPoint => spec(
                a.black_point * 100.0,
                -BLACK_POINT_PERCENT..=BLACK_POINT_PERCENT,
                Message::BlackPoint,
            ),
            Control::Exposure => spec(a.exposure, -5.0..=5.0, Message::Exposure),
            Control::Saturation => spec(a.saturation, 0.0..=2.0, Message::Saturation),
            Control::Contrast => spec(a.contrast, CONTRAST_RANGE, Message::Contrast),
            Control::LocalContrast => spec(a.local_contrast, 0.0..=1.0, Message::LocalContrast),
            Control::Midpoint => SliderSpec {
                step: 0.005,
                ..spec(agx::encode(a.midpoint), 0.0..=1.0, Message::Midpoint)
            },
            Control::Level(_) | Control::Wheel(_) => return None,
        })
    }

    /// The number the control's value label shows, without its unit; `None`
    /// if there is nothing to edit (no image, or no noise estimate).
    fn shown(self, app: &App) -> Option<f32> {
        let x = self.slider(app)?.x;
        app.loaded.as_ref()?;
        Some(match self {
            Control::Noise if app.wanted_noise().is_none() => return None,
            Control::Noise | Control::Tint | Control::Saturation | Control::LocalContrast => {
                x * 100.0
            }
            Control::Temperature => match app.white() {
                Some(white) => white.kelvin(x),
                None => x * 100.0,
            },
            Control::Midpoint => app.adjustments.midpoint * 100.0,
            _ => x,
        })
    }

    /// The slider position that makes the label show `shown`.
    fn position_for(self, app: &App, shown: f32) -> f32 {
        match self {
            Control::Noise | Control::Tint | Control::Saturation | Control::LocalContrast => {
                shown / 100.0
            }
            Control::Temperature => match app.white() {
                Some(white) => white.temperature_for(shown.max(1.0)),
                None => shown / 100.0,
            },
            Control::Midpoint => agx::encode((shown / 100.0).clamp(0.001, 1.0)),
            _ => shown,
        }
    }

    /// Decimals the label shows.
    fn decimals(self) -> usize {
        match self {
            Control::Midpoint => 1,
            Control::BlackPoint | Control::Exposure | Control::Contrast => 2,
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

/// What a numeric control's slider is: position, range, step and message.
struct SliderSpec {
    x: f32,
    range: std::ops::RangeInclusive<f32>,
    step: f32,
    message: fn(f32) -> Message,
}

impl App {
    fn new() -> (Self, Task<Message>) {
        let mut app = Self {
            panes: initial_panes(),
            folder: None,
            entries: Vec::new(),
            folder_generation: 0,
            thumbnail_task: None,
            current: None,
            loaded: None,
            status: Status::default(),
            export_format: Format::JpegL,
            exporting: false,
            view: ViewState::default(),
            rotation: Rotation::default(),
            crop: Crop::default(),
            cropping: false,
            preview_size: Size::ZERO,
            adjustments: Adjustments::default(),
            wheels_linked: [false; 3],
            histogram: None,
            histogram_for: None,
            histogram_running: false,
            histogram_ratio: HISTOGRAM_RATIO,
            noise_reduction: NOISE_REDUCTION_DEFAULT,
            lens: LENS_DEFAULT,
            quick_demosaic: false,
            redeveloping: None,
            generation: 0,
            cancel: Arc::new(AtomicBool::new(false)),
            tab: SidebarTab::Adjustments,
            collapsed: HashSet::new(),
            editing: None,
            ctrl: false,
            drop_hover: false,
        };
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
        (app, task)
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
            iced::event::listen_with(|event, _status, _window| match event {
                Event::Window(window::Event::FileHovered(_)) => Some(Message::DropHover(true)),
                Event::Window(window::Event::FilesHoveredLeft) => Some(Message::DropHover(false)),
                Event::Window(window::Event::FileDropped(path)) => Some(Message::Dropped(path)),
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

    /// The image's size after rotation, before cropping.
    fn rotated_size(&self) -> Option<Size> {
        self.loaded
            .as_ref()
            .map(|l| {
                self.rotation
                    .size(Size::new(l.image.width as f32, l.image.height as f32))
            })
            .filter(|_| self.preview_size.width > 0.0 && self.preview_size.height > 0.0)
    }

    /// The shown image's size: what is inside the crop frame, after rotation.
    fn image_size(&self) -> Option<Size> {
        self.rotated_size().map(|s| self.crop.frame_size(s))
    }

    /// Logical pixels per image pixel in crop mode.
    fn crop_scale(&self) -> Option<f32> {
        self.rotated_size().map(|s| self.crop.scale(self.preview_size, s))
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        let task = self.handle(message);
        Task::batch([task, self.refresh_histogram()])
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
                return Task::perform(
                    dialog.pick_folder(),
                    |handle| Message::FolderChosen(handle.map(|h| h.path().to_path_buf())),
                );
            }
            Message::FolderChosen(None) => {}
            Message::FolderChosen(Some(folder)) => return self.open_folder(folder),
            Message::Thumbnail(generation, index, thumbnail) => {
                if generation == self.folder_generation {
                    if let Some(entry) = self.entries.get_mut(index) {
                        entry.thumbnail = match thumbnail {
                            Some(handle) => Thumbnail::Ready(handle),
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
                // Drop the old image right away.
                self.loaded = None;
                self.histogram = None;
                self.histogram_for = None;
                self.view = ViewState::default();
                self.rotation = Rotation::default();
                self.crop = Crop::default();
                self.cropping = false;
                self.noise_reduction = NOISE_REDUCTION_DEFAULT;
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
                self.lens = lens_default(&raw);
                let decode_time = loaded.decode_time;
                self.loaded = Some(loaded);
                return Task::perform(
                    develop(
                        raw,
                        decode_time,
                        self.lens.and(LENS_OFFERED).and(Corrections::CHROMATIC),
                        self.quick_demosaic,
                        self.cancel.clone(),
                    ),
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
            Message::NoiseReduction(value) => {
                self.noise_reduction = value;
                return self.redevelop();
            }
            Message::LensCorrection(kind, on) => {
                self.lens.set(kind, on);
                // Distortion and vignetting are drawn; CA re-demosaics.
                return self.redevelop();
            }
            Message::QuickDemosaic(on) => {
                self.quick_demosaic = on;
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
                }
            }
            Message::Preview(Interaction::Pan(delta)) if self.cropping => {
                if let Some(scale) = self.crop_scale() {
                    self.crop = self.crop.pan_by(delta, scale);
                }
            }
            Message::Preview(Interaction::Rotate(delta)) => {
                self.crop = self.crop.tilt_by(delta);
            }
            Message::ToggleCrop => {
                self.cropping = !self.cropping;
                // The view is in the frame's space, which the crop changed.
                self.view = ViewState::default();
            }
            Message::CropReset => self.crop = Crop::default(),
            Message::CropAspect(aspect) => self.crop.aspect = aspect,
            Message::Preview(Interaction::Zoom { factor, anchor }) => {
                if let Some(image) = self.image_size() {
                    self.view = self.view.zoom_by(factor, anchor, self.preview_size, image);
                }
            }
            Message::Preview(Interaction::Pan(delta)) => {
                if let Some(image) = self.image_size() {
                    self.view = self.view.pan_by(delta, self.preview_size, image);
                }
            }
            Message::ZoomIn | Message::ZoomOut => {
                if let Some(image) = self.image_size() {
                    let zoom_in = matches!(message, Message::ZoomIn);
                    self.view = self.view.zoom_step(zoom_in, self.preview_size, image);
                }
            }
            Message::ZoomFit => self.view = ViewState::default(),
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
            }
            Message::ZoomActual => {
                if let Some(image) = self.image_size() {
                    self.view = ViewState::actual_size(image);
                }
            }
            Message::BlackPoint(percent) => {
                self.adjustments.black_point = percent / 100.0;
            }
            Message::Exposure(value) => {
                self.adjustments.exposure = value;
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
            Message::Contrast(value) => {
                self.adjustments.contrast = value;
            }
            Message::LocalContrast(value) => {
                self.adjustments.local_contrast = value;
            }
            Message::Midpoint(encoded) => {
                self.adjustments.midpoint = agx::decode(encoded);
            }
            Message::Grading(range, position) => {
                // Moving a wheel that is at its default links it with every
                // other wheel at its default; linked wheels follow.
                let default = Adjustments::default().grading;
                let grading = self.adjustments.grading;
                if grading[range] == default[range] {
                    let at_default: [bool; 3] = std::array::from_fn(|i| grading[i] == default[i]);
                    // Alone at its default, the wheel leaves existing links alone.
                    if at_default.iter().filter(|&&d| d).count() >= 2 {
                        self.wheels_linked = at_default;
                    } else {
                        self.wheels_linked[range] = false;
                    }
                }
                let linked = self.wheels_linked;
                self.adjustments.grading[range] = position;
                if linked[range] {
                    for i in (0..3).filter(|&i| linked[i]) {
                        self.adjustments.grading[i] = position;
                    }
                }
            }
            Message::Level(range, value) => {
                self.adjustments.levels[range] = value;
            }
            Message::HistogramReady((image_id, _), result) => {
                self.histogram_running = false;
                let current = self.loaded.as_ref().map(|l| l.image.id);
                match result {
                    Ok(Some(histogram)) if current == Some(image_id) => self.histogram = Some(histogram),
                    Ok(Some(_)) => {}
                    // The preview has not been drawn yet: try again.
                    Ok(None) => self.histogram_for = None,
                    // Then the preview cannot show the image either.
                    Err(error) if current == Some(image_id) && error.starts_with("Out of GPU memory") => {
                        self.fail(error)
                    }
                    Err(error) => log::write(&format!("Error: {error}")),
                }
            }
            Message::HistogramRatio(ratio) => {
                self.histogram_ratio = ratio.clamp(HISTOGRAM_MIN_RATIO, HISTOGRAM_MAX_RATIO);
            }
            Message::ResetControl(control) => {
                let d = Adjustments::default();
                let a = &mut self.adjustments;
                match control {
                    Control::Noise => {
                        self.noise_reduction = NOISE_REDUCTION_DEFAULT;
                        return self.redevelop();
                    }
                    Control::Temperature => a.temperature = d.temperature,
                    Control::Tint => a.tint = d.tint,
                    Control::BlackPoint => a.black_point = d.black_point,
                    Control::Exposure => a.exposure = d.exposure,
                    Control::Saturation => a.saturation = d.saturation,
                    Control::Contrast => a.contrast = d.contrast,
                    Control::Midpoint => a.midpoint = d.midpoint,
                    Control::LocalContrast => a.local_contrast = d.local_contrast,
                    Control::Level(i) => a.levels[i] = d.levels[i],
                    Control::Wheel(i) => {
                        a.grading[i] = d.grading[i];
                        for j in (0..3).filter(|&j| self.wheels_linked[i] && self.wheels_linked[j])
                        {
                            a.grading[j] = d.grading[j];
                        }
                    }
                }
            }
            Message::Tab(tab) => self.tab = tab,
            Message::LinkWheel(index) => {
                if let Some(other) = (0..3).find(|&i| self.wheels_linked[i]) {
                    self.adjustments.grading[index] = self.adjustments.grading[other];
                    self.wheels_linked[index] = true;
                }
            }
            Message::UnlinkWheel(index) => {
                self.wheels_linked[index] = false;
            }
            Message::ResetAdjustments => {
                self.adjustments = Adjustments::default();
                self.wheels_linked = [false; 3];
            }
            Message::PaneResized(pane_grid::ResizeEvent { split, ratio }) => {
                self.panes.resize(split, ratio);
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
                self.editing = Some((control, format!("{shown:.*}", control.decimals())));
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
            Message::Export => {
                let Some(loaded) = &self.loaded else {
                    return Task::none();
                };
                let name = std::path::Path::new(&loaded.image.name)
                    .with_extension(self.export_format.extension());
                let mut dialog = rfd::AsyncFileDialog::new()
                    .set_title("Export image")
                    .set_file_name(name.to_string_lossy())
                    .add_filter(self.export_format.to_string(), &[self.export_format.extension()]);
                if let Some(dir) = self.current.as_ref().and_then(|p| p.parent()) {
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
                self.exporting = true;
                let (image, adjustments, rotation, crop, lens, format) = (
                    loaded.image.clone(),
                    self.adjustments,
                    self.rotation,
                    self.crop,
                    self.drawn_lens(),
                    self.export_format,
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
                            export::export(&device, &queue, &image, view, format, &path).map(|()| path)
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
                name: path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
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

    /// The noise the slider asks for, if the demosaic stage denoises.
    fn wanted_noise(&self) -> Option<Noise> {
        if self.quick_demosaic {
            return None;
        }
        let estimate = self.loaded.as_ref()?.estimate?;
        Some(estimate.scaled(noise_factor(self.noise_reduction)))
    }

    /// The lens corrections switched on that the file's profile has and the
    /// app offers.
    fn available_lens(&self) -> Corrections {
        let available = self.loaded.as_ref().and_then(|l| l.raw.lens.as_ref());
        available.map_or(Corrections::NONE, |l| self.lens.and(l.available()).and(LENS_OFFERED))
    }

    /// The lens corrections the demosaic stage should apply: CA (none while
    /// [`LENS_OFFERED`] leaves it out).
    fn wanted_lens(&self) -> Corrections {
        self.available_lens().and(Corrections::CHROMATIC)
    }

    /// The lens corrections applied where the image is drawn (preview and
    /// export): distortion and vignetting, so toggling them is instant.
    fn drawn_lens(&self) -> Corrections {
        self.available_lens().without(Corrections::CHROMATIC)
    }

    /// Re-demosaics with the slider's noise and the lens checkbox unless
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
            lens: self.wanted_lens(),
            quick: self.quick_demosaic,
        };
        if let Some(run) = &self.redeveloping {
            if run.settings != settings {
                run.cancel.store(true, Ordering::Relaxed);
            }
            return Task::none();
        }
        let (noise, lens) = (settings.noise, settings.lens);
        if loaded.preview
            || (loaded.image.noise == noise
                && loaded.image.lens == lens
                && loaded.image.quick == self.quick_demosaic)
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
                self.noise_reduction,
                lens,
                self.quick_demosaic,
                cancel,
            ),
            move |result| Message::Redeveloped(generation, result),
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
        let panes = pane_grid(&self.panes, |_pane, kind, _maximized| {
            pane_grid::Content::new(match kind {
                PaneKind::FileBrowser => self.view_file_browser(),
                PaneKind::Preview => self.view_center(),
                PaneKind::Tools => match self.tab {
                    SidebarTab::Adjustments => self.view_tools(),
                    SidebarTab::Metadata => self.view_metadata(),
                },
            })
        })
        .spacing(1)
        .min_size(PANE_MIN_WIDTH)
        .on_resize(PANE_RESIZE_LEEWAY, Message::PaneResized)
        .style(styling::panes);

        // The 1 px gaps between panes show this background as divider lines.
        let content = container(panes).style(styling::pane_dividers);

        content.into()
    }

    // Left pane.

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
        let title = responsive(move |size| match &folder {
            Some(folder) => text(shorten_path(folder, size.width))
                .size(FONT_SIZE_STANDARD)
                .wrapping(text::Wrapping::None)
                .into(),
            None => Element::from(text("Library").size(FONT_SIZE_STANDARD)),
        });
        let header = row![
            container(title)
                .width(Length::Fill)
                .height(FONT_SIZE_STANDARD * 1.5)
                .align_y(Alignment::Center),
            header_button(CODEPOINT_FOLDER_OPEN, "Open folder…", Message::OpenFolder),
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
                let picture: Element<'_, Message> = match &entry.thumbnail {
                    Thumbnail::Ready(handle) => iced::widget::image(handle.clone())
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .content_fit(ContentFit::Contain)
                        .into(),
                    Thumbnail::Missing => {
                        center(icon(CODEPOINT_PHOTO, ICON_SIZE_TOOLBAR).style(styling::muted))
                            .into()
                    }
                    Thumbnail::Loading => space().into(),
                };
                let picture_style: fn(&iced::Theme) -> container::Style = match entry.thumbnail {
                    Thumbnail::Loading => styling::thumbnail_loading,
                    _ => styling::thumbnail_picture,
                };
                let side = cell - 2.0 * THUMBNAIL_PADDING;
                let picture = container(picture)
                    .width(side)
                    .height(side)
                    .clip(true)
                    .style(picture_style);
                let name = text(&entry.name)
                    .size(FONT_SIZE_SMALLER)
                    .wrapping(text::Wrapping::None);
                let content = column![picture, name]
                    .spacing(SPACING_SMALL)
                    .width(Length::Fill)
                    .align_x(Alignment::Center);
                button(content)
                    .on_press(Message::FileChosen(Some(entry.path.clone())))
                    .style(styling::thumbnail_button(
                        self.current.as_ref() == Some(&entry.path),
                    ))
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
            .into()
    }

    // Center pane.

    fn view_center(&self) -> Element<'_, Message> {
        column![
            self.view_toolbar(),
            rule::horizontal(1).style(styling::divider),
            self.view_preview(),
        ]
        .into()
    }

    fn view_toolbar(&self) -> Element<'_, Message> {
        // Only what identifies the photo; the details are in the Metadata tab
        // and progress is shown over the picture.
        let (title, subtitle): (Element<'_, Message>, Option<Element<'_, Message>>) =
            match &self.loaded {
                Some(loaded) => {
                    let image = &loaded.image;
                    (
                        text(&image.name)
                            .font(BOLD_FONT)
                            .size(FONT_SIZE_SMALL)
                            .into(),
                        Some(
                            text(format!(
                                "{} · {}×{}",
                                image.camera, image.width, image.height
                            ))
                            .size(FONT_SIZE_SMALL)
                            .style(styling::muted)
                            .into(),
                        ),
                    )
                }
                None => (
                    text(APP_TITLE)
                        .font(BOLD_FONT)
                        .size(FONT_SIZE_SMALL)
                        .into(),
                    None,
                ),
            };

        let has_image = self.loaded.is_some();
        let toolbar_button = |codepoint, message: Message| {
            button(icon(codepoint, ICON_SIZE_TOOLBAR))
                .on_press_maybe(has_image.then_some(message))
                .style(styling::toolbar_button)
                .padding(SPACING_SMALL)
        };

        let zoom_label = match (self.view.zoom, self.image_size()) {
            _ if self.cropping => String::from("Crop"),
            (Zoom::Fit, Some(image)) => format!(
                "Fit · {:.0}%",
                ViewState::fit_scale(self.preview_size, image) * 100.0
            ),
            (Zoom::Scale(scale), _) => format!("{:.0}%", scale * 100.0),
            (Zoom::Fit, None) => String::from("Fit"),
        };

        // Zoom and pan are the crop's in crop mode.
        let view_button = |codepoint, message: Message| {
            button(icon(codepoint, ICON_SIZE_TOOLBAR))
                .on_press_maybe((has_image && !self.cropping).then_some(message))
                .style(styling::toolbar_button)
                .padding(SPACING_SMALL)
        };
        let crop_button = button(icon(CODEPOINT_CROP, ICON_SIZE_TOOLBAR))
            .on_press_maybe(has_image.then_some(Message::ToggleCrop))
            .style(styling::segment(self.cropping))
            .padding(SPACING_SMALL);

        let view_controls = row![
            view_button(CODEPOINT_ZOOM_OUT, Message::ZoomOut),
            container(text(zoom_label).size(FONT_SIZE_SMALL))
                .width(80)
                .align_x(Alignment::Center),
            view_button(CODEPOINT_ZOOM_IN, Message::ZoomIn),
            view_button(CODEPOINT_FIT_SCREEN, Message::ZoomFit),
            view_button(CODEPOINT_CROP_FREE, Message::ZoomActual),
            space().width(SPACING_NORMAL),
            toolbar_button(CODEPOINT_ROTATE_LEFT, Message::RotateLeft),
            toolbar_button(CODEPOINT_ROTATE_RIGHT, Message::RotateRight),
            crop_button,
        ]
        .align_y(Alignment::Center);

        let export_controls = row![
            pick_list(Format::ALL, Some(self.export_format), Message::ExportFormat)
                .text_size(FONT_SIZE_SMALL),
            button(icon(CODEPOINT_DOWNLOAD, ICON_SIZE_TOOLBAR))
                .on_press_maybe((has_image && !self.exporting).then_some(Message::Export))
                .style(styling::toolbar_button)
                .padding(SPACING_SMALL),
        ]
        .align_y(Alignment::Center);

        container(header_bar(
            // Equal-width sides keep the view controls centred in the bar.
            row![
                container(column![title].extend(subtitle)).width(Length::Fill),
                view_controls,
                container(export_controls)
                    .width(Length::Fill)
                    .align_x(Alignment::End),
            ]
                .spacing(SPACING_LARGE)
                .align_y(Alignment::Center),
        ))
        .padding([0.0, SPACING_LARGE])
        .style(styling::toolbar)
        .into()
    }

    fn view_preview(&self) -> Element<'_, Message> {
        let picture = shader(Preview {
            image: self.loaded.as_ref().map(|l| &l.image),
            view: self.view,
            adjustments: self.adjustments,
            rotation: self.rotation,
            crop: self.crop,
            cropping: self.cropping,
            lens: self.drawn_lens(),
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
            Status::Idle if self.loaded.is_none() => Some(
                center(
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
                ),
            ),
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

        let crop_bar = self.view_crop_bar();

        container(
            stack![picture]
                .extend(crop_bar.map(Element::from))
                .extend(overlay.map(Element::from))
                .extend(busy.map(Element::from))
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

    /// Crop mode: a small bar under the frame with the tilt, zoom, the frame's
    /// top left corner and size (in pixels of the unrotated image), reset and
    /// done.
    fn view_crop_bar(&self) -> Option<Element<'_, Message>> {
        let loaded = self.loaded.as_ref().filter(|_| self.cropping)?;
        let source = Size::new(loaded.image.width as f32, loaded.image.height as f32);
        let rotated = self.rotation.size(source);
        let frame = self.crop.frame_size(rotated);
        let top_left = self
            .rotation
            .unapply(self.crop.to_image(iced::Point::ORIGIN, rotated), source);
        let size = self.rotation.size(frame);

        // The icons are as large as the header's reset-all icon, the text as
        // the library's folder path.
        let glyph = ICON_SIZE_TOOLBAR - 4.0;
        let info = |label: String| {
            text(label)
                .font(MONO_FONT)
                .size(FONT_SIZE_CROP)
                .color(styling::CROP_TEXT)
        };
        let action = |codepoint, message: Option<Message>| {
            button(icon(codepoint, glyph))
                .on_press_maybe(message)
                .style(styling::crop_button)
                .padding(4)
        };
        let aspect = pick_list(Aspect::ALL, Some(self.crop.aspect), Message::CropAspect)
            .font(MONO_FONT)
            .text_size(FONT_SIZE_CROP)
            .padding([0.0, SPACING_SMALL])
            .style(styling::crop_pick_list);
        let bar = row![
            aspect,
            info(format!("{:.1}°", self.crop.angle.to_degrees())),
            info(format!("{:.0}%", self.crop.zoom * 100.0)),
            info(format!("{:.1}, {:.1}", top_left.x, top_left.y)),
            info(format!("{:.1} × {:.1}", size.width, size.height)),
            action(
                CODEPOINT_RESTART,
                (self.crop != Crop::default()).then_some(Message::CropReset)
            ),
            action(CODEPOINT_CHECK, Some(Message::ToggleCrop)),
        ]
        .spacing(SPACING_LARGE)
        .align_y(Alignment::Center);

        // The frame's double line: black outside, white just inside.
        let bar = container(container(bar).padding([8.0, 2.0 * SPACING_NORMAL]).style(styling::crop_bar_inner))
            .padding(styling::CROP_LINE)
            .style(styling::crop_bar_outer);

        let below = {
            let rect = preview::frame_rect(self.preview_size, self.crop.frame_aspect(rotated));
            rect.y + rect.height
        };
        Some(
            container(bar)
                .width(Length::Fill)
                .height(Length::Fill)
                .align_x(Alignment::Center)
                .align_y(Alignment::Start)
                .padding(Padding::default().top(below + 2.0 * SPACING_NORMAL))
                .into(),
        )
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
                .padding([4.0, 10.0])
        };
        container(
            row![
                tab("Adjustments", SidebarTab::Adjustments),
                tab("Metadata", SidebarTab::Metadata)
            ]
            .spacing(2),
        )
        .padding(2)
        .style(styling::segmented)
        .into()
    }

    /// The right panel's header: the tabs and `extra` controls.
    fn panel_header<'a>(&'a self, extra: Option<Element<'a, Message>>) -> Element<'a, Message> {
        row![self.tab_switcher(), space::horizontal()]
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
                scrollable(container(rows).padding(PANEL_INSET)).width(Length::Fill)
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
                text(section.label()).font(BOLD_FONT).size(FONT_SIZE_SMALLER),
            ]
            .spacing(SPACING_SMALL)
            .align_y(Alignment::Center),
        )
        .on_press(Message::ToggleSection(section))
        .style(styling::section_header)
        .padding([2.0, 0.0])
        .width(Length::Fill);

        let mut panel = column![header].spacing(SPACING_NORMAL);
        if !collapsed {
            panel = panel.push(body);
        }
        panel.into()
    }

    fn view_tools(&self) -> Element<'_, Message> {
        let adjustments = self.adjustments;

        let noise = self.adjustment(
            "Noise reduction",
            Control::Noise,
            match self.wanted_noise() {
                Some(_) => format!("{:.0}%", self.noise_reduction * 100.0),
                None => String::from("—"),
            },
            slider(0.0..=1.0, self.noise_reduction, Message::NoiseReduction)
                .step(0.01_f32)
                .default(NOISE_REDUCTION_DEFAULT),
        );

        let temperature = self.adjustment_with(
            "Temperature",
            Control::Temperature,
            match self.white() {
                Some(white) => format!("{:.0} K", white.kelvin(adjustments.temperature)),
                None => format!("{:+.0}", adjustments.temperature * 100.0),
            },
            centered_slider(1.0, adjustments.temperature, Message::Temperature),
        );
        let tint = self.adjustment_with(
            "Tint",
            Control::Tint,
            format!("{:+.0}", adjustments.tint * 100.0),
            centered_slider(1.0, adjustments.tint, Message::Tint),
        );
        let black_point = self.adjustment_with(
            "Black point",
            Control::BlackPoint,
            format!("{:+.2}%", adjustments.black_point * 100.0),
            centered_slider(
                BLACK_POINT_PERCENT,
                adjustments.black_point * 100.0,
                Message::BlackPoint,
            ),
        );
        let exposure = self.adjustment_with(
            "Exposure",
            Control::Exposure,
            format!("{:+.2} EV", adjustments.exposure),
            centered_slider(5.0, adjustments.exposure, Message::Exposure),
        );
        let saturation = self.adjustment(
            "Saturation",
            Control::Saturation,
            format!("{:.0}%", adjustments.saturation * 100.0),
            slider(0.0..=2.0, adjustments.saturation, Message::Saturation).step(0.01_f32),
        );
        let contrast = self.adjustment(
            "Contrast",
            Control::Contrast,
            format!("{:.2}", adjustments.contrast),
            slider(CONTRAST_RANGE, adjustments.contrast, Message::Contrast)
                .step(0.01_f32)
                .default(agx::DEFAULT_CONTRAST),
        );
        // Along the curve's own axis, so the crossover in the graph follows
        // the handle; the label is linear.
        let midpoint = self.adjustment(
            "Midpoint",
            Control::Midpoint,
            format!("{:.1}%", adjustments.midpoint * 100.0),
            slider(
                0.0..=1.0,
                agx::encode(adjustments.midpoint),
                Message::Midpoint,
            )
            .step(0.005_f32)
            .default(agx::encode(MID_GRAY)),
        );

        let local_contrast = self.adjustment(
            "Local contrast",
            Control::LocalContrast,
            format!("{:.0}", adjustments.local_contrast * 100.0),
            slider(0.0..=1.0, adjustments.local_contrast, Message::LocalContrast).step(0.01_f32),
        );

        // One checkbox per correction the file's lens profile has and the
        // app offers.
        let lens_corrections = self
            .loaded
            .as_ref()
            .and_then(|l| l.raw.lens.as_ref())
            .map(|profile| profile.available().and(LENS_OFFERED))
            .filter(|available| available.any())
            .map(|available| {
                lens::Kind::ALL
                    .into_iter()
                    .filter(|&kind| available.get(kind))
                    .fold(column![], |panel, kind| {
                        panel.push(
                            checkbox(self.lens.get(kind))
                                .label(kind.label())
                                .on_toggle(move |on| Message::LensCorrection(kind, on))
                                .size(16)
                                .text_size(FONT_SIZE_STANDARD),
                        )
                    })
                    .spacing(SPACING_NORMAL)
            });
        let debug = checkbox(self.quick_demosaic)
            .label("Quick demosaic (no neural network)")
            .on_toggle(Message::QuickDemosaic)
            .size(16)
            .text_size(FONT_SIZE_STANDARD);
        let enhancements = column![noise, local_contrast].spacing(SPACING_NORMAL);

        // The midtones slider is hidden; its level stays at 0.
        let levels = row([(0, "Shadows"), (2, "Highlights")]
            .into_iter()
            .map(|(index, label)| {
                column![
                    label_with_reset(label, self.reset_for(Control::Level(index))),
                    centered_slider(1.0, adjustments.levels[index], move |value| Message::Level(
                        index, value
                    )),
                ]
                .spacing(SPACING_SMALL)
                .width(Length::Fill)
                .into()
            }))
        .spacing(SPACING_NORMAL);

        // Linked wheels move together; with fewer than two there is no link.
        let mut linked = self.wheels_linked;
        if linked.iter().filter(|&&l| l).count() < 2 {
            linked = [false; 3];
        }
        let default = Adjustments::default().grading;
        let wheels = row(["Shadows", "Midtones", "Highlights"]
            .into_iter()
            .enumerate()
            .map(|(index, label)| {
                let wheel = canvas(wheel::View {
                    position: adjustments.grading[index],
                    index,
                    on_change: Message::Grading,
                });
                let others_linked = !linked[index] && linked.iter().filter(|&&l| l).count() == 2;
                let action = if linked[index] {
                    Some((CODEPOINT_LINK, Message::UnlinkWheel(index), false))
                } else if others_linked && adjustments.grading[index] == default[index] {
                    // Reset next to a linked pair: offer to join it.
                    Some((CODEPOINT_LINK, Message::LinkWheel(index), true))
                } else {
                    self.reset_for(Control::Wheel(index))
                        .map(|message| (CODEPOINT_RESTART, message, false))
                };
                column![
                    label_with_button(label, action),
                    histogram::Aspect::new(wheel, 1.0),
                ]
                .spacing(SPACING_SMALL)
                .width(Length::Fill)
                .into()
            }))
        .spacing(SPACING_SMALL);

        let histogram = canvas(histogram::View {
            histogram: self.histogram.as_ref(),
            curve: agx::Curve::new(&adjustments),
            on_resize: Message::HistogramRatio,
        });
        let histogram = histogram::Aspect::new(histogram, self.histogram_ratio);

        let reset = with_tooltip(
            button(icon(CODEPOINT_RESTART, ICON_SIZE_TOOLBAR - 4.0))
                .on_press_maybe(
                    (adjustments != Adjustments::default()).then_some(Message::ResetAdjustments),
                )
                .style(styling::toolbar_button)
                .padding(4),
            "Reset all adjustments",
        );

                sidebar(
            self.panel_header(Some(reset)),
            histogram,
            scrollable(
                container(
                    column![
                        space().height(SPACING_SMALL),
                        self.section(
                            Section::Exposure,
                            column![black_point, exposure, levels].spacing(SPACING_NORMAL),
                        ),
                        self.section(
                            Section::Tone,
                            column![midpoint, contrast].spacing(SPACING_NORMAL),
                        ),
                        self.section(
                            Section::WhiteBalance,
                            column![temperature, tint].spacing(SPACING_NORMAL),
                        ),
                        self.section(Section::Color, saturation),
                        self.section(Section::Enhancements, enhancements),
                        lens_corrections
                            .map(|body| self.section(Section::LensCorrections, body)),
                        self.section(Section::ColorGrading, wheels),
                        self.section(Section::Debug, debug),
                        space().height(SPACING_SMALL),
                    ]
                    .spacing(SPACING_LARGE),
                )
                .padding(PANEL_INSET),
            )
            .width(Length::Fill),
        )
        .style(styling::sidebar)
        .into()
    }

    /// An adjustment with a plain slider; see [`Self::adjustment_with`].
    fn adjustment<'a>(
        &self,
        label: &'a str,
        control: Control,
        value: String,
        slider: iced::widget::Slider<'a, f32, Message>,
    ) -> Element<'a, Message> {
        self.adjustment_with(label, control, value, slider.style(styling::adjustment_slider).into())
    }

    /// A control's name with its reset button, its value (click it to type
    /// a number over it) and the slider below. With Ctrl held the wheel
    /// nudges it.
    fn adjustment_with<'a>(
        &self,
        label: &'a str,
        control: Control,
        value: String,
        input: Element<'a, Message>,
    ) -> Element<'a, Message> {
        let value: Element<'a, Message> = match &self.editing {
            Some((editing, typed)) if *editing == control => text_input("", typed)
                .id(VALUE_INPUT_ID)
                .on_input(Message::EditInput)
                .on_submit(Message::EditCommit)
                .size(FONT_SIZE_SMALL)
                .padding([1.0, 4.0])
                .width(VALUE_INPUT_WIDTH)
                .align_x(Alignment::End)
                .style(styling::value_input)
                .into(),
            _ => button(text(value).size(FONT_SIZE_SMALL))
                .on_press_maybe(control.shown(self).map(|_| Message::EditValue(control)))
                .style(styling::value_button)
                .padding([1.0, 4.0])
                .into(),
        };
        let body = column![
            row![
                label_with_reset(label, self.reset_for(control)),
                space::horizontal(),
                value,
            ]
            .spacing(SPACING_NORMAL)
            .align_y(Alignment::Center),
            input,
        ]
        .spacing(SPACING_SMALL);

        // Always wrapped, so the widget tree does not change (and a field
        // being typed in keeps its focus) when Ctrl goes down.
        let area = mouse_area(body);
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

/// The lens corrections `raw` starts with: [`LENS_DEFAULT`] without the ones
/// its profile marks unverified.
fn lens_default(raw: &RawFrame) -> Corrections {
    raw.lens
        .as_ref()
        .map_or(LENS_DEFAULT, |lens| LENS_DEFAULT.without(lens.unverified()))
}

/// Decodes and demosaics off the UI thread.
/// Decodes `path` and bins it into a quick low-quality image.
async fn preview(path: PathBuf) -> Result<Loaded, String> {
    tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let raw = raw::decode(&path)?;
        let decode_time = start.elapsed();

        let start = Instant::now();
        let image = Engine::get()?.develop_preview(&raw, lens_default(&raw).and(LENS_OFFERED).and(Corrections::CHROMATIC))?;
        let develop_time = start.elapsed();

        Ok(Loaded {
            estimate: None,
            image,
            raw: Arc::new(raw),
            decode_time,
            develop_time,
            preview: true,
        })
    })
    .await
    .map_err(|e| format!("Worker failed: {e}"))?
}

/// The full neural demosaic of a decoded frame.
async fn develop(
    raw: Arc<RawFrame>,
    decode_time: Duration,
    lens: Corrections,
    quick: bool,
    cancel: Arc<AtomicBool>,
) -> Result<Loaded, String> {
    tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let engine = Engine::get()?;
        // The quick stage does not denoise, so it has no estimate to scale.
        let estimate = if quick { None } else { engine.estimate_noise(&raw)? };
        let noise = estimate.map(|e| e.scaled(noise_factor(NOISE_REDUCTION_DEFAULT)));
        let image = engine.develop(&raw, noise, lens, quick, &cancel)?;
        let develop_time = start.elapsed();

        Ok(Loaded {
            estimate,
            image,
            raw,
            decode_time,
            develop_time,
            preview: false,
        })
    })
    .await
    .map_err(|e| format!("Worker failed: {e}"))?
}

/// What a re-demosaic makes: the noise it denoises for (as far as known
/// when it starts), the lens corrections and the stage.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Settings {
    noise: Option<Noise>,
    lens: Corrections,
    quick: bool,
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

/// The factor on the estimated noise sigma for noise reduction slider
/// position `strength` (0..1): `(9^s - 1) / 2`, so 0 turns denoising off,
/// 0.5 uses the estimate as is and 1 is 4×. Exponential like a stop scale over
/// most of the range, but it reaches 0 at the bottom.
fn noise_factor(strength: f32) -> f64 {
    (9f64.powf(f64::from(strength)) - 1.0) / 2.0
}

/// Demosaics `raw` again with the noise reduction `strength`, the lens
/// corrections and the chosen stage, off the UI thread. `estimate` is the
/// noise estimate already known, if any: switching from the quick stage back
/// to the neural one has none yet, so it is measured then.
async fn redevelop(
    raw: Arc<RawFrame>,
    estimate: Option<Noise>,
    strength: f32,
    lens: Corrections,
    quick: bool,
    cancel: Arc<AtomicBool>,
) -> Result<Redeveloped, String> {
    tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let engine = Engine::get()?;
        let estimate = if quick {
            None
        } else {
            match estimate {
                Some(estimate) => Some(estimate),
                None => engine.estimate_noise(&raw)?,
            }
        };
        let noise = estimate.map(|e| e.scaled(noise_factor(strength)));
        let image = engine.develop(&raw, noise, lens, quick, &cancel)?;
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

/// Rail of a slider symmetric about zero (the value is scaled to -1..=1):
/// the handle filled. Sits under the slider (see [`centered_slider`]); the
/// geometry matches iced's slider with `HANDLE_RADIUS`.
struct CenteredTrack(f32);

const HANDLE_RADIUS: f32 = 7.0;
const RAIL_WIDTH: f32 = 4.0;

impl canvas::Program<Message> for CenteredTrack {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &iced::Renderer,
        theme: &iced::Theme,
        bounds: iced::Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let (fill, rest) = styling::centered_rail(theme);
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let y = (bounds.height - RAIL_WIDTH) / 2.0;
        let rail = |x: f32, width: f32, color| {
            let path = canvas::Path::rounded_rectangle(
                iced::Point::new(x, y),
                Size::new(width, RAIL_WIDTH),
                (RAIL_WIDTH / 2.0).into(),
            );
            (path, color)
        };
        let (path, color) = rail(0.0, bounds.width, rest);
        frame.fill(&path, color);

        let center = bounds.width / 2.0;
        let handle = HANDLE_RADIUS + (bounds.width - 2.0 * HANDLE_RADIUS) * (self.0 + 1.0) / 2.0;
        let (start, end) = (center.min(handle), center.max(handle));
        let (path, color) = rail(start, end - start, fill);
        frame.fill(&path, color);
        vec![frame.into_geometry()]
    }
}

/// A slider over -limit..=limit whose rail fills from the centre to the handle.
fn centered_slider<'a>(
    limit: f32,
    value: f32,
    on_change: impl Fn(f32) -> Message + 'a,
) -> Element<'a, Message> {
    let handle = slider(-limit..=limit, value, on_change)
        .step(0.01_f32)
        .default(0.0_f32)
        .height(HANDLE_RADIUS * 2.0)
        .style(styling::centered_slider);
    let track = canvas(CenteredTrack(value / limit))
        .width(Length::Fill)
        .height(HANDLE_RADIUS * 2.0);
    stack![track, handle].into()
}

/// A control's label followed by a small reset icon, shown only when `reset`
/// is set (the control differs from its default). Its slot is always
/// reserved so the row does not change height.
fn label_with_reset<'a>(label: &'a str, reset: Option<Message>) -> Element<'a, Message> {
    label_with_button(
        label,
        reset.map(|message| (CODEPOINT_RESTART, message, false)),
    )
}

/// A control's label followed by a small icon button (codepoint, message,
/// whether to dim the icon), or an empty slot of the same size.
fn label_with_button<'a>(
    label: &'a str,
    action: Option<(char, Message, bool)>,
) -> Element<'a, Message> {
    let slot: Element<'a, Message> = match action {
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
    };
    row![
        text(label)
            .size(FONT_SIZE_STANDARD)
            .color(styling::ADJUSTMENT_GRAY),
        slot
    ]
    .spacing(SPACING_SMALL)
    .align_y(Alignment::Center)
    .into()
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
        "Taken" | "Focal length" | "Aperture" | "Shutter" | "ISO" | "Exposure bias" => "Exposure",
        _ => "File",
    }
}

/// A heading over a two-column table of names and values.
fn metadata_table<'a>(
    group: &'a str,
    rows: Vec<(&'static str, String)>,
) -> Element<'a, Message> {
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

    /// The window icon is dropped silently if it fails to decode.
    #[test]
    fn icon_decodes() {
        assert!(window::icon::from_file_data(ICON_BYTES, None).is_ok());
    }
}
