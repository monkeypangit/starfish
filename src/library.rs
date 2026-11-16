//! The folder shown in the file browser: listing raw files, loading their
//! embedded thumbnails and remembering the folder between runs.

use std::fs;
use std::path::{Path, PathBuf};

use iced::futures::{Stream, StreamExt, stream};
use iced::widget::image::Handle;
use image::DynamicImage;
use rawler::Orientation;
use rawler::decoders::RawDecodeParams;
use rawler::rawsource::RawSource;

use crate::raw::RAW_EXTENSIONS;

/// Longest side of a stored thumbnail, in pixels. Covers wide cells on high-DPI screens.
const THUMBNAIL_SIZE: u32 = 320;
/// Thumbnails decoded at the same time.
const THUMBNAIL_WORKERS: usize = 4;

/// The raw files directly in `folder`, sorted by name.
pub fn list(folder: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(folder)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| RAW_EXTENSIONS.iter().any(|x| x.eq_ignore_ascii_case(e)))
        })
        .collect();
    files.sort_by_key(|path| path.file_name().map(|n| n.to_ascii_lowercase()));
    files
}

/// Decodes the thumbnails of `files` a few at a time, yielding
/// `(index, thumbnail)` as each finishes. `None` if a file has none.
pub fn thumbnails(files: Vec<PathBuf>) -> impl Stream<Item = (usize, Option<(Handle, (u32, u32))>)> {
    stream::iter(files.into_iter().enumerate())
        .map(|(index, path)| async move {
            let handle = tokio::task::spawn_blocking(move || thumbnail(&path))
                .await
                .ok()
                .flatten();
            (index, handle)
        })
        .buffer_unordered(THUMBNAIL_WORKERS)
}

/// The JPEG the camera embedded in the raw file, upright and shrunk. Never
/// decodes the raw data itself, so files without a preview get none.
fn thumbnail(path: &Path) -> Option<(Handle, (u32, u32))> {
    let source = RawSource::new(path).ok()?;
    let decoder = rawler::get_decoder(&source).ok()?;
    let params = RawDecodeParams::default();
    let image = match decoder.thumbnail_image(&source, &params).ok()? {
        Some(image) => image,
        None => decoder.preview_image(&source, &params).ok()??,
    };
    let orientation = decoder
        .raw_metadata(&source, &params)
        .ok()
        .and_then(|metadata| metadata.exif.orientation)
        .map_or(Orientation::Normal, Orientation::from_u16);

    let image = upright(image.thumbnail(THUMBNAIL_SIZE, THUMBNAIL_SIZE), orientation).to_rgba8();
    let size = (image.width(), image.height());
    Some((Handle::from_rgba(size.0, size.1, image.into_raw()), size))
}

/// The JPEG the camera embedded in the raw file, as it is (not upright, not
/// shrunk), for fitting the tone to; `None` if the file has none.
pub fn embedded_preview(path: &Path) -> Option<DynamicImage> {
    let source = RawSource::new(path).ok()?;
    let decoder = rawler::get_decoder(&source).ok()?;
    let params = RawDecodeParams::default();
    match decoder.thumbnail_image(&source, &params).ok()? {
        Some(image) => Some(image),
        None => decoder.preview_image(&source, &params).ok()?,
    }
}

fn upright(image: DynamicImage, orientation: Orientation) -> DynamicImage {
    match orientation {
        Orientation::Normal | Orientation::Unknown => image,
        Orientation::HorizontalFlip => image.fliph(),
        Orientation::Rotate180 => image.rotate180(),
        Orientation::VerticalFlip => image.flipv(),
        Orientation::Transpose => image.rotate90().fliph(),
        Orientation::Rotate90 => image.rotate90(),
        Orientation::Transverse => image.rotate270().fliph(),
        Orientation::Rotate270 => image.rotate270(),
    }
}

/// An absolute path from environment variable `var`.
fn env_dir(var: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
}

/// The folder for Starfish's settings: `%APPDATA%\Starfish` on Windows,
/// `~/Library/Application Support/Starfish` on macOS, else
/// `$XDG_CONFIG_HOME/starfish` (default `~/.config/starfish`).
fn config_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        Some(env_dir("APPDATA")?.join("Starfish"))
    } else if cfg!(target_os = "macos") {
        Some(env_dir("HOME")?.join("Library/Application Support/Starfish"))
    } else {
        let config =
            env_dir("XDG_CONFIG_HOME").or_else(|| Some(env_dir("HOME")?.join(".config")))?;
        Some(config.join("starfish"))
    }
}

/// The folder for files Starfish writes for itself (the log):
/// `%LOCALAPPDATA%\Starfish` on Windows, `~/Library/Logs/Starfish` on macOS,
/// else `$XDG_STATE_HOME/starfish` (default `~/.local/state/starfish`).
pub fn data_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        Some(env_dir("LOCALAPPDATA")?.join("Starfish"))
    } else if cfg!(target_os = "macos") {
        Some(env_dir("HOME")?.join("Library/Logs/Starfish"))
    } else {
        let state =
            env_dir("XDG_STATE_HOME").or_else(|| Some(env_dir("HOME")?.join(".local/state")))?;
        Some(state.join("starfish"))
    }
}

fn settings_file(name: &str) -> Option<PathBuf> {
    Some(config_dir()?.join(name))
}

fn remembered(name: &str) -> Option<PathBuf> {
    Some(PathBuf::from(
        fs::read_to_string(settings_file(name)?)
            .ok()?
            .trim_end_matches('\n'),
    ))
}

/// Best effort: losing the setting is not worth bothering the user.
fn remember(name: &str, path: &Path) {
    let Some(file) = settings_file(name) else {
        return;
    };
    if let Some(dir) = file.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let _ = fs::write(file, path.as_os_str().as_encoded_bytes());
}

/// The folder from the last run, if it still exists.
pub fn remembered_folder() -> Option<PathBuf> {
    remembered("folder").filter(|p| p.is_dir())
}

pub fn remember_folder(folder: &Path) {
    if folder.is_dir() {
        remember("folder", folder);
    }
}

/// The folder of the last export, if it still exists.
pub fn remembered_export_folder() -> Option<PathBuf> {
    remembered("export_folder").filter(|p| p.is_dir())
}

pub fn remember_export_folder(folder: &Path) {
    if folder.is_dir() {
        remember("export_folder", folder);
    }
}

/// The photo selected in the last run, if it still exists.
pub fn remembered_photo() -> Option<PathBuf> {
    remembered("photo").filter(|p| p.is_file())
}

pub fn remember_photo(photo: &Path) {
    if photo.is_file() {
        remember("photo", photo);
    }
}

/// The sidebars' widths (left, right) in logical pixels from the last run.
pub fn remembered_pane_widths() -> Option<(f32, f32)> {
    let text = fs::read_to_string(settings_file("pane_widths")?).ok()?;
    let mut numbers = text.split_whitespace().map(|n| n.parse::<f32>().ok());
    let (left, right) = (numbers.next()??, numbers.next()??);
    (left.is_finite() && right.is_finite() && left > 0.0 && right > 0.0).then_some((left, right))
}

pub fn remember_pane_widths(left: f32, right: f32) {
    let Some(file) = settings_file("pane_widths") else {
        return;
    };
    if let Some(dir) = file.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let _ = fs::write(file, format!("{left} {right}\n"));
}

/// The histogram's height as a share of its width from the last run.
pub fn remembered_histogram_ratio() -> Option<f32> {
    let ratio = fs::read_to_string(settings_file("histogram_ratio")?)
        .ok()?
        .trim()
        .parse::<f32>()
        .ok()?;
    (ratio.is_finite() && ratio > 0.0).then_some(ratio)
}

pub fn remember_histogram_ratio(ratio: f32) {
    let Some(file) = settings_file("histogram_ratio") else {
        return;
    };
    if let Some(dir) = file.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let _ = fs::write(file, format!("{ratio}\n"));
}

/// The sites the user marked as hot or dead pixels for a camera and frame
/// size, `(y, x)` in the raw frame: one line per camera body (model + serial number if the file has one) in the `defects`
/// settings file, `camera<TAB>WIDTHxHEIGHT<TAB>y,x y,x ...`.
pub fn defects(camera: &str, width: usize, height: usize) -> Vec<(usize, usize)> {
    let Some(text) = settings_file("defects").and_then(|f| fs::read_to_string(f).ok()) else {
        return Vec::new();
    };
    let (camera, size) = (defect_key(camera), format!("{width}x{height}"));
    text.lines()
        .filter_map(|line| {
            let mut fields = line.split('\t');
            (fields.next()? == camera && fields.next()? == size)
                .then(|| fields.next().unwrap_or(""))
        })
        .next()
        .map(|sites| {
            sites
                .split(' ')
                .filter_map(|site| {
                    let (y, x) = site.split_once(',')?;
                    Some((y.parse().ok()?, x.parse().ok()?))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Remembers `sites` for the camera and size (none: forgets them). Best effort.
pub fn remember_defects(camera: &str, width: usize, height: usize, sites: &[(usize, usize)]) {
    let Some(file) = settings_file("defects") else {
        return;
    };
    let (camera, size) = (defect_key(camera), format!("{width}x{height}"));
    let old = fs::read_to_string(&file).unwrap_or_default();
    let mut lines: Vec<String> = old
        .lines()
        .filter(|line| {
            let mut fields = line.split('\t');
            !(fields.next() == Some(&camera) && fields.next() == Some(&size))
        })
        .map(String::from)
        .collect();
    if !sites.is_empty() {
        let sites: Vec<String> = sites.iter().map(|(y, x)| format!("{y},{x}")).collect();
        lines.push(format!("{camera}\t{size}\t{}", sites.join(" ")));
    }
    if let Some(dir) = file.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let _ = fs::write(file, lines.join("\n") + "\n");
}

fn defect_key(camera: &str) -> String {
    camera.replace(['\t', '\n'], " ")
}
