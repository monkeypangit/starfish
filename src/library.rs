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
pub fn thumbnails(files: Vec<PathBuf>) -> impl Stream<Item = (usize, Option<Handle>)> {
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
fn thumbnail(path: &Path) -> Option<Handle> {
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
    Some(Handle::from_rgba(
        image.width(),
        image.height(),
        image.into_raw(),
    ))
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

/// The photo selected in the last run, if it still exists.
pub fn remembered_photo() -> Option<PathBuf> {
    remembered("photo").filter(|p| p.is_file())
}

pub fn remember_photo(photo: &Path) {
    if photo.is_file() {
        remember("photo", photo);
    }
}
