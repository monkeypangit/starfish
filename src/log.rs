//! A plain log file for bug reports: `starfish.log` in the platform's data
//! folder ([`crate::library::data_dir`]), started afresh on every run, plus
//! stderr. Release builds on Windows have no console, so the file is the only
//! place a tester can find what went wrong. Panics are written there too.

use std::fs::File;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::SystemTime;

static FILE: OnceLock<Option<Mutex<File>>> = OnceLock::new();

/// Where the log goes, if the data folder can be found.
pub fn path() -> Option<PathBuf> {
    crate::library::data_dir().map(|dir| dir.join("starfish.log"))
}

/// Opens the log and installs the panic hook. Logging is best effort: without
/// a writable data folder lines only go to stderr.
pub fn init() {
    FILE.get_or_init(|| {
        let path = path()?;
        std::fs::create_dir_all(path.parent()?).ok()?;
        File::create(path).ok().map(Mutex::new)
    });
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        write(&format!(
            "panic: {info}\n{}",
            std::backtrace::Backtrace::force_capture()
        ));
        default(info);
    }));
    write(&format!(
        "Starfish {} on {} {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
}

/// Appends one line, prefixed with seconds since the Unix epoch.
pub fn write(line: &str) {
    let time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    let line = format!("[{time:.3}] {line}\n");
    eprint!("{line}");
    if let Some(Some(file)) = FILE.get() {
        let mut file = file.lock().unwrap_or_else(PoisonError::into_inner);
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
    }
}
