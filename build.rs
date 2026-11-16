//! Embeds the neural demosaic exports (`models/*_wgpu/{f16,f32}/`) for
//! release builds: generates `models.rs` with `file(export, name)`, which
//! `src/processing/neural.rs` loads them through. Debug builds read the
//! exports from disk instead and ignore it. Windows builds also get the
//! app icon embedded in the .exe.

use std::fmt::Write as _;
use std::path::Path;

const EXPORTS: &[&str] = &[
    "bayer_wgpu/f16",
    "bayer_wgpu/f32",
    "xtrans_wgpu/f16",
    "xtrans_wgpu/f32",
];

/// The light and tiny networks for weaker GPUs: embedded when exported. While one is
/// missing, any change under models/ reruns this, so a new export is
/// picked up.
const OPTIONAL: &[&str] = &[
    "bayer_lite_wgpu/f16",
    "bayer_lite_wgpu/f32",
    "xtrans_lite_wgpu/f16",
    "xtrans_lite_wgpu/f32",
    "bayer_tiny_wgpu/f16",
    "bayer_tiny_wgpu/f32",
    "xtrans_tiny_wgpu/f16",
    "xtrans_tiny_wgpu/f32",
];

fn main() {
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    let mut arms = String::new();
    let mut names = Vec::new();
    let optional = OPTIONAL.iter().filter(|export| {
        let found = models.join(export).is_dir();
        if !found {
            println!("cargo:rerun-if-changed={}", models.display());
        }
        found
    });
    for export in EXPORTS.iter().chain(optional) {
        let dir = models.join(export);
        println!("cargo:rerun-if-changed={}", dir.display());
        names.push(*export);
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}; sync the exports into models/ (scripts/sync-models.sh)", dir.display()))
            .map(|entry| entry.unwrap().path())
            .collect();
        files.sort();
        for path in files {
            let name = path.file_name().unwrap().to_str().unwrap();
            writeln!(
                arms,
                "        ({export:?}, {name:?}) => Some(include_bytes!({:?})),",
                path.to_str().unwrap()
            )
            .unwrap();
        }
    }
    let code = format!(
        "/// The embedded exports, like \"xtrans_wgpu/f16\".\n\
         pub const EXPORTS: &[&str] = &{names:?};\n\
         /// An embedded export file: `export` like \"xtrans_wgpu/f16\", `name` like \"graph.txt\".\n\
         pub fn file(export: &str, name: &str) -> Option<&'static [u8]> {{\n    \
             match (export, name) {{\n{arms}        _ => None,\n    }}\n}}\n"
    );
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("models.rs");
    std::fs::write(out, code).unwrap();

    // The .exe's own icon, for Explorer and the taskbar (main.rs sets the
    // window icon at run time).
    #[cfg(windows)]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rerun-if-changed=resources/icon/starfish.ico");
        winresource::WindowsResource::new()
            .set_icon("resources/icon/starfish.ico")
            .compile()
            .expect("embedding the Windows icon");
    }
}
