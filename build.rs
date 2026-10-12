//! Embeds the neural demosaic exports (`training/*_wgpu/{f16,f32}/`) for
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

fn main() {
    let training = Path::new(env!("CARGO_MANIFEST_DIR")).join("training");
    let mut arms = String::new();
    for export in EXPORTS {
        let dir = training.join(export);
        println!("cargo:rerun-if-changed={}", dir.display());
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}; run the export in training/", dir.display()))
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
        "/// An embedded export file: `export` like \"xtrans_wgpu/f16\", `name` like \"graph.txt\".\n\
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
