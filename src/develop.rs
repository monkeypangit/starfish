//! The per-pixel develop shared by the preview and the export
//! (`shaders/develop.wgsl`): its uniform and the joining of its source in
//! front of the shaders that call it (WGSL has no includes).

use bytemuck::{Pod, Zeroable};

use crate::color_edit::{self, MIX_COLORS, RANGES};
use crate::preview::{Adjustments, LOCAL_GAIN};
use crate::processing::Image;
use crate::tone::{self, Curve};

/// `develop.wgsl`'s source.
pub const SHADER: &str = include_str!("shaders/develop.wgsl");

/// `source` with `develop.wgsl` in front of it, ready to compile.
pub fn shader(label: &'static str, source: &str) -> wgpu::ShaderModuleDescriptor<'static> {
    wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(format!("{SHADER}\n{source}").into()),
    }
}

/// `develop.wgsl`'s `Lens`: where the lens table applies.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LensUniform {
    pub centre: [f32; 2],
    pub inv_half_diagonal: f32,
    pub lut_max: f32,
}

/// The user's edits as `develop.wgsl`'s `Edit` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct EditUniform {
    /// Camera RGB → linear sRGB, row-major (w unused).
    pub cam_to_srgb: [[f32; 4]; 3],
    curve: [f32; 4],
    limits: [f32; 4],
    /// The colour mixer (`color_edit::ColorMix::pack`) and the grading
    /// (`color_edit::Grading::pack`).
    colors: [[f32; 4]; MIX_COLORS],
    grade: [[f32; 4]; RANGES],
    exposure: f32,
    saturation: f32,
    local: f32,
    black: f32,
    grain: f32,
    sharpness: f32,
    edit_flags: u32,
    vibrance: f32,
    vibrance_range: f32,
    added_grain: f32,
    _pad: [f32; 2],
}

impl EditUniform {
    pub fn new(adjustments: &Adjustments, image: &Image) -> Self {
        let row = |r: [f32; 3]| [r[0], r[1], r[2], 0.0];
        let curve = Curve::new(adjustments);
        Self {
            cam_to_srgb: adjustments
                .cam_to_srgb(image.cam_to_srgb, image.white)
                .map(row),
            curve: [curve.pivot_x, curve.pivot_y, curve.slope, curve.toe_power],
            colors: adjustments.color_mix.pack(),
            grade: adjustments.grading.pack(),
            limits: [curve.toe, curve.shoulder, tone::WHITE, curve.shoulder_power],
            exposure: adjustments.exposure,
            saturation: adjustments.saturation,
            local: adjustments.local_contrast * LOCAL_GAIN,
            black: adjustments.black_point,
            grain: adjustments.grain,
            sharpness: adjustments.sharpness,
            edit_flags: color_edit::flags(&adjustments.color_mix, &adjustments.grading),
            vibrance: adjustments.vibrance,
            vibrance_range: crate::preview::VIBRANCE_RANGE,
            added_grain: adjustments.added_grain,
            _pad: [0.0; 2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_uniform_matches_the_shader() {
        let module = naga::front::wgsl::parse_str(SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let edit = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("Edit"))
            .unwrap()
            .1;
        assert_eq!(
            edit.inner.size(module.to_ctx()) as usize,
            std::mem::size_of::<EditUniform>()
        );
    }
}
