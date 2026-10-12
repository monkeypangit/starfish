//! The per-pixel develop shared by the preview and the export
//! (`shaders/develop.wgsl`): its uniform and the joining of its source in
//! front of the shaders that call it (WGSL has no includes).

use bytemuck::{Pod, Zeroable};

use crate::agx::{self, Curve};
use crate::preview::{Adjustments, LOCAL_GAIN};
use crate::processing::Image;

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
    grade: [[f32; 4]; 3],
    curve: [f32; 4],
    limits: [f32; 4],
    levels: [f32; 4],
    exposure: f32,
    saturation: f32,
    local: f32,
    black: f32,
}

impl EditUniform {
    pub fn new(adjustments: &Adjustments, image: &Image) -> Self {
        let row = |r: [f32; 3]| [r[0], r[1], r[2], 0.0];
        let curve = Curve::new(adjustments);
        let [shadows, midtones, highlights] = adjustments.levels;
        Self {
            cam_to_srgb: adjustments.cam_to_srgb(image.cam_to_srgb, image.white).map(row),
            grade: adjustments.grade_tints(),
            curve: [curve.pivot_x, curve.pivot_y, curve.slope, curve.power],
            limits: [curve.toe, curve.shoulder, agx::WHITE, 0.0],
            levels: [shadows, midtones, highlights, 0.0],
            exposure: adjustments.exposure,
            saturation: adjustments.saturation,
            local: adjustments.local_contrast * LOCAL_GAIN,
            black: adjustments.black_point,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_uniform_matches_the_shader() {
        let module = naga::front::wgsl::parse_str(SHADER).unwrap();
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap();
        let edit = module.types.iter().find(|(_, t)| t.name.as_deref() == Some("Edit")).unwrap().1;
        assert_eq!(edit.inner.size(module.to_ctx()) as usize, std::mem::size_of::<EditUniform>());
    }
}
