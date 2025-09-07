use wgpu::{include_wgsl, ShaderModuleDescriptor};

pub const SHADER_TERRAIN: ShaderModuleDescriptor = include_wgsl!("shaders/terrain.wgsl");

pub const SHADER_WATER: ShaderModuleDescriptor = include_wgsl!("shaders/water.wgsl");

pub const SHADER_RETICLE: ShaderModuleDescriptor = include_wgsl!("shaders/reticle.wgsl");

pub const SHADER_FRUSTUM_CULLING: ShaderModuleDescriptor =
    include_wgsl!("shaders/frustum-culling.wgsl");

pub const SHADER_BLOCK_OUTLINES: ShaderModuleDescriptor =
    include_wgsl!("shaders/block-outlines.wgsl");
