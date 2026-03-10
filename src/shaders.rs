use wgpu::{ShaderModuleDescriptor, include_wgsl};

pub const SHADER_TERRAIN: ShaderModuleDescriptor = include_wgsl!("shaders/terrain.wgsl");

pub const SHADER_WATER: ShaderModuleDescriptor = include_wgsl!("shaders/water.wgsl");

pub const SHADER_CROSSHAIR: ShaderModuleDescriptor = include_wgsl!("shaders/debug_crosshair.wgsl");

pub const SHADER_FRUSTUM_CULLING: ShaderModuleDescriptor =
    include_wgsl!("shaders/frustum-culling.wgsl");

pub const SHADER_BLOCK_OUTLINES: ShaderModuleDescriptor =
    include_wgsl!("shaders/block-outlines.wgsl");
