use std::fs;

use anyhow::*;
use image::GenericImageView;
use wgpu::{
    Device, Sampler, TexelCopyBufferLayout, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor,
};

const TEXTURE_DIR: &str = "res/assets/minecraft/textures/";

const TEXTURES: [&str; 7] = [
    "block/stone.png",
    "block/grass_block_top.png",
    "block/dirt.png",
    "block/sand.png",
    "block/gravel.png",
    "block/andesite.png",
    "block/snow.png",
];

/// Load textures and return texture views
pub fn load_textures(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<TextureView>> {
    let mut texture_views = Vec::new();

    for file in TEXTURES {
        let img = image::load_from_memory(fs::read(TEXTURE_DIR.to_owned() + file)?.as_slice())?;
        let dimensions = img.dimensions();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&TextureDescriptor {
            label: Some(&("texture ".to_owned() + file)),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            texture.as_image_copy(),
            &img.to_rgba8(),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        texture_views.push(texture.create_view(&TextureViewDescriptor::default()));
    }

    Ok(texture_views)
}

/// Create texture sampler
pub fn create_sampler(device: &Device) -> Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    })
}
