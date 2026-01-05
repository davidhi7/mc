use std::fs;

use anyhow::*;
use image::GenericImageView;
use wgpu::{
    Device, Sampler, TexelCopyBufferLayout, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor,
};

const TEXTURE_DIR: &str = "res/assets/minecraft/textures/";

#[repr(u8)]
pub enum Texture {
    Stone,
    GrassBlockTop,
    Dirt,
    Sand,
    Gravel,
    Andesite,
    Snow,
    Water,
    LogOakTopBottom,
    LogOakSide,
    LogSpruceTopBottom,
    LogSpruceSide,
    LeavesOak,
    LeavesSpruce,
}

impl Texture {
    fn texture_path(&self) -> &'static str {
        match self {
            Texture::Stone => "block/stone.png",
            Texture::GrassBlockTop => "block/grass_block_top.png",
            Texture::Dirt => "block/dirt.png",
            Texture::Sand => "block/sand.png",
            Texture::Gravel => "block/gravel.png",
            Texture::Andesite => "block/andesite.png",
            Texture::Snow => "block/snow.png",
            Texture::Water => "block/water_still.png",
            Texture::LogOakTopBottom => "block/oak_log_top.png",
            Texture::LogOakSide => "block/oak_log.png",
            Texture::LogSpruceTopBottom => "block/spruce_log_top.png",
            Texture::LogSpruceSide => "block/spruce_log.png",
            Texture::LeavesOak => "block/oak_leaves.png",
            Texture::LeavesSpruce => "block/spruce_leaves.png",
        }
    }

    fn iter() -> impl Iterator<Item = Texture> {
        [
            Texture::Stone,
            Texture::GrassBlockTop,
            Texture::Dirt,
            Texture::Sand,
            Texture::Gravel,
            Texture::Andesite,
            Texture::Snow,
            Texture::Water,
            Texture::LogOakTopBottom,
            Texture::LogOakSide,
            Texture::LogSpruceTopBottom,
            Texture::LogSpruceSide,
            Texture::LeavesOak,
            Texture::LeavesSpruce,
        ]
        .into_iter()
    }
}

/// Load textures and return texture views
pub fn load_textures(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<TextureView>> {
    let mut texture_views = Vec::new();

    for texture in Texture::iter() {
        let img = image::load_from_memory(
            fs::read(TEXTURE_DIR.to_owned() + texture.texture_path())?.as_slice(),
        )?;
        let dimensions = img.dimensions();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&TextureDescriptor {
            label: Some(&("texture ".to_owned() + texture.texture_path())),
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
