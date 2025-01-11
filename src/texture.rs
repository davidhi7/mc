use std::{fs, num::NonZeroU32};

use anyhow::*;
use image::GenericImageView;
use wgpu::{BindGroup, BindGroupLayout, TextureUsages, TextureView};

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

/// Create bind group and bind group layout for a texture array and a texture sampler.
pub fn load_textures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(BindGroupLayout, BindGroup)> {
    let texture_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: NonZeroU32::new(TEXTURES.len() as u32),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("texture bind group layout"),
        });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let mut texture_views: Vec<TextureView> = Vec::new();

    for file in TEXTURES {
        let img = image::load_from_memory(fs::read(TEXTURE_DIR.to_owned() + file)?.as_slice())?;
        let dimensions = img.dimensions();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&("texture ".to_owned() + file)),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            texture.as_image_copy(),
            &img.to_rgba8(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        texture_views.push(texture.create_view(&wgpu::TextureViewDescriptor::default()));
    }

    let texture_view_refs: Vec<&TextureView> = texture_views.iter().collect();

    let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&texture_view_refs),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
        label: Some("texture bind group"),
    });

    Ok((texture_bind_group_layout, texture_bind_group))
}
