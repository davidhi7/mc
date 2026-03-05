use image::{DynamicImage, GenericImageView, ImageError};
use thiserror::Error;
use wgpu::{
    Device, Extent3d, Origin3d, Queue, Sampler, TexelCopyBufferLayout, TexelCopyTextureInfo,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
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
    LogOakTopBottom,
    LogOakSide,
    LogSpruceTopBottom,
    LogSpruceSide,
    LeavesOak,
    LeavesSpruce,
}

impl Texture {
    const fn texture_path(&self) -> &'static str {
        match self {
            Texture::Stone => "block/stone.png",
            Texture::GrassBlockTop => "block/grass_block_top.png",
            Texture::Dirt => "block/dirt.png",
            Texture::Sand => "block/sand.png",
            Texture::Gravel => "block/gravel.png",
            Texture::Andesite => "block/andesite.png",
            Texture::Snow => "block/snow.png",
            Texture::LogOakTopBottom => "block/oak_log_top.png",
            Texture::LogOakSide => "block/oak_log.png",
            Texture::LogSpruceTopBottom => "block/spruce_log_top.png",
            Texture::LogSpruceSide => "block/spruce_log.png",
            Texture::LeavesOak => "block/oak_leaves.png",
            Texture::LeavesSpruce => "block/spruce_leaves.png",
        }
    }

    const fn all() -> &'static [Texture] {
        &[
            Texture::Stone,
            Texture::GrassBlockTop,
            Texture::Dirt,
            Texture::Sand,
            Texture::Gravel,
            Texture::Andesite,
            Texture::Snow,
            Texture::LogOakTopBottom,
            Texture::LogOakSide,
            Texture::LogSpruceTopBottom,
            Texture::LogSpruceSide,
            Texture::LeavesOak,
            Texture::LeavesSpruce,
        ]
    }
}

#[derive(Debug, Error)]
pub enum TextureLoadError {
    #[error("Texture fetch via wasm failed")]
    #[cfg(target_arch = "wasm32")]
    WasmFetch,
    #[error(transparent)]
    NativeIo(#[from] std::io::Error),
    #[error(transparent)]
    TextureParse(#[from] ImageError),
}

async fn load_texture_files() -> Result<Vec<DynamicImage>, TextureLoadError> {
    let mut textures = Vec::with_capacity(Texture::all().len());
    #[cfg(not(target_arch = "wasm32"))]
    {
        for texture in Texture::all() {
            let bytes = &std::fs::read(TEXTURE_DIR.to_string() + texture.texture_path())?;
            textures.push(image::load_from_memory(bytes)?);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use futures::future;

        use crate::wasm_fetch;

        for texture in future::join_all(
            Texture::all()
                .iter()
                .map(|texture| TEXTURE_DIR.to_string() + texture.texture_path())
                .map(wasm_fetch::fetch),
        )
        .await
        {
            let bytes = texture.map_err(|_| TextureLoadError::WasmFetch)?;
            textures.push(image::load_from_memory(&bytes)?);
        }
    }

    Ok(textures)
}

pub async fn load_textures(
    device: &Device,
    queue: &Queue,
) -> Result<TextureView, TextureLoadError> {
    let mut texture_files = load_texture_files().await?.into_iter().peekable();
    let Some(img) = texture_files.peek() else {
        panic!("There should be at least one texture");
    };
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("texture array"),
        size: Extent3d {
            width: img.dimensions().0,
            height: img.dimensions().1,
            depth_or_array_layers: Texture::all().len().try_into().unwrap(),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (i, img) in texture_files.enumerate() {
        assert_eq!(
            (texture.size().width, texture.size().height),
            img.dimensions(),
            "Textures have inconsistent dimensions"
        );

        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d {
                    x: 0,
                    y: 0,
                    z: i.try_into().unwrap(),
                },
                aspect: TextureAspect::All,
            },
            &img.to_rgba8(),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * img.dimensions().0),
                rows_per_image: Some(img.dimensions().1),
            },
            Extent3d {
                width: img.dimensions().0,
                height: img.dimensions().1,
                depth_or_array_layers: 1,
            },
        );
    }

    Ok(texture.create_view(&TextureViewDescriptor::default()))
}

/// Create texture sampler
pub fn create_sampler(device: &Device) -> Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    })
}
