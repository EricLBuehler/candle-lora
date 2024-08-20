//! The DINOv2 model.

use candle_core::{IndexOp, Result, Tensor, D};
use candle_lora::{
    Conv2dLayerLike, LinearLayerLike, LoraConfig, LoraConv2dConfig, LoraLinearConfig,
};
use candle_lora_macro::{replace_layer_fields, AutoLoraConvert};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};
use std::ops::Deref;
use std::sync::Arc;

const IMG_SIZE: usize = 518;
const PATCH_SIZE: usize = 14;
const NUM_CLASSES: usize = 1000;

fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug, AutoLoraConvert)]
#[replace_layer_fields]
struct DinoLinear {
    inner: Linear,
}

impl Deref for DinoLinear {
    type Target = Arc<dyn LinearLayerLike>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Debug)]
struct Attention {
    qkv: DinoLinear,
    proj: DinoLinear,
    num_heads: usize,
    scale: f64,
}

impl Attention {
    fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        proj_bias: bool,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let mut qkv = DinoLinear {
            inner: Arc::new(linear(vb.pp("qkv"), dim, dim * 3, qkv_bias)?),
        };
        let mut proj = DinoLinear {
            inner: Arc::new(linear(vb.pp("proj"), dim, dim, proj_bias)?),
        };
        let scale = 1. / ((dim / num_heads) as f64).sqrt();

        let loraconfig_qkv = LoraLinearConfig::new(dim, dim * 3);
        if merge {
            qkv.get_merged_lora_model(
                lora_config.clone(),
                &vb.pp("lora_qkv"),
                Some(loraconfig_qkv),
                None,
                None,
                None,
            )
        } else {
            qkv.get_lora_model(
                lora_config.clone(),
                &vb.pp("lora_qkv"),
                Some(loraconfig_qkv),
                None,
                None,
                None,
            )
        }

        let loraconfig_proj = LoraLinearConfig::new(dim, dim);
        if merge {
            proj.get_merged_lora_model(
                lora_config.clone(),
                &vb.pp("lora_proj"),
                Some(loraconfig_proj),
                None,
                None,
                None,
            )
        } else {
            proj.get_lora_model(
                lora_config.clone(),
                &vb.pp("lora_proj"),
                Some(loraconfig_proj),
                None,
                None,
                None,
            )
        }

        Ok(Self {
            qkv,
            proj,
            num_heads,
            scale,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)? // 02134
            .transpose(0, 1)? // 20134
            .transpose(2, 3)?; // 20314
        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;
        let attn = candle_nn::ops::softmax(&q.matmul(&k.t()?)?, D::Minus1)?;
        let attn = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.proj.forward(&attn)
    }
}

#[derive(Debug)]
struct LayerScale {
    gamma: Tensor,
}

impl LayerScale {
    fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self { gamma })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.gamma)
    }
}

#[derive(Debug)]
struct Mlp {
    fc1: DinoLinear,
    fc2: DinoLinear,
}

impl Mlp {
    fn new(
        vb: VarBuilder,
        in_features: usize,
        hidden_features: usize,
        bias: bool,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let out_features = in_features;
        let mut fc1 = DinoLinear {
            inner: Arc::new(linear(vb.pp("fc1"), in_features, hidden_features, bias)?),
        };
        let mut fc2 = DinoLinear {
            inner: Arc::new(linear(vb.pp("fc2"), hidden_features, out_features, bias)?),
        };

        let loraconfig_fc1 = LoraLinearConfig::new(in_features, hidden_features);
        if merge {
            fc1.get_merged_lora_model(
                lora_config.clone(),
                &vb.pp("lora_fc1"),
                Some(loraconfig_fc1),
                None,
                None,
                None,
            )
        } else {
            fc1.get_lora_model(
                lora_config.clone(),
                &vb.pp("lora_fc1"),
                Some(loraconfig_fc1),
                None,
                None,
                None,
            )
        }

        let loraconfig_fc2 = LoraLinearConfig::new(hidden_features, out_features);
        if merge {
            fc2.get_merged_lora_model(
                lora_config.clone(),
                &vb.pp("lora_fc2"),
                Some(loraconfig_fc2),
                None,
                None,
                None,
            )
        } else {
            fc2.get_lora_model(
                lora_config.clone(),
                &vb.pp("lora_fc2"),
                Some(loraconfig_fc2),
                None,
                None,
                None,
            )
        }

        Ok(Self { fc1, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.gelu()?;
        self.fc2.forward(&xs)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl Block {
    fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let attn = Attention::new(
            vb.pp("attn"),
            dim,
            num_heads,
            true,
            true,
            merge,
            lora_config.clone(),
        )?;
        let ls1 = LayerScale::new(vb.pp("ls1"), dim)?;
        let norm2 = layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let mlp = Mlp::new(vb.pp("mlp"), dim, dim * 4, true, merge, lora_config)?;
        let ls2 = LayerScale::new(vb.pp("ls2"), dim)?;
        Ok(Self {
            norm1,
            attn,
            ls1,
            norm2,
            mlp,
            ls2,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self
            .ls1
            .forward(&self.attn.forward(&self.norm1.forward(xs)?)?)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)?)?)?;
        xs + residual
    }
}

#[derive(Debug, AutoLoraConvert)]
#[replace_layer_fields]
struct DinoConv2d {
    inner: Conv2d,
}

impl Deref for DinoConv2d {
    type Target = Arc<dyn Conv2dLayerLike>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: DinoConv2d,
    patch_size: (usize, usize),
    num_patches: usize,
}

impl PatchEmbed {
    fn new(
        vb: VarBuilder,
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let config = candle_nn::Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let mut proj = DinoConv2d {
            inner: Arc::new(candle_nn::conv2d(
                in_chans,
                embed_dim,
                patch_size,
                config,
                vb.pp("proj"),
            )?),
        };
        let num_patches = (img_size / patch_size) * (img_size / patch_size);

        let loraconfig_proj = LoraConv2dConfig::new(in_chans, embed_dim);
        if merge {
            proj.get_merged_lora_model(
                lora_config.clone(),
                &vb.pp("lora_proj"),
                None,
                None,
                Some(loraconfig_proj),
                None,
            )
        } else {
            proj.get_lora_model(
                lora_config.clone(),
                &vb.pp("lora_proj"),
                None,
                None,
                Some(loraconfig_proj),
                None,
            )
        }

        Ok(Self {
            proj,
            patch_size: (patch_size, patch_size),
            num_patches,
        })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        let (patch_h, patch_w) = self.patch_size;
        if (h % patch_h) != 0 {
            candle_core::bail!("image height {h} is not a multiple of patch height {patch_h}")
        }
        if (w % patch_w) != 0 {
            candle_core::bail!("image width {w} is not a multiple of patch width {patch_w}")
        }
        let xs = self.proj.forward(xs)?;
        let (b, c, h, w) = xs.dims4()?;
        // flatten embeddings.
        xs.reshape((b, c, h * w))?.transpose(1, 2)
    }
}

#[derive(Debug)]
pub struct DinoVisionTransformer {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: DinoLinear,
}

impl DinoVisionTransformer {
    pub fn new(
        vb: VarBuilder,
        depth: usize,
        embed_dim: usize,
        num_heads: usize,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::new(
            vb.pp("patch_embed"),
            IMG_SIZE,
            PATCH_SIZE,
            3,
            embed_dim,
            merge,
            lora_config.clone(),
        )?;
        let cls_token = vb.get((1, 1, embed_dim), "cls_token")?;
        let num_tokens = 1;
        let pos_embed = vb.get(
            (1, patch_embed.num_patches + num_tokens, embed_dim),
            "pos_embed",
        )?;
        let mut head = DinoLinear {
            inner: Arc::new(linear(vb.pp("head"), 2 * embed_dim, NUM_CLASSES, true)?),
        };
        let norm = layer_norm(embed_dim, 1e-5, vb.pp("norm"))?;
        let vb_b = vb.pp("blocks");
        let blocks = (0..depth)
            .map(|i| {
                Block::new(
                    vb_b.pp(i.to_string()),
                    embed_dim,
                    num_heads,
                    merge,
                    lora_config.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let loraconfig_head = LoraLinearConfig::new(2 * embed_dim, NUM_CLASSES);
        if merge {
            head.get_merged_lora_model(
                lora_config,
                &vb.pp("lora_head"),
                Some(loraconfig_head),
                None,
                None,
                None,
            )
        } else {
            head.get_lora_model(
                lora_config,
                &vb.pp("lora_head"),
                Some(loraconfig_head),
                None,
                None,
                None,
            )
        }

        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
            head,
        })
    }

    fn interpolate_pos_encoding(&self, xs: &Tensor, w: usize, h: usize) -> Result<Tensor> {
        let npatch = xs.dim(1)? - 1;
        let n = self.pos_embed.dim(1)? - 1;
        let sqrt_n = (n as f64).sqrt();
        if npatch == n && w == h {
            return Ok(xs.clone());
        }
        let class_pos_embed = self.pos_embed.i((.., ..1))?;
        let patch_pos_embed = self.pos_embed.i((.., 1..))?;
        let dim = xs.dim(D::Minus1)?;
        let (w0, h0) = ((w / PATCH_SIZE) as f64 + 0.1, (h / PATCH_SIZE) as f64 + 0.1);
        let patch_pos_embed = patch_pos_embed
            .reshape((1, sqrt_n as usize, sqrt_n as usize, dim))?
            .transpose(2, 3)?
            .transpose(1, 2)?;
        // This uses bicubic interpolation in the original implementation.
        let patch_pos_embed = patch_pos_embed.upsample_nearest2d(h0 as usize, w0 as usize)?;
        let el_count = patch_pos_embed.shape().elem_count();
        let patch_pos_embed =
            patch_pos_embed
                .transpose(1, 2)?
                .transpose(2, 3)?
                .reshape((1, el_count / dim, dim))?;
        Tensor::cat(&[&class_pos_embed, &patch_pos_embed], 1)
    }

    fn prepare_tokens_with_mask(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _nc, w, h) = xs.dims4()?;
        let xs = self.patch_embed.forward(xs)?;
        let xs = Tensor::cat(&[&self.cls_token, &xs], 1)?;
        &xs + &self.interpolate_pos_encoding(&xs, w, h)?
    }
}

impl Module for DinoVisionTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.prepare_tokens_with_mask(xs)?;
        for blk in self.blocks.iter() {
            xs = blk.forward(&xs)?
        }
        let xs = self.norm.forward(&xs)?;
        let xs_norm_clstoken = xs.i((.., 0))?;
        let xs_norm_patchtokens = xs.i((.., 1..))?.mean(1)?;
        let xs = Tensor::cat(&[xs_norm_clstoken, xs_norm_patchtokens], D::Minus1)?;
        self.head.forward(&xs)
    }
}

pub fn vit_small(
    vb: VarBuilder,
    merge: bool,
    lora_config: LoraConfig,
) -> Result<DinoVisionTransformer> {
    DinoVisionTransformer::new(vb, 12, 384, 6, merge, lora_config)
}
