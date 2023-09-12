use std::ops::Mul;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{init, Embedding, VarMap};

use crate::{frozenembed::FrozenEmbedding, EmbeddingLayerLike};

#[derive(Debug)]
pub struct LoraEmbedding {
    old: FrozenEmbedding,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    merged: bool,
}

/// Configuration for LoraEmbedding, with `num_embeddings` vectors of `embedding_dim` size`.
pub struct LoraEmbeddingConfig<'a> {
    pub rank: usize,
    pub alpha: f64,
    pub device: &'a Device,
    pub dtype: DType,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
}

/// Builder for LoraEmbeddingConfig. Call `build` to construct the config.
pub struct LoraEmbeddingConfigBuilder<'a> {
    pub config: LoraEmbeddingConfig<'a>,
}

impl<'a> LoraEmbeddingConfigBuilder<'a> {
    pub fn default(
        device: &'a Device,
        dtype: DType,
        num_embeddings: usize,
        embedding_dim: usize,
    ) -> Self {
        LoraEmbeddingConfigBuilder {
            config: LoraEmbeddingConfig {
                rank: 1,
                alpha: 1.,
                device,
                dtype,
                num_embeddings,
                embedding_dim,
            },
        }
    }

    /// Set the rank parameter
    pub fn rank(mut self, rank: usize) -> Self {
        self.config.rank = rank;
        self
    }

    /// Set the alpha parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Construct the config
    pub fn build(self) -> LoraEmbeddingConfig<'a> {
        self.config
    }
}

impl LoraEmbedding {
    pub fn new(old: &dyn EmbeddingLayerLike, config: &LoraEmbeddingConfig) -> Result<Self> {
        let map = VarMap::new();
        let a = map.get(
            (config.rank, config.num_embeddings),
            "a.weight",
            init::ZERO,
            config.dtype,
            config.device,
        )?;
        let b = map.get(
            (config.embedding_dim, config.rank),
            "b.weight",
            init::ZERO,
            config.dtype,
            config.device,
        )?;

        Ok(LoraEmbedding {
            old: FrozenEmbedding::new_from_embed(old)?,
            a,
            b,
            scale: if config.rank > 0 {
                Some(config.alpha / config.rank as f64)
            } else {
                None
            },
            merged: false,
        })
    }

    fn get_delta_weight(&self) -> Result<Tensor> {
        let result = self.b.matmul(&self.a)?;
        Ok(match self.scale {
            Some(scale) => result.mul(scale)?,
            None => result,
        })
    }

    pub fn merge(&mut self) -> Result<()> {
        self.old = FrozenEmbedding::new(
            &(self.embeddings() + self.get_delta_weight()?.transpose(0, 1))?,
            self.hidden_size(),
        )?;
        self.merged = true;
        Ok(())
    }

    pub fn unmerge(&mut self) -> Result<()> {
        self.old = FrozenEmbedding::new(
            &(self.embeddings() - self.get_delta_weight()?.transpose(0, 1))?,
            self.hidden_size(),
        )?;
        self.merged = false;
        Ok(())
    }

}

impl Module for LoraEmbedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut result = self.old.forward(input)?;
        if let Some(scale) = self.scale {
            let weight = self.a.transpose(0, 1)?;
            let weight = weight.reshape(weight.shape())?; //Get contiguous
            let hidden = weight.dim(1)?;

            let embed = Embedding::new(weight, hidden);
            let after_a = embed.forward(input)?;

            result = (result + after_a.broadcast_matmul(&self.b.transpose(0, 1)?)?)?;
            result = (result * scale)?;
        }
        Ok(result)
    }
}

impl EmbeddingLayerLike for LoraEmbedding {
    fn embeddings(&self) -> &Tensor {
        self.old.embeddings()
    }
    fn hidden_size(&self) -> usize {
        self.old.hidden_size()
    }
}
