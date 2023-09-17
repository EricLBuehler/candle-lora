use std::ops::Mul;

use candle_core::{Module, Result, Tensor};
use candle_nn::{init, Embedding, Init, VarBuilder};
use either::Either;
use trc::Trc;

use crate::{
    frozenembed::FrozenEmbedding, EmbeddingLayerLike, LoraConfig, Merge, MergeError,
    MergeErrorOrError,
};

#[derive(Debug, Clone)]
pub struct LoraEmbedding {
    old: Trc<FrozenEmbedding>,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    merged: bool,
}

/// Configuration for LoraEmbedding, with `num_embeddings` vectors of `embedding_dim` size`.
pub struct LoraEmbeddingConfig {
    num_embeddings: usize,
    embedding_dim: usize,
}

impl LoraEmbeddingConfig {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        LoraEmbeddingConfig {
            num_embeddings,
            embedding_dim,
        }
    }
}

impl LoraEmbedding {
    pub fn new(
        old: &dyn EmbeddingLayerLike,
        embed_config: &LoraEmbeddingConfig,
        config: &LoraConfig,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let a = vb.get_with_hints(
            (config.rank, embed_config.num_embeddings),
            "a.weight",
            init::ZERO,
        )?;
        let b = vb.get_with_hints(
            (embed_config.embedding_dim, config.rank),
            "b.weight",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        Ok(LoraEmbedding {
            old: Trc::new(FrozenEmbedding::new_from_embed(old)?),
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
}

impl Merge for LoraEmbedding {
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError> {
        let result = self.b.matmul(&self.a).map_err(Either::Right)?;
        Ok(match self.scale {
            Some(scale) => result.mul(scale).map_err(Either::Right)?,
            None => result,
        })
    }

    fn merge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError> {
        if self.merged {
            Err(Either::Left(MergeError::AlreadyMerged))
        } else {
            self.old = Trc::new(
                FrozenEmbedding::new(
                    &(self.embeddings() + self.get_delta_weight()?.transpose(0, 1))
                        .map_err(Either::Right)?,
                    self.hidden_size(),
                )
                .map_err(Either::Right)?,
            );
            self.merged = true;
            Ok(())
        }
    }

    fn unmerge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError> {
        if !self.merged {
            Err(Either::Left(MergeError::NotMerged))
        } else {
            self.old = Trc::new(
                FrozenEmbedding::new(
                    &(self.embeddings() - self.get_delta_weight()?.transpose(0, 1))
                        .map_err(Either::Right)?,
                    self.hidden_size(),
                )
                .map_err(Either::Right)?,
            );
            self.merged = false;
            Ok(())
        }
    }
}

impl Module for LoraEmbedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut result = self.old.forward(input)?;
        if let Some(scale) = self.scale {
            let a = self.a.t()?;
            let a = a.reshape(a.shape())?;
            let b = self.b.t()?;
            let b = b.reshape(b.shape())?;

            let embed = Embedding::new(a.clone(), a.dim(1)?);
            let after_a = embed.forward(input)?;
            result = (result + (after_a.broadcast_matmul(&b)?).mul(scale))?
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
