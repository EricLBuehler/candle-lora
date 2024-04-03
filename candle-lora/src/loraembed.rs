use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{Module, Result, Tensor};
use candle_nn::{init, Embedding, Init, VarBuilder};
use either::Either;

use crate::{
    frozenembed::FrozenEmbedding, EmbeddingLayerLike, LoraConfig, Merge, MergeError,
    MergeErrorOrError, Saveable,
};

#[derive(Debug, Clone)]
pub struct LoraEmbedding {
    old: Arc<FrozenEmbedding>,
    embed_a: Embedding,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    merged: bool,
    prefix: String,
    id: usize,
}

#[derive(Clone, Debug)]
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
        id: usize,
    ) -> Result<Self> {
        let a = vb.pp(format!("a{id}")).get_with_hints(
            (config.rank, embed_config.num_embeddings),
            "weight",
            init::ZERO,
        )?;
        let b: Tensor = vb.pp(format!("b{id}")).get_with_hints(
            (embed_config.embedding_dim, config.rank),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        let mut a_t = a.t()?;
        a_t = a_t.reshape(a_t.shape())?;
        let embed_a = Embedding::new(a_t.clone(), a_t.dim(1)?);

        Ok(LoraEmbedding {
            old: Arc::new(FrozenEmbedding::new_from_embed(old)?),
            embed_a,
            a,
            b,
            scale: if config.rank > 0 {
                Some(config.alpha / config.rank as f64)
            } else {
                None
            },
            merged: false,
            prefix: vb.prefix(),
            id,
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
            self.old = Arc::new(
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
            self.old = Arc::new(
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
            let b = self.b.t()?;
            let b = b.reshape(b.shape())?;

            let after_a = self.embed_a.forward(input)?;
            result = (result + (after_a.broadcast_matmul(&b)?).mul(scale))?
        }
        Ok(result)
    }
}

impl Saveable for LoraEmbedding {
    fn get_tensors(&self, accum: &mut HashMap<String, Tensor>) {
        accum.insert(
            self.prefix.clone() + &format!(".a{}.weight", self.id),
            self.a.clone(),
        );
        accum.insert(
            self.prefix.clone() + &format!(".b{}.weight", self.id),
            self.b.clone(),
        );
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
