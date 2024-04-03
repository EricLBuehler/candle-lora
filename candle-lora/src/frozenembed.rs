use std::collections::HashMap;

use candle_core::{Result, Tensor};
use candle_nn::Embedding;

use crate::{EmbeddingLayerLike, Saveable};

/// Embedding, but with a `new` implementation that ensures the embeddings are detached (frozen).
#[derive(Debug)]
pub(crate) struct FrozenEmbedding {
    embed: Embedding,
}

impl FrozenEmbedding {
    pub(crate) fn new(embeddings: &Tensor, hidden_size: usize) -> Result<Self> {
        Ok(Self {
            embed: Embedding::new(embeddings.detach(), hidden_size),
        })
    }

    pub(crate) fn new_from_embed(old: &dyn EmbeddingLayerLike) -> Result<Self> {
        Self::new(old.embeddings(), old.hidden_size())
    }
}

impl crate::Module for FrozenEmbedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        self.embed.forward(indexes)
    }
}

impl Saveable for FrozenEmbedding {
    fn get_tensors(&self, _accum: &mut HashMap<String, Tensor>) {
        unimplemented!("Saving not supported for frozen layers, only for candle_lora layers.");
    }
}

impl EmbeddingLayerLike for FrozenEmbedding {
    fn embeddings(&self) -> &Tensor {
        self.embed.embeddings()
    }
    fn hidden_size(&self) -> usize {
        self.embed.hidden_size()
    }
}
