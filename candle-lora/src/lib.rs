use candle_core::{Error, Shape, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Embedding, Linear, Module, VarBuilder,
};
use either::Either;
pub use loraconv1d::{LoraConv1d, LoraConv1dConfig};
pub use loraconv2d::{LoraConv2d, LoraConv2dConfig};
pub use loraembed::{LoraEmbedding, LoraEmbeddingConfig};
pub use loralinear::{LoraLinear, LoraLinearConfig};
use std::{collections::HashMap, fmt::Debug, hash::Hash};
use thiserror::Error;

mod frozenconv;
mod frozenembed;
mod frozenlinear;
mod loraconv1d;
mod loraconv2d;
mod loraembed;
mod loralinear;

pub struct Lora;

impl Lora {
    /// Convert the selected layers into their LoRA counterparts.
    pub fn convert_model<T: Eq + PartialEq + Hash>(
        selected: SelectedLayers<'_, T>,
        config: LoraConfig,
        vb: &VarBuilder,
    ) -> NewLayers<T> {
        let mut new = NewLayers {
            linear: HashMap::new(),
            conv1d: HashMap::new(),
            conv2d: HashMap::new(),
            embed: HashMap::new(),
        };

        let mut id = 0;

        for (name, layer) in selected.linear {
            new.linear.insert(
                name,
                LoraLinear::new(
                    layer,
                    selected.linear_config.as_ref().unwrap(),
                    &config,
                    vb,
                    id,
                )
                .unwrap(),
            );
            id += 1;
        }

        for (name, layer) in selected.conv1d {
            new.conv1d.insert(
                name,
                LoraConv1d::new(
                    layer,
                    selected.conv1d_config.as_ref().unwrap(),
                    &config,
                    vb,
                    id,
                )
                .unwrap(),
            );
            id += 1;
        }

        for (name, layer) in selected.conv2d {
            new.conv2d.insert(
                name,
                LoraConv2d::new(
                    layer,
                    selected.conv2d_config.as_ref().unwrap(),
                    &config,
                    vb,
                    id,
                )
                .unwrap(),
            );
            id += 1;
        }

        for (name, layer) in selected.embed {
            new.embed.insert(
                name,
                LoraEmbedding::new(
                    layer,
                    selected.embed_config.as_ref().unwrap(),
                    &config,
                    vb,
                    id,
                )
                .unwrap(),
            );
            id += 1;
        }

        new
    }
}

#[derive(Clone, Debug)]
pub struct LoraConfig {
    rank: usize,
    alpha: f64,
    dropout: Option<f32>,
}

impl LoraConfig {
    /// Create a new LoRA config.
    /// - `rank`: The dimensions of low-rank matrices.
    /// - `alpha`: Scaling factor for the LoRA signal.
    /// - `dropout`: Dropout probability for the LoRA layers.
    pub const fn new(rank: usize, alpha: f64, dropout: Option<f32>) -> Self {
        Self {
            rank,
            alpha,
            dropout,
        }
    }
}

pub struct SelectedLayers<'a, T: Eq + PartialEq + Hash> {
    linear: HashMap<T, &'a dyn LinearLayerLike>,
    linear_config: Option<LoraLinearConfig>,
    conv1d: HashMap<T, &'a dyn Conv1dLayerLike>,
    conv1d_config: Option<LoraConv1dConfig>,
    conv2d: HashMap<T, &'a dyn Conv2dLayerLike>,
    conv2d_config: Option<LoraConv2dConfig>,
    embed: HashMap<T, &'a dyn EmbeddingLayerLike>,
    embed_config: Option<LoraEmbeddingConfig>,
}

pub struct SelectedLayersBuilder<'a, T: Eq + PartialEq + Hash> {
    selected: SelectedLayers<'a, T>,
}

impl<T: Eq + PartialEq + Hash> Default for SelectedLayersBuilder<'_, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: Eq + PartialEq + Hash> SelectedLayersBuilder<'a, T> {
    pub fn new() -> Self {
        Self {
            selected: SelectedLayers {
                linear: HashMap::new(),
                linear_config: None,
                conv1d: HashMap::new(),
                conv1d_config: None,
                conv2d: HashMap::new(),
                conv2d_config: None,
                embed: HashMap::new(),
                embed_config: None,
            },
        }
    }

    pub fn add_linear_layers(
        mut self,
        layers: HashMap<T, &'a dyn LinearLayerLike>,
        linear_config: LoraLinearConfig,
    ) -> Self {
        self.selected.linear = layers;
        self.selected.linear_config = Some(linear_config);
        self
    }

    pub fn add_embed_layers(
        mut self,
        layers: HashMap<T, &'a dyn EmbeddingLayerLike>,
        embed_config: LoraEmbeddingConfig,
    ) -> Self {
        self.selected.embed = layers;
        self.selected.embed_config = Some(embed_config);
        self
    }

    pub fn add_conv1d_layers(
        mut self,
        layers: HashMap<T, &'a dyn Conv1dLayerLike>,
        conv1d_config: LoraConv1dConfig,
    ) -> Self {
        self.selected.conv1d = layers;
        self.selected.conv1d_config = Some(conv1d_config);
        self
    }

    pub fn add_conv2d_layers(
        mut self,
        layers: HashMap<T, &'a dyn Conv2dLayerLike>,
        conv2d_config: LoraConv2dConfig,
    ) -> Self {
        self.selected.conv2d = layers;
        self.selected.conv2d_config = Some(conv2d_config);
        self
    }

    pub fn build(self) -> SelectedLayers<'a, T> {
        self.selected
    }
}

/// New layers, after conversion
pub struct NewLayers<T: Eq + PartialEq + Hash> {
    pub linear: HashMap<T, LoraLinear>,
    pub conv1d: HashMap<T, LoraConv1d>,
    pub conv2d: HashMap<T, LoraConv2d>,
    pub embed: HashMap<T, LoraEmbedding>,
}

pub trait Saveable {
    fn get_tensors(&self, accum: &mut HashMap<String, Tensor>);
}

/// Any layer that is linear-like.
pub trait LinearLayerLike: Module + Debug + Saveable + Send + Sync {
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
    fn shape(&self) -> &Shape;
}

impl Saveable for Linear {
    fn get_tensors(&self, _accum: &mut HashMap<String, Tensor>) {
        unimplemented!("Saving not supported for candle_nn layers, only for candle_lora layers.");
    }
}

impl LinearLayerLike for Linear {
    fn weight(&self) -> &Tensor {
        self.weight()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
    fn shape(&self) -> &Shape {
        self.weight().shape()
    }
}

/// Any layer that is conv1d-like.
pub trait Conv1dLayerLike: Module + Debug + Saveable + Send + Sync {
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
    fn config(&self) -> &Conv1dConfig;
}

impl Saveable for Conv1d {
    fn get_tensors(&self, _accum: &mut HashMap<String, Tensor>) {
        unimplemented!("Saving not supported for candle_nn layers, only for candle_lora layers.");
    }
}

impl Conv1dLayerLike for Conv1d {
    fn config(&self) -> &Conv1dConfig {
        self.config()
    }
    fn weight(&self) -> &Tensor {
        self.weight()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
}

/// Any layer that is conv2d-like.
pub trait Conv2dLayerLike: Module + Debug + Saveable + Send + Sync {
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
    fn config(&self) -> &Conv2dConfig;
}

impl Saveable for Conv2d {
    fn get_tensors(&self, _accum: &mut HashMap<String, Tensor>) {
        unimplemented!("Saving not supported for candle_nn layers, only for candle_lora layers.");
    }
}

impl Conv2dLayerLike for Conv2d {
    fn config(&self) -> &Conv2dConfig {
        self.config()
    }
    fn weight(&self) -> &Tensor {
        self.weight()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
}

/// Any layer that is embedding-like.
pub trait EmbeddingLayerLike: Module + Debug + Saveable + Send + Sync {
    fn embeddings(&self) -> &Tensor;
    fn hidden_size(&self) -> usize;
}

impl Saveable for Embedding {
    fn get_tensors(&self, _accum: &mut HashMap<String, Tensor>) {
        unimplemented!("Saving not supported for candle_nn layers, only for candle_lora layers.");
    }
}

impl EmbeddingLayerLike for Embedding {
    fn embeddings(&self) -> &Tensor {
        self.embeddings()
    }
    fn hidden_size(&self) -> usize {
        self.embeddings().dim(1).unwrap() //Reason: 2nd dim is always the hidden
    }
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum MergeError {
    #[error("AlreadyMerged")]
    AlreadyMerged,
    #[error("NotMerged")]
    NotMerged,
}

pub type MergeErrorOrError = Either<MergeError, Error>;

pub trait Merge {
    /// Get the delta weight of the LoRA layer. This is meant to be an internal method.
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError>;
    /// Merge the LoRA weights.
    fn merge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError>;
    /// Unmerge the LoRA weights.
    fn unmerge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError>;
}
