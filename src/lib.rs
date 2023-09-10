//According to https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
#[doc = include_str!("../README.md")]
use candle_core::{Shape, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module};
use loralinear::{LoraLinear, LoraLinearConfig};
use std::{collections::HashMap, hash::Hash};

mod frozenconv;
mod frozenlinear;
pub mod loraconv1d;
pub mod loralinear;

pub struct Lora;

impl Lora {
    pub fn convert_model<T: Eq + PartialEq + Hash>(
        layers: HashMap<T, &dyn LinearLayerLike>,
        metadata: LoraLinearConfig,
        in_features: usize,
        out_features: usize,
    ) -> HashMap<T, LoraLinear> {
        let mut output = HashMap::new();
        for (name, layer) in layers {
            output.insert(
                name,
                LoraLinear::new(layer, &metadata, in_features, out_features).unwrap(),
            );
        }
        output
    }
}

pub type Vars = HashMap<String, Tensor>;

pub trait LinearLayerLike: Module {
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
    fn shape(&self) -> &Shape;
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

pub trait Conv1dLayerLike: Module {
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
    fn config(&self) -> &Conv1dConfig;
}

#[derive(Debug)]
pub struct Conv1DWithWB {
    pub this: Conv1d,
    pub weights: Tensor,
    pub bias: Option<Tensor>,
}

impl Module for Conv1DWithWB {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.this.forward(xs)
    }
}

impl Conv1dLayerLike for Conv1DWithWB {
    fn config(&self) -> &Conv1dConfig {
        self.this.config()
    }
    fn weight(&self) -> &Tensor {
        &self.weights
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}
