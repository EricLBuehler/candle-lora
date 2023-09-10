#[doc = include_str!("../README.md")]
use candle_core::{Shape, Tensor};
use candle_nn::{Linear, Module};
use loralinear::{LoraLinear, LoraLinearConfig};
use std::{collections::HashMap, hash::Hash};

pub mod loralinear;
mod frozenlinear;

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
