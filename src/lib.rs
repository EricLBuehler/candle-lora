//According to https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
#[doc = include_str!("../README.md")]
use candle_core::{Shape, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Linear, Module};
use loraconv1d::{LoraConv1D, LoraConv1DConfig};
use loraconv2d::{LoraConv2D, LoraConv2DConfig};
use loralinear::{LoraLinear, LoraLinearConfig};
use std::{collections::HashMap, hash::Hash};

mod frozenconv;
mod frozenlinear;
pub mod loraconv1d;
pub mod loraconv2d;
pub mod loralinear;

pub struct Lora;

impl Lora {
    pub fn convert_model<T: Eq + PartialEq + Hash>(
        selected: SelectedLayers<'_, T>,
    ) -> NewLayers<T> {
        let mut new = NewLayers {
            linear: HashMap::new(),
            conv1d: HashMap::new(),
            conv2d: HashMap::new(),
        };

        for (name, layer) in selected.linear {
            new.linear.insert(
                name,
                LoraLinear::new(layer, selected.linear_config.as_ref().unwrap()).unwrap(),
            );
        }

        for (name, layer) in selected.conv1d {
            new.conv1d.insert(
                name,
                LoraConv1D::new(layer, selected.conv1d_config.as_ref().unwrap()).unwrap(),
            );
        }

        for (name, layer) in selected.conv2d {
            new.conv2d.insert(
                name,
                LoraConv2D::new(layer, selected.conv2d_config.as_ref().unwrap()).unwrap(),
            );
        }

        new
    }
}

pub struct SelectedLayers<'a, T: Eq + PartialEq + Hash> {
    pub linear: HashMap<T, &'a dyn LinearLayerLike>,
    pub linear_config: Option<LoraLinearConfig<'a>>,
    pub conv1d: HashMap<T, &'a dyn Conv1dLayerLike>,
    pub conv1d_config: Option<LoraConv1DConfig<'a>>,
    pub conv2d: HashMap<T, &'a dyn Conv2dLayerLike>,
    pub conv2d_config: Option<LoraConv2DConfig<'a>>,
}

pub struct NewLayers<T: Eq + PartialEq + Hash> {
    pub linear: HashMap<T, LoraLinear>,
    pub conv1d: HashMap<T, LoraConv1D>,
    pub conv2d: HashMap<T, LoraConv2D>,
}

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

pub trait Conv2dLayerLike: Module {
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
    fn config(&self) -> &Conv2dConfig;
}

#[derive(Debug)]
pub struct Conv2DWithWB {
    pub this: Conv2d,
    pub weights: Tensor,
    pub bias: Option<Tensor>,
}

impl Module for Conv2DWithWB {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.this.forward(xs)
    }
}

impl Conv2dLayerLike for Conv2DWithWB {
    fn config(&self) -> &Conv2dConfig {
        self.this.config()
    }
    fn weight(&self) -> &Tensor {
        &self.weights
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}
