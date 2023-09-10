# candle-lora
[![MIT License](https://img.shields.io/badge/License-MIT-informational)](LICENSE)
[![Continuous integration](https://github.com/EricLBuehler/candle-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-lora/actions/workflows/ci.yml)

LoRA implemented in Rust for use with Candle.

Current working example:
```rust
use std::{collections::HashMap, hash::Hash};

use candle_core::{DType, Device, Result, Tensor};
use candle_lora::{
    loralinear::LoraLinearConfig,
    LinearLayerLike, Lora, NewLayers, SelectedLayers,
};
use candle_nn::{init, Linear, Module, VarMap};

#[derive(PartialEq, Eq, Hash)]
enum ModelLayers {
    Layer,
}

#[derive(Debug)]
struct Model {
    layer: Box<dyn LinearLayerLike>,
}

impl Module for Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.layer.forward(input)
    }
}

impl Model {
    fn insert_new(&mut self, new: NewLayers<ModelLayers>) {
        for (name, conv) in new.linear {
            match name {
                ModelLayers::Layer => self.layer = Box::new(conv),
            }
        }
    }
}

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    //Create the model
    let map = VarMap::new();
    let layer_weight = map.get(
        (10, 10),
        "layer.weight",
        init::DEFAULT_KAIMING_NORMAL,
        dtype,
        &device,
    )?;

    let mut model = Model {
        layer: Box::new(Linear::new(layer_weight.clone(), None)),
    };

    let dummy_image = Tensor::zeros((10, 10), DType::F32, &device)?;

    //Test the model
    let digit = model.forward(&dummy_image).unwrap();
    println!("Output: {digit:?}");

    //Select layers we want to convert
    let mut linear_layers = HashMap::new();
    linear_layers.insert(ModelLayers::Layer, &*model.layer);
    let conv1d_layers = HashMap::new();
    let conv2d_layers = HashMap::new();
    let selected = SelectedLayers {
        linear: linear_layers,
        linear_config: Some(LoraLinearConfig::default(&device, dtype, 10, 10)),
        conv1d: conv1d_layers,
        conv1d_config: None,
        conv2d: conv2d_layers,
        conv2d_config: None,
    };

    //Create new LoRA layers from our layers
    let new_layers = Lora::convert_model(selected);

    //Custom methods to implement
    model.insert_new(new_layers);

    //Test the model
    let digit = model.forward(&dummy_image).unwrap();
    println!("LoRA Output: {digit:?}");

    Ok(())
}
```