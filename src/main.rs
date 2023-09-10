use std::{collections::HashMap, hash::Hash};

use candle_core::{DType, Device, Result, Tensor};
use candle_lora::{
    loralinear::{LoraLinear, LoraLinearConfig},
    Conv1DWithWB, Conv1dLayerLike, LinearLayerLike, Lora,
};
use candle_nn::{init, Conv1d, Conv1dConfig, Linear, Module, VarMap};

#[derive(PartialEq, Eq, Hash)]
enum ModelLayers {
    Layer,
}

#[derive(Debug)]
struct Model {
    layer: Box<dyn LinearLayerLike>,
    conv: Box<dyn Conv1dLayerLike>,
}

impl Module for Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.conv.forward(input)
        //self.layer.forward(input)
    }
}

impl Model {
    fn insert_loralinear(&mut self, layers: HashMap<ModelLayers, LoraLinear>) {
        for (name, layer) in layers {
            match name {
                ModelLayers::Layer => {
                    self.layer = Box::new(layer);
                }
            }
        }
    }
}

fn main() -> Result<()> {
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

    let conv_weight = map.get(
        (1, 10, 10),
        "conv.weight",
        init::DEFAULT_KAIMING_NORMAL,
        dtype,
        &device,
    )?;
    let conv_bias = map.get(
        10,
        "conv.bias",
        init::DEFAULT_KAIMING_NORMAL,
        dtype,
        &device,
    )?;

    let conv = Conv1DWithWB {
        this: Conv1d::new(
            conv_weight.clone(),
            Some(conv_bias.clone()),
            Conv1dConfig::default(),
        ),
        weights: conv_weight,
        bias: Some(conv_bias),
    };

    let mut model = Model {
        layer: Box::new(Linear::new(layer_weight.clone(), None)),
        conv: Box::new(conv),
    };

    let dummy_image = Tensor::zeros((1, 10, 10), DType::F32, &device)?;

    //Test the model
    let digit = model.forward(&dummy_image).unwrap();
    println!("Output: {digit:?}");

    //Isolate layers we want to convert
    let mut linear_layers = HashMap::new();
    linear_layers.insert(ModelLayers::Layer, &*model.layer);

    //Create new LoRA layers from our layers
    let new_layers = Lora::convert_model(
        linear_layers,
        LoraLinearConfig::default(&device, dtype),
        10,
        10,
    );

    //Custom methods to implement
    model.insert_loralinear(new_layers);

    //Test the model
    let digit = model.forward(&dummy_image).unwrap();
    println!("LoRA Output: {digit:?}");

    Ok(())
}
