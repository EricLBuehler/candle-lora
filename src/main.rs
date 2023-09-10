use std::{collections::HashMap, hash::Hash};

use candle_core::{DType, Device, Result, Tensor};
use candle_lora::{
    loralinear::{LoraLinear, LoraLinearConfig, ALPHA_DEFAULT},
    LinearLayerLike, Lora,
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

    let mut model = Model {
        layer: Box::new(Linear::new(layer_weight.clone(), None)),
    };

    let dummy_image = Tensor::zeros((10, 10), DType::F32, &device)?;

    //Test the model
    let digit = model.forward(&dummy_image).unwrap();
    println!("Output: {digit:?}");

    //Isolate layers we want to convert
    let mut layers = HashMap::new();
    layers.insert(ModelLayers::Layer, &*model.layer);

    //Create new LoRA layers from our layers
    let new_layers = Lora::convert_model(
        layers,
        LoraLinearConfig {
            rank: layer_weight.rank(),
            alpha: ALPHA_DEFAULT,
            dropout: Some(0.),
            device: &device,
            dtype,
        },
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
