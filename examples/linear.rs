use std::{collections::HashMap, hash::Hash};

use candle_core::{DType, Device, Result, Tensor};
use candle_lora::{
    LinearLayerLike, Lora, LoraConfig, LoraLinearConfig, NewLayers, SelectedLayersBuilder,
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
        for (name, linear) in new.linear {
            match name {
                ModelLayers::Layer => self.layer = Box::new(linear),
            }
        }
    }
}

fn main() {
    let device = Device::Cpu;
    let dtype = DType::F32;

    //Create the model
    let map = VarMap::new();
    let layer_weight = map
        .get(
            (10, 10),
            "layer.weight",
            init::DEFAULT_KAIMING_NORMAL,
            dtype,
            &device,
        )
        .unwrap();

    let mut model = Model {
        layer: Box::new(Linear::new(layer_weight.clone(), None)),
    };

    let mut linear_layers = HashMap::new();
    linear_layers.insert(ModelLayers::Layer, &*model.layer);
    let selected = SelectedLayersBuilder::new()
        .add_linear_layers(linear_layers, LoraLinearConfig::new(10, 10))
        .build();

    let loraconfig = LoraConfig::new(1, 1., None, &device, dtype);

    let new_layers = Lora::convert_model(selected, loraconfig);

    model.insert_new(new_layers);

    let dummy_image = Tensor::zeros((10, 10), DType::F32, &device).unwrap();

    let lora_output = model.forward(&dummy_image).unwrap();
    println!("Output: {lora_output:?}");
}
