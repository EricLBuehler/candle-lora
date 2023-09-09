use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use candle_lora::{
    LinearLayerLike, Lora,
};
use candle_nn::{linear_no_bias, Module, VarBuilder};

#[derive(PartialEq, Eq, Hash)]
enum ModelLayers {
    Layer
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

fn main() -> Result<()> {
    let device = Device::Cpu;

    let mut vars = HashMap::new();
    vars.insert(
        "1.weight".to_string(),
        Tensor::zeros((10, 10), DType::F32, &device)?,
    );

    let varbuilder = VarBuilder::from_tensors(vars, DType::F32, &device);

    let mut model = Model {
        layer: Box::new(linear_no_bias(10, 10, varbuilder.pp("1")).unwrap()),
    };

    let dummy_image = Tensor::zeros((10, 10), DType::F32, &device)?;

    let digit = model.forward(&dummy_image).unwrap();
    println!("Output: {digit:?}");

    let mut layers = HashMap::new();
    layers.insert(ModelLayers::Layer, &*model.layer);

    let new_layers = Lora::convert_model(layers, &device);

    for (name, layer) in new_layers{
        match name {
            ModelLayers::Layer => {
                model.layer = Box::new(layer);
            }
        }
    }

    let digit = model.forward(&dummy_image).unwrap();
    println!("LoRA Output: {digit:?}");

    Ok(())
}
