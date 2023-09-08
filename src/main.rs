use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, linear_no_bias, VarBuilder};
use rlora::{LoraLayersModule, LayerType, Lora};
use trc::Trc;

#[derive(Debug)]
struct Model {
    layer: Trc<Linear>
}

impl Module for Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.layer.forward(input)
    }
}

impl LoraLayersModule for Model {
    fn get_layers(&self) -> HashMap<String, rlora::LayerType> {
        let mut layers = HashMap::new();
        layers.insert("layer".to_string(), LayerType::Linear(self.layer.clone()));
        layers
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    let mut tensors = HashMap::new();
    tensors.insert("1.weight".to_string(), Tensor::zeros((10,10), DType::F32, &device)?);

    let varbuilder = VarBuilder::from_tensors(tensors, DType::F32, &device);

    let model = Model { layer: Trc::new(linear_no_bias(10, 10, varbuilder.pp("1")).unwrap()) };

    let dummy_image = Tensor::zeros((10, 10), DType::F32, &device)?;

    let digit = model.forward(&dummy_image).unwrap();
    println!("Digit {digit:?} digit");

    let loramodel = Lora::get_lora_model(&model);
    
    let digit = loramodel.forward(&dummy_image).unwrap();
    println!("Loramodel {digit:?} digit");

    Ok(())
}
