use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use candle_lora::{
    loralinear::{LoraLinear, ALPHA_DEFAULT},
    LayerType, Lora, LoraLayersModule,
};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use trc::Trc;

#[derive(Debug)]
struct Model {
    layer: Trc<Linear>,
}

impl Module for Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.layer.forward(input)
    }
}

impl LoraLayersModule for Model {
    fn get_layers(&self) -> HashMap<String, LayerType> {
        let mut layers = HashMap::new();
        layers.insert("layer".to_string(), LayerType::Linear(self.layer.clone()));
        layers
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

    let model = Model {
        layer: Trc::new(linear_no_bias(10, 10, varbuilder.pp("1")).unwrap()),
    };

    let dummy_image = Tensor::zeros((10, 10), DType::F32, &device)?;

    let digit = model.forward(&dummy_image).unwrap();
    println!("Digit {digit:?} digit");

    LoraLinear::new(
        model.layer.clone(),
        model.layer.weight().rank(),
        ALPHA_DEFAULT,
        &device,
    )?;

    Lora::get_lora_model(&model);

    //let digit = loramodel.forward(&dummy_image).unwrap();
    //println!("Loramodel {digit:?} digit");

    Ok(())
}
