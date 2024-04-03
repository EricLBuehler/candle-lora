use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_lora::{LinearLayerLike, LoraConfig, LoraLinearConfig};
use candle_lora_macro::{replace_layer_fields, AutoLoraConvert};
use candle_nn::{init, Linear, VarBuilder, VarMap};

#[replace_layer_fields]
#[derive(AutoLoraConvert, Debug)]
struct Model {
    a: Arc<dyn LinearLayerLike>,
    b: i32,
}

impl Module for Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.a.forward(input)
    }
}

fn main() {
    let device = Device::Cpu;
    let dtype = DType::F32;

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
        a: Arc::new(Linear::new(layer_weight.clone(), None)),
        b: 1,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    let loraconfig = LoraConfig::new(1, 1., None);
    model.get_lora_model(
        loraconfig,
        &vb,
        Some(LoraLinearConfig::new(10, 10)),
        None,
        None,
        None,
    );

    dbg!(model.get_tensors());

    let dummy_image = Tensor::zeros((10, 10), DType::F32, &device).unwrap();

    //Test the model
    let digit = model.forward(&dummy_image).unwrap();
    println!("Output: {digit:?}");

    println!("{:?}", model.a);
    println!("{:?}", model.b);
}
