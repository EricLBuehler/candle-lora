use candle_core::{DType, Device, Module, Result, Tensor};
use candle_lora::{LinearLayerLike, LoraConfig, LoraLinearConfig};
use candle_lora_macro::{replace_layer_fields, AutoLoraConvert};
use candle_nn::{init, Linear, VarBuilder, VarMap};
use std::sync::Arc;

#[replace_layer_fields]
#[derive(AutoLoraConvert, Debug)]
struct Model {
    a: Option<Linear>,
}

impl Module for Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.a.as_ref().unwrap().forward(input)
    }
}

#[test]
fn option_linear() {
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
        a: Some(Arc::new(Linear::new(layer_weight.clone(), None))),
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

    let dummy_image = Tensor::zeros((10, 10), DType::F32, &device).unwrap();

    //Test the model
    let digit = model.forward(&dummy_image).unwrap();
    println!("Output: {digit:?}");

    println!("{:?}", model.a);
}
