use std::sync::Arc;

use candle_lora::{LoraConfig, SelectedLayersBuilder};
use candle_nn::VarBuilder;

#[test]
fn conv1d() -> candle_core::Result<()> {
    use std::{collections::HashMap, hash::Hash};

    use candle_core::{DType, Device, Result, Tensor};
    use candle_lora::{Conv1dLayerLike, Lora, LoraConv1dConfig, NewLayers};
    use candle_nn::{init, Conv1d, Conv1dConfig, Module, VarMap};

    #[derive(PartialEq, Eq, Hash)]
    enum ModelLayers {
        Conv,
    }

    #[derive(Debug)]
    struct Model {
        conv: Arc<dyn Conv1dLayerLike>,
    }

    impl Module for Model {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            self.conv.forward(input)
        }
    }

    impl Model {
        fn insert_new(&mut self, new: NewLayers<ModelLayers>) {
            for (name, conv) in new.conv1d {
                match name {
                    ModelLayers::Conv => self.conv = Arc::new(conv),
                }
            }
        }
    }

    let device = Device::Cpu;
    let dtype = DType::F32;

    //Create the model
    let map = VarMap::new();
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

    let mut model = Model {
        conv: Arc::new(Conv1d::new(
            conv_weight.clone(),
            Some(conv_bias.clone()),
            Conv1dConfig::default(),
        )),
    };

    let dummy_image = Tensor::zeros((1, 10, 10), DType::F32, &device)?;

    //Test the model
    let output = model.forward(&dummy_image).unwrap();
    println!("Output: {output:?}");

    //Select layers we want to convert
    let mut conv1d_layers = HashMap::new();
    conv1d_layers.insert(ModelLayers::Conv, &*model.conv);
    let selected = SelectedLayersBuilder::new()
        .add_conv1d_layers(conv1d_layers, LoraConv1dConfig::new(1, 10, 10))
        .build();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    let loraconfig = LoraConfig::new(1, 1., None);

    //Create new LoRA layers from our layers
    let new_layers = Lora::convert_model(selected, loraconfig, &vb);

    //Custom methods to implement
    model.insert_new(new_layers);

    //Test the model
    let lora_output = model.forward(&dummy_image).unwrap();
    println!("LoRA Output: {lora_output:?}");

    assert_eq!(lora_output.shape(), output.shape());

    Ok(())
}
