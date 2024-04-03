use std::sync::Arc;

use candle_lora::{LoraConfig, LoraConv2dConfig, SelectedLayersBuilder};
use candle_nn::VarBuilder;

#[test]
fn conv2d() -> candle_core::Result<()> {
    use std::{collections::HashMap, hash::Hash};

    use candle_core::{DType, Device, Result, Tensor};
    use candle_lora::{Conv2dLayerLike, Lora, NewLayers};
    use candle_nn::{init, Conv2d, Conv2dConfig, Module, VarMap};

    #[derive(PartialEq, Eq, Hash)]
    enum ModelLayers {
        Conv,
    }

    #[derive(Debug)]
    struct Model {
        conv: Arc<dyn Conv2dLayerLike>,
    }

    impl Module for Model {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            self.conv.forward(input)
        }
    }

    impl Model {
        fn insert_new(&mut self, new: NewLayers<ModelLayers>) {
            for (name, conv) in new.conv2d {
                match name {
                    ModelLayers::Conv => self.conv = Arc::new(conv),
                }
            }
        }
    }

    let device = Device::Cpu;
    let dtype = DType::F32;

    let out_channels = 10;
    let in_channels = 10;
    let kernel = 2;

    let cfg = Conv2dConfig::default();

    //Create the model
    let map = VarMap::new();
    let conv_weight = map.get(
        (
            out_channels,
            in_channels / cfg.groups, //cfg.groups in this case are 1
            kernel,
            kernel,
        ),
        "conv.weight",
        init::DEFAULT_KAIMING_NORMAL,
        dtype,
        &device,
    )?;
    let conv_bias = map.get(
        out_channels,
        "conv.bias",
        init::DEFAULT_KAIMING_NORMAL,
        dtype,
        &device,
    )?;

    let mut model = Model {
        conv: Arc::new(Conv2d::new(
            conv_weight.clone(),
            Some(conv_bias.clone()),
            cfg,
        )),
    };

    let shape = [2, in_channels, 20, 20]; //(BS, K, X, Y)
    let dummy_image = Tensor::zeros(&shape, DType::F32, &device)?;

    //Test the model
    let output = model.forward(&dummy_image).unwrap();
    println!("Output: {output:?}");

    //Select layers we want to convert
    let mut conv2d_layers = HashMap::new();
    conv2d_layers.insert(ModelLayers::Conv, &*model.conv);
    let selected = SelectedLayersBuilder::new()
        .add_conv2d_layers(
            conv2d_layers,
            LoraConv2dConfig::new(in_channels, out_channels),
        )
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
