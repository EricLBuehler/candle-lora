use std::sync::Arc;

use candle_lora::{LoraConfig, Merge, NewLayers, SelectedLayersBuilder};
use candle_nn::VarBuilder;

#[test]
fn linear() -> candle_core::Result<()> {
    use std::{collections::HashMap, hash::Hash};

    use candle_core::{DType, Device, Result, Tensor};
    use candle_lora::{LinearLayerLike, Lora, LoraLinearConfig};
    use candle_nn::{init, Linear, Module, VarMap};

    #[derive(PartialEq, Eq, Hash)]
    enum ModelLayers {
        Layer,
    }

    #[derive(Debug)]
    struct Model {
        layer: Arc<dyn LinearLayerLike>,
    }

    impl Module for Model {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            self.layer.forward(input)
        }
    }

    impl Model {
        fn insert_new(&mut self, new: NewLayers<ModelLayers>) {
            for (name, mut linear) in new.linear {
                match name {
                    ModelLayers::Layer => {
                        linear.merge_weights().unwrap();
                        self.layer = Arc::new(linear)
                    }
                }
            }
        }
    }

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
        layer: Arc::new(Linear::new(layer_weight.clone(), None)),
    };

    let dummy_image = Tensor::zeros((10, 10), DType::F32, &device)?;

    //Test the model
    let output = model.forward(&dummy_image).unwrap();
    println!("Output: {output:?}");

    //Select layers we want to convert
    let mut linear_layers = HashMap::new();
    linear_layers.insert(ModelLayers::Layer, &*model.layer);
    let selected = SelectedLayersBuilder::new()
        .add_linear_layers(linear_layers, LoraLinearConfig::new(10, 10))
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
