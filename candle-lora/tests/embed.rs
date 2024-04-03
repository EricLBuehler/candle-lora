use std::sync::Arc;

use candle_lora::{LoraConfig, LoraEmbeddingConfig, SelectedLayersBuilder};
use candle_nn::VarBuilder;

#[test]
fn embed() -> candle_core::Result<()> {
    use std::{collections::HashMap, hash::Hash};

    use candle_core::{DType, Device, Result, Tensor};
    use candle_lora::{EmbeddingLayerLike, Lora, NewLayers};
    use candle_nn::{init, Embedding, Module, VarMap};

    #[derive(PartialEq, Eq, Hash)]
    enum ModelLayers {
        Embed,
    }

    #[derive(Debug)]
    struct Model {
        embed: Arc<dyn EmbeddingLayerLike>,
    }

    impl Module for Model {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            self.embed.forward(input)
        }
    }

    impl Model {
        fn insert_new(&mut self, new: NewLayers<ModelLayers>) {
            for (name, embed) in new.embed {
                match name {
                    ModelLayers::Embed => self.embed = Arc::new(embed),
                }
            }
        }
    }

    let device = Device::Cpu;
    let dtype = DType::F32;

    let in_size = 10;
    let hidden_size = 3;

    //Create the model
    let map = VarMap::new();
    let embed_weight = map.get(
        (in_size, hidden_size),
        "embed.weight",
        init::ZERO,
        dtype,
        &device,
    )?;

    let mut model = Model {
        embed: Arc::new(Embedding::new(embed_weight, hidden_size)),
    };

    let dummy_image = Tensor::zeros((2, 4), DType::U32, &device)?;

    //Test the model
    let output = model.forward(&dummy_image).unwrap();
    println!("Output: {output:?}");

    //Select layers we want to convert
    let mut embed_layers = HashMap::new();
    embed_layers.insert(ModelLayers::Embed, &*model.embed);
    let selected = SelectedLayersBuilder::new()
        .add_embed_layers(embed_layers, LoraEmbeddingConfig::new(in_size, hidden_size))
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
