# candle-lora
[![MIT License](https://img.shields.io/badge/License-MIT-informational)](LICENSE)
[![Continuous integration](https://github.com/EricLBuehler/candle-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-lora/actions/workflows/ci.yml)

LoRA (low rank adaptation) implemented in Rust for use with [`Candle`](https://github.com/huggingface/candle/tree/main).

It is based on HuggingFace's [`peft`](https://github.com/huggingface/peft/tree/main) library. See the original paper [here](https://arxiv.org/pdf/2106.09685.pdf). 

All conversions are done as implemented in HuggingFace's official LoRA implementation.

Specifically, `candle-lora` is able to convert `Linear`, `Conv1d`, `Conv2d`, and `Embedding` into their respective LoRA counterparts. To improve inference performance, both merging and unmerging LoRA weights are also implemented.

## [candle-lora-macro](https://github.com/EricLBuehler/candle-lora-macro)
This library makes using `candle-lora` as simple as adding 2 macros to your model structs and calling a method! It is inspired by the simplicity of the Python `peft` library's `get_peft_model` method. 
Together, these macros mean that `candle-lora` can be added to any `candle` model with minimal code changes! To see an example of the benefits, compare the example below (or [here](examples/linear_macro.rs)) to [this](examples/linear.rs), equivalent example. See a precise diff [here](examples/macro_diff.txt).

## How to use
1) Derive `AutoLoraConvert` from candle-lora-macro on each model struct and add the `replace_layer_fields` attribute macro.
2) Call `get_lora_model` on each model struct.
3) Enjoy your new LoRA model!


## Examples
See a training example with Llama + LoRA [here](examples/llama).

```rust
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_lora::{LinearLayerLike, LoraConfig, LoraLinearConfig};
use candle_lora_macro::{replace_layer_fields, AutoLoraConvert};
use candle_nn::{init, Linear, VarBuilder, VarMap};

#[replace_layer_fields]
#[derive(AutoLoraConvert, Debug)]
struct Model {
    layer: Linear,
}

impl Module for Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.layer.forward(input)
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
        layer: Box::new(Linear::new(layer_weight.clone(), None)),
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

    let digit = model.forward(&dummy_image).unwrap();
    println!("Output: {digit:?}");
}
```