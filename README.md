# candle-lora
[![MIT License](https://img.shields.io/badge/License-MIT-informational)](LICENSE)
[![Continuous integration](https://github.com/EricLBuehler/candle-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-lora/actions/workflows/ci.yml)

LoRA (low rank adaptation) implemented in Rust for use with [`Candle`](https://github.com/huggingface/candle/tree/main).

It is based on HuggingFace's [`peft`](https://github.com/huggingface/peft/tree/main) library. See the original paper [here](https://arxiv.org/pdf/2106.09685.pdf). 

All conversions are done as implemented in HuggingFace's official LoRA implementation.

Specifically, `candle-lora` is able to convert `Linear`, `Conv1d`, `Conv2d`, and `Embedding` into their respective LoRA counterparts. To improve inference performance, both merging and unmerging LoRA weights are also implemented.

## [candle-lora-macro](https://github.com/EricLBuehler/candle-lora-macro)
`candle-lora-macro` provides a derive macro that defines a method `get_lora_model`. It automates the process of selecting and swapping all `Box<dyn ...LayerLike>` layers. This aims to provide ergonomics similar to the `peft` `.get_peft_model` method.

In addition, `candle-lora-macro` also provides an attribute macro called `replace_layer_fields`. This replaces `Linear`, `Conv1d`, `Conv2d`, and `Embedding` concrete types with their `candle-lora` `Box<dyn ...LayerLike>` counterpart.

Together, these macros mean that `candle-lora` can be added to any `candle` model with minimal code changes!

## How to use
1) Derive `AutoLora` from candle-lora-macro on each model struct and add the `replace_layer_fields` attribute macro.
2) Call `get_lora_model` on each model struct.
3) Enjoy your new LoRA model!


## Example
```rust
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_lora::{LinearLayerLike, LoraConfig, LoraLinearConfig};
use candle_lora_macro::{replace_layer_fields, AutoLoraConvert};
use candle_nn::{init, Linear, VarMap};

#[replace_layer_fields]
#[derive(AutoLoraConvert, Debug)]
struct Model {
    a: Linear,
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
        a: Box::new(Linear::new(layer_weight.clone(), None)),
        b: 1,
    };

    let loraconfig = LoraConfig::new(1, 1., None, &device, dtype);
    model.get_lora_model(
        loraconfig,
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
    println!("{:?}", model.b);
}

```