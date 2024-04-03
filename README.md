# candle-lora
[![MIT License](https://img.shields.io/badge/License-MIT-informational)](LICENSE)
[![Continuous integration](https://github.com/EricLBuehler/candle-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-lora/actions/workflows/ci.yml)
[![Documentation](https://github.com/EricLBuehler/candle-lora/actions/workflows/docs.yml/badge.svg)](https://ericlbuehler.github.io/candle-lora/candle_lora/)

LoRA (low rank adaptation) implemented in Rust for use with [`Candle`](https://github.com/huggingface/candle/tree/main). This technique
interchanges the fully-trainable layers of the model with new, LoRA layers. These LoRA layers act as a wrapper over the original layers, but freeze
the original layers. Because they contain fewer trainable parameters, LoRA allows for more efficient fine-tuning. 

However, using a fine-tuned LoRA model for inference will have a negative impact on performance. This is because the original layer must still be used to calculate the outputs. However, for a LoRA model, an algorithm known as weight merging nullifies the added cost of using the
fine-tuned LoRA model by merging the LoRA and original weights. Weights may also be unmerged.

Please see our recent paper [X-LoRA](https://github.com/EricLBuehler/xlora). We introduce a MoE inspired method to densely gate LoRA adapters powered by a model self-reflection forward pass. For inference, we have created [mistral.rs](https://github.com/EricLBuehler/mistral.rs), which is written in Rust and enables inference of X-LoRA and other models including quantized.

## Get started
1) To install, run the following:
```
cargo add --git https://github.com/EricLBuehler/candle-lora.git candle-lora candle-lora-macro
```

2) To allow `candle-lora` to swap layers, do the following for each model struct
    - Derive `AutoLoraConvert` from `candle-lora-macro`
    - Add the `replace_layer_fields` attribute macro.
3) During instantiation of each model struct, call `get_lora_model` with the appropriate parameters to convert.

## Features
- Convert `Linear`, `Conv1d`, `Conv2d`, `Embedding` layers into LoRA layers
    - All conversions are implemented in accordance with HuggingFace's official LoRA implementation
- Weight merging is implemented to improve inference performance
- Weight unmerging
- Easy-to-use APIs
- Extensible trait-based layer swapping mechanism

## Conversion Ergonomics
`candle-lora-macro` makes using `candle-lora` as simple as adding 2 macros to your model structs and calling a method!

It is inspired by the simplicity of the Python `peft` library's `get_peft_model` method. 
Together, these macros mean that `candle-lora` can be added to any `candle` model with minimal code changes!

## LoRA transformers
See transformers from Candle which have LoRA integrated [here](candle-lora-transformers/examples/). Currently, the following
transformers have been converted:
- `llama`
- `mistral`
- `falcon`
- `bert`
- `stable_lm`
- `t5`
- `dinov2` 
- `resnet`
- `mpt`
- `blip`
- `starcoder`
    
To use a LoRA transformer, simply replace the model from `candle-transformers` with its counterpart in `candle-lora-transformers`!

## Saving and loading
`candle_lora` supports retrieving weights for LoRA adapters via the `get_tensors` method, defined automatically in `#[auto_layer_convert]`. This function is meant to be used with `candle_core::safetensors::save()`. To load, simply load the `VarBuilder` and pass that to `get_lora_model`.

`candle_lora`'s weight naming is not compatible with `peft` yet.

## Resources
`candle-lora`'s LoRA conversion implementations are based on HuggingFace's [`peft`](https://github.com/huggingface/peft/tree/main) library. See the original paper [here](https://arxiv.org/pdf/2106.09685.pdf), as well as Microsoft's [implementation](https://github.com/microsoft/LoRA).