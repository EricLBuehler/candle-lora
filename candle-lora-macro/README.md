# candle-lora-macro
This library makes using [`candle-lora`](https://github.com/EricLBuehler/candle-lora) as simple as adding 2 macros to your model structs and calling a method! It is inspired by the simplicity of the Python `peft` library's `get_peft_model` method. Like `candle-lora`, the supported concrete layer types are `Linear`, `Conv1d`, `Conv2d`, and `Embedding`.

`candle-lora-macro` exports 2 macros: `AutoLoraConvert` and `replace_layer_fields`.

The `AutoLoraConvert` derive macro automatically creates a method `get_lora_model`, when called which selects and swaps all supported layers for their LoRA counterparts. This method is the equivalent of `peft`'s `get_peft_model` method, and modifies the model in place. It expects all
layers of the supported types to be a `dyn` type: `Arc<dyn ...LayerLike>`. **Therefore the type wrapping the layer must be `Arc`.**

In addition, `AutoLoraConvert` also defines a method `get_merged_lora_model` which does everything `get_lora_model` does, but also merges the weights of the LoRA layers to improve inference performance.

To further automate the process of using `candle-lora`, `candle-lora-macro` also provides an attribute macro called `replace_layer_fields`.
`replace_layer_fields` swaps out the concrete types for `dyn` types. If this macro is not added to the model structs, be sure to change the member types to `Arc<dyn ...LayerLike>`.

`replace_layer_fields` is able to swap:
- `Linear` to `Arc<dyn LinearLayerLike>`
- `Conv1d` to `Arc<dyn Conv1dLayerLike>`
- `Conv2d` to `Arc<dyn Conv2dLayerLike>`
- `Embedding` to `Arc<dyn EmbeddigLayerLike>`
- `Option<Linear>` to `Option<Arc<dyn LinearLayerLike>>`
- `Option<Conv1d>` to `Option<Arc<dyn Conv1dLayerLike>>`
- `Option<Conv2d>` to `Option<Arc<dyn Conv2dLayerLike>>`
- `Option<Embedding>` to `Option<Arc<dyn EmbeddigLayerLike>>`