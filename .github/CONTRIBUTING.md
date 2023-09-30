Hello! Thank you for taking an interest in candle-lora.

## Found a bug
Please search the issues [here](https://github.com/EricLBuehler/candle-lora/issues).
If you cannot find a solution or any relevant material, please file an [issue](https://github.com/EricLBuehler/candle-lora/issues/new).

## Contributing a PR
Please ensure your code follows these guidelines:

- You have run `cargo fmt` and `cargo clippy`, and applied any changes.
- You have run `typos` and either added any erroneous to a config file or fixed all others.
- All new layer types are in their seperate `xxlora(linear|embed|conv1d|conv2d).rs` files.
    - If you think there should be an exception made, please notate in the PR and I will check in my review.
- All new layer types implement the `(Linear|Embed|Conv1d|Conv2d)LayerLike` trait.
- Only necessary files (no log or byproduct files) are added.
- All new and existing tests as well as CI pass.