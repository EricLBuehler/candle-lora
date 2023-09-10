//According to https://keras.io/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora/#create-lora-layer

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{init, Linear, VarMap};
use trc::Trc;

use crate::{nontrainlinear::NonTrainableLinear, LinearLayerLike};

pub const ALPHA_DEFAULT: usize = 1;

#[derive(Debug)]
pub struct LoraLinear {
    old: Trc<NonTrainableLinear>,
    a: Trc<Linear>,
    b: Trc<Linear>,
    scale: usize,
}

impl LoraLinear {
    pub fn new(
        old: &dyn LinearLayerLike,
        rank: usize,
        alpha: usize,
        device: &Device,
        dtype: DType,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        let map = VarMap::new();
        let a_weight = map.get(
            (rank, in_features),
            "a.weight",
            init::DEFAULT_KAIMING_NORMAL,
            dtype,
            device,
        )?;
        let b_weight = map.get((out_features, rank), "b.weight", init::ZERO, dtype, device)?;
        let b_bias = map.get(
            (out_features, out_features),
            "b.bias",
            init::ZERO,
            dtype,
            device,
        )?;

        let a = Trc::new(Linear::new(a_weight, None));
        let b = Trc::new(Linear::new(b_weight, Some(b_bias)));

        Ok(LoraLinear {
            old: Trc::new(NonTrainableLinear::new_from_linear(old)?),
            a,
            b,
            scale: alpha / rank,
        })
    }
}

impl Module for LoraLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let old_output = self.old.forward(input).unwrap();
        let lora_output = self.b.forward(&self.a.forward(input)?)? * self.scale as f64;
        old_output + lora_output
    }
}

impl LinearLayerLike for LoraLinear {
    fn bias(&self) -> Option<&Tensor> {
        unimplemented!("Cannot get bias of LoraLinear layer");
    }
    fn weight(&self) -> &Tensor {
        unimplemented!("Cannot get weight of LoraLinear layer");
    }
    fn shape(&self) -> &candle_core::Shape {
        unimplemented!("Cannot get shape of LoraLinear layer");
    }
}
