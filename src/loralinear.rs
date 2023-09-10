//According to https://github.com/microsoft/LoRA/blob/main/loralib/layers.py

use std::ops::Mul;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{init, VarMap, Dropout};
use trc::Trc;

use crate::{nontrainlinear::NonTrainableLinear, LinearLayerLike};

pub const ALPHA_DEFAULT: f64 = 1.;

#[derive(Debug)]
pub struct LoraLinear {
    old: Trc<NonTrainableLinear>,
    a: Trc<Tensor>,
    b: Trc<Tensor>,
    scale: Option<f64>,
    dropout: Option<Dropout>,
}

impl LoraLinear {
    pub fn new(
        old: &dyn LinearLayerLike,
        rank: usize,
        alpha: f64,
        dropout: Option<f32>,
        device: &Device,
        dtype: DType,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        let map = VarMap::new();
        let a = map.get(
            (rank, in_features),
            "a.weight",
            init::DEFAULT_KAIMING_NORMAL,
            dtype,
            device,
        )?;
        let b = map.get((out_features, rank), "b.weight", init::ZERO, dtype, device)?;

        let a = Trc::new(a);
        let b = Trc::new(b);

        Ok(LoraLinear {
            old: Trc::new(NonTrainableLinear::new_from_linear(old)?),
            a,
            b,
            scale: if rank > 0 {Some(alpha / rank as f64)}else{None},
            dropout: dropout.map(Dropout::new),
        })
    }
}

impl Module for LoraLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        //No fan_in_fan_out so no weight.transpose(0,1)
        let mut result = self.old.forward(input)?;
        if let Some(scale) = self.scale {
            if self.dropout.is_some() {
                result = (result + self.dropout.as_ref().unwrap().forward(input, true)?)?;
            }
            else {
                result = (result + input)?;
            }
            result = result.matmul(&self.a.transpose(0,1)?)?;
            result = result.matmul(&self.b.transpose(0,1)?)?;
            result = result.mul(scale)?;
        }
        Ok(result)
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
