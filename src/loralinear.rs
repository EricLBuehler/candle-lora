use std::ops::Mul;

//According to https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{init, Dropout, VarMap};

use crate::{nontrainlinear::NonTrainableLinear, LinearLayerLike};

#[derive(Debug)]
pub struct LoraLinear {
    old: NonTrainableLinear,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    dropout: Option<Dropout>,
}

pub struct LoraLinearConfig<'a> {
    pub rank: usize,
    pub alpha: f64,
    pub dropout: Option<f32>,
    pub device: &'a Device,
    pub dtype: DType,
}

impl<'a> LoraLinearConfig<'a> {
    pub fn default(device: &'a Device, dtype: DType) -> Self {
        LoraLinearConfig {
            rank: 1,
            alpha: 1.,
            dropout: Some(0.),
            device,
            dtype,
        }
    }
}

impl LoraLinear {
    pub fn new(
        old: &dyn LinearLayerLike,
        metadata: &LoraLinearConfig,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        let map = VarMap::new();
        let a = map.get(
            (metadata.rank, in_features),
            "a.weight",
            init::DEFAULT_KAIMING_NORMAL,
            metadata.dtype,
            metadata.device,
        )?;
        let b = map.get(
            (out_features, metadata.rank),
            "b.weight",
            init::ZERO,
            metadata.dtype,
            metadata.device,
        )?;

        Ok(LoraLinear {
            old: NonTrainableLinear::new_from_linear(old)?,
            a,
            b,
            scale: if metadata.rank > 0 {
                Some(metadata.alpha / metadata.rank as f64)
            } else {
                None
            },
            dropout: metadata.dropout.map(Dropout::new),
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
            } else {
                result = (result + input)?;
            }
            result = result.matmul(&self.a.transpose(0, 1)?)?;
            result = result.matmul(&self.b.transpose(0, 1)?)?;
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
