use std::ops::Mul;

use candle_core::{DType, Device, Module, Result, Shape, Tensor};
use candle_nn::{init, Dropout, VarMap};

use crate::{frozenlinear::FrozenLinear, LinearLayerLike};

#[derive(Debug)]
pub struct LoraLinear {
    old: FrozenLinear,
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
        config: &LoraLinearConfig,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        let map = VarMap::new();
        let a = map.get(
            (config.rank, in_features),
            "a.weight",
            init::DEFAULT_KAIMING_NORMAL,
            config.dtype,
            config.device,
        )?;
        let b = map.get(
            (out_features, config.rank),
            "b.weight",
            init::ZERO,
            config.dtype,
            config.device,
        )?;

        Ok(LoraLinear {
            old: FrozenLinear::new_from_linear(old)?,
            a,
            b,
            scale: if config.rank > 0 {
                Some(config.alpha / config.rank as f64)
            } else {
                None
            },
            dropout: config.dropout.map(Dropout::new),
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
        self.old.bias()
    }
    fn weight(&self) -> &Tensor {
        self.old.weight()
    }
    fn shape(&self) -> &Shape {
        self.old.shape()
    }
}
