use std::ops::Mul;

use candle_core::{Module, Result, Shape, Tensor};
use candle_nn::{init, Dropout, VarMap};
use either::Either;

use crate::{
    frozenlinear::FrozenLinear, LinearLayerLike, LoraConfig, Merge, MergeError, MergeErrorOrError,
};

#[derive(Debug)]
pub struct LoraLinear {
    old: FrozenLinear,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    dropout: Option<Dropout>,
    merged: bool,
}

/// Configuration for LoraLinear
pub struct LoraLinearConfig {
    pub in_features: usize,
    pub out_features: usize,
}

impl LoraLinearConfig {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        LoraLinearConfig {
            in_features,
            out_features,
        }
    }
}

impl LoraLinear {
    pub fn new(
        old: &dyn LinearLayerLike,
        linear_config: &LoraLinearConfig,
        config: &LoraConfig,
    ) -> Result<Self> {
        let map = VarMap::new();
        let a = map.get(
            (config.rank, linear_config.in_features),
            "a.weight",
            init::DEFAULT_KAIMING_NORMAL,
            config.dtype,
            config.device,
        )?;
        let b = map.get(
            (linear_config.out_features, config.rank),
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
            merged: false,
        })
    }
}

impl Merge for LoraLinear {
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError> {
        let result = self.b.matmul(&self.a).map_err(Either::Right)?;
        Ok(match self.scale {
            Some(scale) => result.mul(scale).map_err(Either::Right)?,
            None => result,
        })
    }

    fn merge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError> {
        if self.merged {
            Err(Either::Left(MergeError::AlreadyMerged))
        } else {
            self.old = FrozenLinear::new(
                (self.old.weight() + self.get_delta_weight()?).map_err(Either::Right)?,
                self.old.bias().cloned(),
            )
            .map_err(Either::Right)?;
            self.merged = true;
            Ok(())
        }
    }

    fn unmerge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError> {
        if !self.merged {
            Err(Either::Left(MergeError::NotMerged))
        } else {
            self.old = FrozenLinear::new(
                (self.old.weight() - self.get_delta_weight()?).map_err(Either::Right)?,
                self.old.bias().cloned(),
            )
            .map_err(Either::Right)?;
            self.merged = false;
            Ok(())
        }
    }
}

impl Module for LoraLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.merged {
            self.old.forward(input)
        } else {
            //No fan_in_fan_out so no weight.transpose(0,1)
            let mut result = self.old.forward(input)?;
            if let Some(scale) = self.scale {
                if self.dropout.is_some() {
                    result = (result + self.dropout.as_ref().unwrap().forward(input, true)?)?;
                } else {
                    result = (result + input)?;
                }
                result = result.broadcast_add(
                    &result.matmul(&self.b.broadcast_matmul(&self.a.matmul(&result)?)?)?,
                )?;
                result = result.broadcast_add(&result.clone().mul(scale)?)?;
            }
            Ok(result)
        }
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
