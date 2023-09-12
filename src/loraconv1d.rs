use std::ops::Mul;

use candle_core::{Module, Result, Tensor};
use candle_nn::{init, Conv1d, Conv1dConfig, Dropout, VarMap};
use either::Either;

use crate::{
    frozenconv::FrozenConv1d, Conv1dLayerLike, LoraConfig, Merge, MergeError, MergeErrorOrError,
};

#[derive(Debug)]
pub struct LoraConv1d {
    old: FrozenConv1d,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    dropout: Option<Dropout>,
    merged: bool,
}

/// Configuration for LoraConv1d. Other configurations are inherited from the `Conv1d` struct.
pub struct LoraConv1dConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
}

impl LoraConv1dConfig {
    pub fn new(kernel_size: usize, in_channels: usize, out_channels: usize) -> Self {
        LoraConv1dConfig {
            in_channels,
            out_channels,
            kernel_size,
        }
    }
}

impl LoraConv1d {
    pub fn new(
        old: &dyn Conv1dLayerLike,
        conv_config: &LoraConv1dConfig,
        config: &LoraConfig,
    ) -> Result<Self> {
        let map = VarMap::new();
        let a = map.get(
            (
                config.rank * conv_config.kernel_size,
                conv_config.in_channels * conv_config.kernel_size,
            ),
            "a.weight",
            init::DEFAULT_KAIMING_NORMAL,
            config.dtype,
            config.device,
        )?;
        let b = map.get(
            (
                conv_config.out_channels / old.config().groups * conv_config.kernel_size,
                config.rank * conv_config.kernel_size,
            ),
            "b.weight",
            init::ZERO,
            config.dtype,
            config.device,
        )?;

        Ok(LoraConv1d {
            old: FrozenConv1d::new_from_conv1d(old)?,
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

impl Merge for LoraConv1d {
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError> {
        let result = self
            .b
            .matmul(&self.a)
            .map_err(Either::Right)?
            .reshape(self.old.weight().shape())
            .map_err(Either::Right)?;

        Ok(match self.scale {
            Some(scale) => result.mul(scale).map_err(Either::Right)?,
            None => result,
        })
    }

    fn merge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError> {
        if self.merged {
            Err(Either::Left(MergeError::AlreadyMerged))
        } else {
            self.old = FrozenConv1d::new(
                &(self.old.weight() + self.get_delta_weight()?).map_err(Either::Right)?,
                self.old.bias(),
                *self.old.config(),
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
            self.old = FrozenConv1d::new(
                &(self.old.weight() - self.get_delta_weight()?).map_err(Either::Right)?,
                self.old.bias(),
                *self.old.config(),
            )
            .map_err(Either::Right)?;
            self.merged = false;
            Ok(())
        }
    }
}

impl Module for LoraConv1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.merged {
            return self.old.forward(input);
        }

        if let Some(scale) = self.scale {
            let bias = self.bias().cloned();

            let mut weight = self.old.weight().clone();
            if self.dropout.is_some() {
                weight = self.dropout.as_ref().unwrap().forward(input, true)?;
            }
            let weight = (&weight
                + &self
                    .b
                    .broadcast_matmul(&self.a.broadcast_matmul(&weight)?)?
                    .reshape(self.old.weight().shape())?
                    .mul(scale)?)?;

            let conv = Conv1d::new(weight, bias, *self.config());
            conv.forward(input)
        } else {
            self.old.forward(input)
        }
    }
}

impl Conv1dLayerLike for LoraConv1d {
    fn config(&self) -> &Conv1dConfig {
        self.old.config()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.old.bias()
    }
    fn weight(&self) -> &Tensor {
        self.old.weight()
    }
}
