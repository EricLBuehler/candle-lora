use std::ops::Mul;

use candle_core::{Module, Result, Tensor};
use candle_nn::{init, Conv2d, Conv2dConfig, Dropout, VarBuilder};
use either::Either;
use trc::Trc;

use crate::{
    frozenconv::FrozenConv2d, Conv2dLayerLike, LoraConfig, Merge, MergeError, MergeErrorOrError,
};

#[derive(Debug, Clone)]
pub struct LoraConv2d {
    old: Trc<FrozenConv2d>,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    dropout: Option<Trc<Dropout>>,
    merged: bool,
}

/// Configuration for LoraConv2d. Other configurations are inherited from the `Conv2d` struct.
pub struct LoraConv2dConfig {
    in_channels: usize,
    out_channels: usize,
}

impl LoraConv2dConfig {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        LoraConv2dConfig {
            in_channels,
            out_channels,
        }
    }
}

impl LoraConv2d {
    pub fn new(
        old: &dyn Conv2dLayerLike,
        conv_config: &LoraConv2dConfig,
        config: &LoraConfig,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let a = vb.get_with_hints(
            (
                config.rank,
                conv_config.in_channels / old.config().groups,
                old.weight().dim(2).unwrap(),
                old.weight().dim(3).unwrap(),
            ),
            "a.weight",
            init::DEFAULT_KAIMING_NORMAL,
        )?;
        let b = vb.get_with_hints(
            (
                conv_config.out_channels,
                config.rank / old.config().groups,
                1,
                1,
            ),
            "b.weight",
            init::ZERO,
        )?;

        Ok(LoraConv2d {
            old: Trc::new(FrozenConv2d::new_from_conv2d(old)?),
            a,
            b,
            scale: if config.rank > 0 {
                Some(config.alpha / config.rank as f64)
            } else {
                None
            },
            dropout: config.dropout.map(|x| Trc::new(Dropout::new(x))),
            merged: false,
        })
    }
}

impl Merge for LoraConv2d {
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError> {
        let result = match self.old.weight().shape().dims()[2..4] {
            [1, 1] => self
                .b
                .squeeze(3)
                .map_err(Either::Right)?
                .squeeze(2)
                .map_err(Either::Right)?
                .matmul(
                    &self
                        .a
                        .squeeze(3)
                        .map_err(Either::Right)?
                        .squeeze(2)
                        .map_err(Either::Right)?,
                )
                .map_err(Either::Right)?
                .unsqueeze(2)
                .map_err(Either::Right)?
                .unsqueeze(3)
                .map_err(Either::Right)?,
            _ => {
                let conv = Conv2d::new(self.b.clone(), None, *self.old.config());
                conv.forward(&self.a.permute((1, 0, 2, 3)).map_err(Either::Right)?)
                    .map_err(Either::Right)?
            }
        };

        Ok(match self.scale {
            Some(scale) => result.mul(scale).map_err(Either::Right)?,
            None => result,
        })
    }

    fn merge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError> {
        if self.merged {
            Err(Either::Left(MergeError::AlreadyMerged))
        } else {
            self.old = Trc::new(
                FrozenConv2d::new(
                    &(self.old.weight() + self.get_delta_weight()?).map_err(Either::Right)?,
                    self.old.bias(),
                    *self.old.config(),
                )
                .map_err(Either::Right)?,
            );
            self.merged = true;
            Ok(())
        }
    }

    fn unmerge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError> {
        if !self.merged {
            Err(Either::Left(MergeError::NotMerged))
        } else {
            self.old = Trc::new(
                FrozenConv2d::new(
                    &(self.old.weight() - self.get_delta_weight()?).map_err(Either::Right)?,
                    self.old.bias(),
                    *self.old.config(),
                )
                .map_err(Either::Right)?,
            );
            self.merged = false;
            Ok(())
        }
    }
}

impl Module for LoraConv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.merged {
            return self.old.forward(input);
        }

        if let Some(scale) = self.scale {
            let weight = self.old.forward(input)?;
            let mut a_input = input.clone();
            if self.dropout.is_some() {
                a_input = self.dropout.as_ref().unwrap().forward(input, true)?;
            }

            let a_conv = Conv2d::new(self.a.clone(), None, *self.config());
            let b_conv = Conv2d::new(
                self.b.clone(),
                None,
                Conv2dConfig {
                    stride: 1,
                    ..*self.config()
                },
            );

            let tmp = b_conv.forward(&a_conv.forward(&a_input)?)?;

            &weight + tmp.mul(scale)?
        } else {
            self.old.forward(input)
        }
    }
}

impl Conv2dLayerLike for LoraConv2d {
    fn config(&self) -> &Conv2dConfig {
        self.old.config()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.old.bias()
    }
    fn weight(&self) -> &Tensor {
        self.old.weight()
    }
}
