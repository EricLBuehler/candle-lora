use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{Module, Result, Tensor};
use candle_nn::{init, Conv2d, Conv2dConfig, Dropout, VarBuilder};
use either::Either;

use crate::{
    frozenconv::FrozenConv2d, Conv2dLayerLike, LoraConfig, Merge, MergeError, MergeErrorOrError,
    Saveable,
};

#[derive(Debug, Clone)]
pub struct LoraConv2d {
    old: Arc<FrozenConv2d>,
    a_conv: Conv2d,
    b_conv: Conv2d,
    scale: Option<f64>,
    dropout: Option<Arc<Dropout>>,
    merged: bool,
    prefix: String,
    id: usize,
}

#[derive(Clone, Debug)]
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
        id: usize,
    ) -> Result<Self> {
        let a = vb.pp(format!("a{id}")).get_with_hints(
            (
                config.rank,
                conv_config.in_channels / old.config().groups,
                old.weight().dim(2).unwrap(),
                old.weight().dim(3).unwrap(),
            ),
            "weight",
            init::DEFAULT_KAIMING_NORMAL,
        )?;
        let b = vb.pp(format!("b{id}")).get_with_hints(
            (
                conv_config.out_channels,
                config.rank / old.config().groups,
                1,
                1,
            ),
            "weight",
            init::ZERO,
        )?;

        let a_conv = Conv2d::new(a, None, *old.config());
        let b_conv = Conv2d::new(
            b,
            None,
            Conv2dConfig {
                stride: 1,
                ..*old.config()
            },
        );

        Ok(LoraConv2d {
            old: Arc::new(FrozenConv2d::new_from_conv2d(old)?),
            a_conv,
            b_conv,
            scale: if config.rank > 0 {
                Some(config.alpha / config.rank as f64)
            } else {
                None
            },
            dropout: config.dropout.map(|x| Arc::new(Dropout::new(x))),
            merged: false,
            prefix: vb.prefix(),
            id,
        })
    }
}

impl Merge for LoraConv2d {
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError> {
        let result = match self.old.weight().shape().dims()[2..4] {
            [1, 1] => self
                .b_conv
                .weight()
                .squeeze(3)
                .map_err(Either::Right)?
                .squeeze(2)
                .map_err(Either::Right)?
                .matmul(
                    &self
                        .a_conv
                        .weight()
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
                let conv = Conv2d::new(self.b_conv.weight().clone(), None, *self.old.config());
                conv.forward(
                    &self
                        .a_conv
                        .weight()
                        .permute((1, 0, 2, 3))
                        .map_err(Either::Right)?,
                )
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
            self.old = Arc::new(
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
            self.old = Arc::new(
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

            let tmp = self.b_conv.forward(&self.a_conv.forward(&a_input)?)?;

            &weight + tmp.mul(scale)?
        } else {
            self.old.forward(input)
        }
    }
}

impl Saveable for LoraConv2d {
    fn get_tensors(&self, accum: &mut HashMap<String, Tensor>) {
        accum.insert(
            self.prefix.clone() + &format!(".a{}.weight", self.id),
            self.a_conv.weight().clone(),
        );
        accum.insert(
            self.prefix.clone() + &format!(".b{}.weight", self.id),
            self.b_conv.weight().clone(),
        );
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
