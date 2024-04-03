use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{Module, Result, Tensor};
use candle_nn::{init, Conv1d, Conv1dConfig, Dropout, VarBuilder};
use either::Either;

use crate::{
    frozenconv::FrozenConv1d, Conv1dLayerLike, LoraConfig, Merge, MergeError, MergeErrorOrError,
    Saveable,
};

#[derive(Debug, Clone)]
pub struct LoraConv1d {
    old: Arc<FrozenConv1d>,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    dropout: Option<Arc<Dropout>>,
    merged: bool,
    prefix: String,
    id: usize,
}

#[derive(Clone, Debug)]
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
        vb: &VarBuilder,
        id: usize,
    ) -> Result<Self> {
        let a = vb.pp(format!("a{id}")).get_with_hints(
            (
                config.rank * conv_config.kernel_size,
                conv_config.in_channels * conv_config.kernel_size,
            ),
            "weight",
            init::DEFAULT_KAIMING_NORMAL,
        )?;
        let b = vb.pp(format!("b{id}")).get_with_hints(
            (
                conv_config.out_channels / old.config().groups * conv_config.kernel_size,
                config.rank * conv_config.kernel_size,
            ),
            "weight",
            init::ZERO,
        )?;

        Ok(LoraConv1d {
            old: Arc::new(FrozenConv1d::new_from_conv1d(old)?),
            a,
            b,
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
            self.old = Arc::new(
                FrozenConv1d::new(
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
                FrozenConv1d::new(
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

impl Saveable for LoraConv1d {
    fn get_tensors(&self, accum: &mut HashMap<String, Tensor>) {
        accum.insert(
            self.prefix.clone() + &format!(".a{}.weight", self.id),
            self.a.clone(),
        );
        accum.insert(
            self.prefix.clone() + &format!(".b{}.weight", self.id),
            self.b.clone(),
        );
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
