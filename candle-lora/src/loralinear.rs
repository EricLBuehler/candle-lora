use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{Module, Result, Shape, Tensor};
use candle_nn::{init, Dropout, Linear, VarBuilder};
use either::Either;

use crate::{
    frozenlinear::FrozenLinear, LinearLayerLike, LoraConfig, Merge, MergeError, MergeErrorOrError,
    Saveable,
};

#[derive(Debug, Clone)]
pub struct LoraLinear {
    old: Arc<FrozenLinear>,
    ff_a: Linear,
    ff_b: Linear,
    scale: Option<f64>,
    dropout: Option<Arc<Dropout>>,
    merged: bool,
    prefix: String,
    id: usize,
}

#[derive(Clone, Debug)]
/// Configuration for LoraLinear
pub struct LoraLinearConfig {
    in_features: usize,
    out_features: usize,
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
        vb: &VarBuilder,
        id: usize,
    ) -> Result<Self> {
        let a = vb.pp(format!("a{id}")).get_with_hints(
            (config.rank, linear_config.in_features),
            "weight",
            init::DEFAULT_KAIMING_NORMAL,
        )?;
        let b = vb.pp(format!("b{id}")).get_with_hints(
            (linear_config.out_features, config.rank),
            "weight",
            init::ZERO,
        )?;

        Ok(LoraLinear {
            old: Arc::new(FrozenLinear::new_from_linear(old)?),
            ff_a: Linear::new(a, None),
            ff_b: Linear::new(b, None),
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

impl Merge for LoraLinear {
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError> {
        let result = self
            .ff_b
            .weight()
            .matmul(self.ff_a.weight())
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
                FrozenLinear::new(
                    (self.old.weight() + self.get_delta_weight()?).map_err(Either::Right)?,
                    self.old.bias().cloned(),
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
                FrozenLinear::new(
                    (self.old.weight() - self.get_delta_weight()?).map_err(Either::Right)?,
                    self.old.bias().cloned(),
                )
                .map_err(Either::Right)?,
            );
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
                let input_new = if self.dropout.is_some() {
                    self.dropout.as_ref().unwrap().forward(input, true)?
                } else {
                    input.clone()
                };

                result =
                    (result + self.ff_b.forward(&self.ff_a.forward(&input_new)?))?.mul(scale)?;
            }
            Ok(result)
        }
    }
}

impl Saveable for LoraLinear {
    fn get_tensors(&self, accum: &mut HashMap<String, Tensor>) {
        accum.insert(
            self.prefix.clone() + &format!(".a{}.weight", self.id),
            self.ff_a.weight().clone(),
        );
        accum.insert(
            self.prefix.clone() + &format!(".b{}.weight", self.id),
            self.ff_b.weight().clone(),
        );
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
