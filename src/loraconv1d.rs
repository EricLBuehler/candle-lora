use std::ops::Mul;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{init, Conv1dConfig, VarMap};

use crate::{frozenconv::FrozenConv1d, Conv1dLayerLike};

#[derive(Debug)]
pub struct LoraConv1D {
    old: FrozenConv1d,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
}

pub struct LoraConv1DConfig<'a> {
    pub rank: usize,
    pub alpha: f64,
    pub kernel_size: usize,
    pub device: &'a Device,
    pub dtype: DType,
    pub in_channels: usize,
    pub out_channels: usize,
}

impl<'a> LoraConv1DConfig<'a> {
    pub fn default(
        device: &'a Device,
        dtype: DType,
        kernel_size: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> Self {
        LoraConv1DConfig {
            rank: 1,
            alpha: 1.,
            kernel_size,
            device,
            dtype,
            in_channels,
            out_channels,
        }
    }
}

impl LoraConv1D {
    pub fn new(old: &dyn Conv1dLayerLike, config: &LoraConv1DConfig) -> Result<Self> {
        let map = VarMap::new();
        let a = map.get(
            (
                config.rank * config.kernel_size,
                config.in_channels * config.kernel_size,
            ),
            "a.weight",
            init::DEFAULT_KAIMING_NORMAL,
            config.dtype,
            config.device,
        )?;
        let b = map.get(
            (
                config.out_channels / old.config().groups * config.kernel_size,
                config.rank * config.kernel_size,
            ),
            "b.weight",
            init::ZERO,
            config.dtype,
            config.device,
        )?;

        Ok(LoraConv1D {
            old: FrozenConv1d::new_from_conv1d(old)?,
            a,
            b,
            scale: if config.rank > 0 {
                Some(config.alpha / config.rank as f64)
            } else {
                None
            },
        })
    }
}

impl Module for LoraConv1D {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if let Some(scale) = self.scale {
            let x = input;
            let bias = self.bias();
            let weight =&self
            .b
            .matmul(&self.a)?
            .reshape(self.old.weight().shape())?
            .mul(scale)?;

            let x = x.conv1d(
                &weight,
                self.config().padding,
                self.config().stride,
                self.config().dilation,
                self.config().groups,
            )?;
            match &bias {
                None => Ok(x),
                Some(bias) => {
                    let b = bias.dims1()?;
                    let bias = bias.reshape((1, b, 1))?;
                    Ok(x.broadcast_add(&bias)?)
                }
            }
        } else {
            self.old.forward(input)
        }
    }
}

impl Conv1dLayerLike for LoraConv1D {
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
