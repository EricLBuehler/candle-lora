use std::ops::Mul;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{init, Conv2dConfig, VarMap};

use crate::{frozenconv::FrozenConv2d, Conv2dLayerLike};

#[derive(Debug)]
pub struct LoraConv2D {
    old: FrozenConv2d,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
}

pub struct LoraConv2DConfig<'a> {
    pub rank: usize,
    pub alpha: f64,
    pub kernel_size: usize,
    pub device: &'a Device,
    pub dtype: DType,
    in_channels: usize,
    out_channels: usize,
}

impl<'a> LoraConv2DConfig<'a> {
    pub fn default(
        device: &'a Device,
        dtype: DType,
        kernel_size: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> Self {
        LoraConv2DConfig {
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

impl LoraConv2D {
    pub fn new(old: &dyn Conv2dLayerLike, config: &LoraConv2DConfig) -> Result<Self> {
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

        Ok(LoraConv2D {
            old: FrozenConv2d::new_from_conv2d(old)?,
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

impl Module for LoraConv2D {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if let Some(scale) = self.scale {
            input.conv1d(
                &self
                    .b
                    .matmul(&self.a)?
                    .reshape(self.old.weight().shape())?
                    .mul(scale)?,
                self.config().padding,
                self.config().stride,
                self.config().dilation,
                self.config().groups,
            )
        } else {
            self.old.forward(input)
        }
    }
}

impl Conv2dLayerLike for LoraConv2D {
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
