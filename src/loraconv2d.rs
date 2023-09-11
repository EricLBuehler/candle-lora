use std::ops::Mul;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{init, Conv2d, Conv2dConfig, Dropout, VarMap};

use crate::{frozenconv::FrozenConv2d, Conv2dLayerLike};

#[derive(Debug)]
pub struct LoraConv2d {
    old: FrozenConv2d,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    dropout: Option<Dropout>,
}

pub struct LoraConv2dConfig<'a> {
    pub rank: usize,
    pub alpha: f64,
    pub kernel_size: usize,
    pub device: &'a Device,
    pub dtype: DType,
    pub in_channels: usize,
    pub out_channels: usize,
    pub dropout: Option<f32>,
}

impl<'a> LoraConv2dConfig<'a> {
    pub fn default(
        device: &'a Device,
        dtype: DType,
        kernel_size: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> Self {
        LoraConv2dConfig {
            rank: 1,
            alpha: 1.,
            kernel_size,
            device,
            dtype,
            in_channels,
            out_channels,
            dropout: Some(0.),
        }
    }
}

impl LoraConv2d {
    pub fn new(old: &dyn Conv2dLayerLike, config: &LoraConv2dConfig) -> Result<Self> {
        let map = VarMap::new();
        let a = map.get(
            (
                config.rank, // * config.kernel_size,
                config.in_channels / old.config().groups,
                old.weight().dim(2).unwrap(),
                old.weight().dim(3).unwrap(), // * config.kernel_size,
            ),
            "a.weight",
            init::DEFAULT_KAIMING_NORMAL,
            config.dtype,
            config.device,
        )?;
        let b = map.get(
            (
                config.out_channels, // / old.config().groups * config.kernel_size,
                config.rank / old.config().groups,
                1,
                1, // * config.kernel_size,
            ),
            "b.weight",
            init::ZERO,
            config.dtype,
            config.device,
        )?;

        Ok(LoraConv2d {
            old: FrozenConv2d::new_from_conv2d(old)?,
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

impl Module for LoraConv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
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
