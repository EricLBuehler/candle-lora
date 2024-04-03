use std::collections::HashMap;

use candle_core::{Module, Result, Shape, Tensor};
use candle_nn::Linear;

use crate::{LinearLayerLike, Saveable};

/// Linear, but with a `new` implementation that ensures the weight and/or biases are detached (frozen).
#[derive(Debug)]
pub(crate) struct FrozenLinear {
    linear: Linear,
}

impl FrozenLinear {
    pub(crate) fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self {
            linear: Linear::new(weight.detach(), bias.map(|bias| bias.detach())),
        })
    }

    pub(crate) fn new_from_linear(old: &dyn LinearLayerLike) -> Result<Self> {
        Self::new(old.weight().detach(), old.bias().map(|bias| bias.detach()))
    }
}

impl Module for FrozenLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

impl Saveable for FrozenLinear {
    fn get_tensors(&self, _accum: &mut HashMap<String, Tensor>) {
        unimplemented!("Saving not supported for frozen layers, only for candle_lora layers.");
    }
}

impl LinearLayerLike for FrozenLinear {
    fn bias(&self) -> Option<&Tensor> {
        self.linear.bias()
    }
    fn weight(&self) -> &Tensor {
        self.linear.weight()
    }
    fn shape(&self) -> &Shape {
        self.weight().shape()
    }
}
