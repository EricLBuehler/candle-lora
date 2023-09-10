use candle_core::{Module, Result, Tensor};

use crate::LinearLayerLike;

#[derive(Debug, Clone)]
pub(crate) struct FrozenLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl FrozenLinear {
    pub(crate) fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub(crate) fn new_from_linear(old: &dyn LinearLayerLike) -> Result<Self> {
        Ok(Self::new(
            old.weight().detach()?,
            match old.bias() {
                Some(bias) => Some(bias.detach()?),
                None => None,
            },
        ))
    }
}

impl Module for FrozenLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = match *x.dims() {
            [b1, b2, _, _] => self.weight.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.weight.broadcast_left(bsize)?.t()?,
            _ => self.weight.t()?,
        };
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}
