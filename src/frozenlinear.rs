use candle_core::{Module, Result, Tensor};
use candle_nn::{init, Init, VarBuilder};

use crate::LinearLayerLike;

#[derive(Debug, Clone)]
pub struct FrozenLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl FrozenLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn new_from_linear(old: &dyn LinearLayerLike) -> Result<Self> {
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

/// Create or initialize a new linear layer.
///
/// This uses some default names for weight and biases, namely `"weight"` and `"bias"`.
///
/// Default training state is `true`. See [`Linear::new`].
#[allow(dead_code)]
pub fn linear(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<FrozenLinear> {
    let init_ws = init::DEFAULT_KAIMING_NORMAL;
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vs.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(FrozenLinear::new(ws, Some(bs)))
}

/// Default training state is `true`. See [`Linear::new`].
#[allow(dead_code)]
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<FrozenLinear> {
    let init_ws = init::DEFAULT_KAIMING_NORMAL;
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(FrozenLinear::new(ws, None))
}
