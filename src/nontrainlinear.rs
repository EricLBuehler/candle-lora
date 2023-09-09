use candle_core::{Module, Result, Tensor};
use candle_nn::{init, Init, Linear, VarBuilder};
use trc::Trc;

#[derive(Debug)]
pub struct NonTrainableLinear {
    weight: Trc<Tensor>,
    bias: Option<Trc<Tensor>>,
}

#[allow(dead_code)]
impl NonTrainableLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            weight: Trc::new(weight),
            bias: bias.map(Trc::new),
        }
    }

    pub fn new_from_linear(old: &Linear) -> Result<Self> {
        Ok(Self::new(
            old.weight().detach()?,
            match old.bias() {
                Some(bias) => Some(bias.detach()?),
                None => None,
            },
        ))
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Trc<Tensor>> {
        self.bias.as_ref()
    }
}

impl Module for NonTrainableLinear {
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
pub fn linear(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<NonTrainableLinear> {
    let init_ws = init::DEFAULT_KAIMING_NORMAL;
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vs.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(NonTrainableLinear::new(ws, Some(bs)))
}

/// Default training state is `true`. See [`Linear::new`].
#[allow(dead_code)]
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<NonTrainableLinear> {
    let init_ws = init::DEFAULT_KAIMING_NORMAL;
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(NonTrainableLinear::new(ws, None))
}
