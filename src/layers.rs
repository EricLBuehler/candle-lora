use candle_core::{Module, Result, Tensor};
use candle_nn::{
    linear, linear_no_bias,
    var_builder::{SimpleBackend, VarBuilderArgs},
    Linear,
};
use trc::Trc;

use crate::nontrainlinear::NonTrainableLinear;

#[derive(Debug)]
struct LoraLinear {
    old: Trc<NonTrainableLinear>,
    a: Trc<Linear>,
    b: Trc<Linear>,
    _scale: usize,
    train: bool,
}

impl LoraLinear {
    pub fn new(
        old: Linear,
        rank: usize,
        alpha: usize,
        vb: VarBuilderArgs<'_, Box<dyn SimpleBackend>>,
    ) -> Result<Self> {
        let a = Trc::new(linear_no_bias(rank, 10, vb.clone())?);
        let b = Trc::new(linear(rank, 10, vb.clone())?);

        Ok(LoraLinear {
            old: Trc::new(NonTrainableLinear::new_from_linear(&old)?),
            a,
            b,
            _scale: alpha / rank,
            train: true,
        })
    }
}

impl Module for LoraLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let old_output = self.old.forward(xs)?;
        if self.train {
            let lora_output = self.b.forward(&self.a.forward(xs)?)? * self._scale as f64;
            old_output + lora_output
        } else {
            Ok(old_output)
        }
    }
}
