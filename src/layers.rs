use candle_core::{Module, Result, Tensor, DType, Device};
use candle_nn::{
    Linear, VarMap, init,
};
use trc::Trc;

use crate::nontrainlinear::NonTrainableLinear;

pub const ALPHA_DEFAULT: usize = 32;

#[derive(Debug)]
pub struct LoraLinear {
    old: Trc<NonTrainableLinear>,
    a: Trc<Linear>,
    b: Trc<Linear>,
    _scale: usize,
    train: bool,
}

impl LoraLinear {
    pub fn new(
        old: Trc<Linear>,
        rank: usize,
        alpha: usize,
        device: &Device,
    ) -> Result<Self> {
        let map = VarMap::new();
        let a_weight = map.get((rank, rank), "a.weight", init::DEFAULT_KAIMING_NORMAL, DType::F32, device)?;
        let b_weight = map.get((rank, rank), "b.weight", init::DEFAULT_KAIMING_NORMAL, DType::F32, device)?;
        let b_bias = map.get((rank, rank), "a.weight", init::DEFAULT_KAIMING_NORMAL, DType::F32, device)?;

        let a = Trc::new(Linear::new(a_weight, None));
        let b = Trc::new(Linear::new(b_weight, Some(b_bias)));

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
