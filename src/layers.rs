use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, linear_no_bias, var_builder::{VarBuilderArgs, SimpleBackend}, linear};
use trc::Trc;

#[derive(Debug)]
struct LoraLinear {
    old: Trc<Linear>,
    a: Trc<Linear>,
    b: Trc<Linear>,
    _scale: usize,
    train: bool,
}

impl LoraLinear {
    pub fn new<'a>(old: Trc<Linear>, rank: usize, alpha: usize, vb: VarBuilderArgs<'a, Box<dyn SimpleBackend>>) -> Result<Self> {
        //old.set_training(false) TODO, Trc<Linear> means this is impossible for now
        
        let a = Trc::new(linear_no_bias(rank, 10, vb.clone())?);

        let b = Trc::new(linear(rank, 10, vb.clone())?);

        Ok(
        LoraLinear {
            old,
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
        }
        else {
            Ok(old_output)
        }
    }
}