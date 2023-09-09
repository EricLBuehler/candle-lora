use candle_core::Tensor;
use candle_nn::{Linear, Module};
use std::collections::HashMap;
use trc::Trc;

pub mod loralinear;
mod nontrainlinear;

pub struct Lora;

pub type Vars = HashMap<String, Trc<Tensor>>;

pub trait LinearLayerLike: Module {
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
}

impl LinearLayerLike for Linear {
    fn weight(&self) -> &Tensor {
        self.weight()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
}