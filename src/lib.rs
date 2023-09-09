use candle_core::Tensor;
use candle_nn::{Conv1d, Conv2d, Embedding, Linear, Module};
use std::collections::HashMap;
use trc::Trc;

pub mod loralinear;
mod nontrainlinear;

pub struct Lora;

pub type Vars = HashMap<String, Trc<Tensor>>;

pub enum LayerType {
    Linear(Trc<Linear>),
    Conv1d(Trc<Conv1d>),
    Conv2d(Trc<Conv2d>),
    Embedding(Trc<Embedding>),
}

pub trait LoraLayersModule: Module {
    fn get_layers(&self) -> HashMap<String, LayerType>;
}

impl Lora {
    pub fn get_lora_model(model: &impl LoraLayersModule) {
        println!("{:?}", model);
        todo!()
    }
}
