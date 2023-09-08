use std::collections::HashMap;
use candle_core::Tensor;
use candle_nn::{Module, Linear, Conv1d, Conv2d, Embedding};
use trc::Trc;

pub mod layers;

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
