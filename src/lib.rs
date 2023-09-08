use std::collections::HashMap;

use candle_nn::{Module, Linear, Conv1d, Conv2d, Embedding};
use trc::Trc;

pub struct Lora;

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
    pub fn get_lora_model(model: &impl LoraLayersModule) -> LoraModel {
        println!("{:?}", model);
        todo!()
    }
}

#[derive(Debug)]
pub struct LoraModel;

impl Module for LoraModel {
    fn forward(&self, xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        todo!()
    }
}