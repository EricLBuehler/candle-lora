use candle_core::{Device, Shape, Tensor};
use candle_nn::{Linear, Module};
use loralinear::{LoraLinear, ALPHA_DEFAULT};
use std::{collections::HashMap, hash::Hash};
use trc::Trc;

pub mod loralinear;
mod nontrainlinear;

pub struct Lora;

impl Lora {
    pub fn convert_model<T: Eq + PartialEq + Hash>(
        layers: HashMap<T, &dyn LinearLayerLike>,
        device: &Device,
    ) -> HashMap<T, LoraLinear> {
        let mut output = HashMap::new();
        for (name, layer) in layers {
            output.insert(
                name,
                LoraLinear::new(layer, layer.weight().rank(), ALPHA_DEFAULT, device, 10, 10)
                    .unwrap(),
            );
        }
        output
    }
}

pub type Vars = HashMap<String, Trc<Tensor>>;

pub trait LinearLayerLike: Module {
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
    fn shape(&self) -> &Shape;
}

impl LinearLayerLike for Linear {
    fn weight(&self) -> &Tensor {
        self.weight()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
    fn shape(&self) -> &Shape {
        self.weight().shape()
    }
}
