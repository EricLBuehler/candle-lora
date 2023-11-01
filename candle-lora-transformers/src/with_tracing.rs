//! Tracing layers.

use candle_core::{Module, Result, Tensor};
use candle_lora::{
    EmbeddingLayerLike, LinearLayerLike, LoraConfig, LoraEmbeddingConfig, LoraLinearConfig,
};
use candle_lora_macro::{replace_layer_fields, AutoLoraConvert};
use candle_nn::{Conv2d, VarBuilder};
use std::sync::Arc;

#[derive(Debug, AutoLoraConvert)]
#[replace_layer_fields]
pub struct TracedLoraEmbedding {
    inner: Embedding,
    span: tracing::Span,
}

impl TracedLoraEmbedding {
    pub fn new(
        d1: usize,
        d2: usize,
        vb: VarBuilder,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let embed_config = LoraEmbeddingConfig::new(d1, d2);
        let inner = candle_nn::embedding(d1, d2, vb.clone())?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");

        let mut this = Self {
            inner: Arc::new(inner),
            span,
        };

        if merge {
            this.get_merged_lora_model(
                lora_config,
                &vb.pp("traced_lora_embed"),
                None,
                None,
                None,
                Some(embed_config),
            );
        } else {
            this.get_lora_model(
                lora_config,
                &vb.pp("traced_lora_embed"),
                None,
                None,
                None,
                Some(embed_config),
            );
        }

        Ok(this)
    }

    pub fn embeddings(&self) -> &Tensor {
        self.inner.embeddings()
    }
}

impl Module for TracedLoraEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, AutoLoraConvert)]
#[replace_layer_fields]
pub struct TracedLoraLinear {
    inner: Linear,
    span: tracing::Span,
}

impl TracedLoraLinear {
    pub fn from_weights(
        weights: Tensor,
        bias: Option<Tensor>,
        vb: VarBuilder,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Self {
        let linear_config =
            LoraLinearConfig::new(weights.dims2().unwrap().0, weights.dims2().unwrap().1);
        let inner = candle_nn::Linear::new(weights, bias);
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        let mut this = Self {
            inner: Arc::new(inner),
            span,
        };
        if merge {
            this.get_merged_lora_model(
                lora_config,
                &vb.pp("traced_lora_linear"),
                Some(linear_config),
                None,
                None,
                None,
            );
        } else {
            this.get_lora_model(
                lora_config,
                &vb.pp("traced_lora_linear"),
                Some(linear_config),
                None,
                None,
                None,
            );
        }
        this
    }
}

pub fn linear(
    d1: usize,
    d2: usize,
    vb: VarBuilder,
    merge: bool,
    lora_config: LoraConfig,
) -> Result<TracedLoraLinear> {
    let linear_config = LoraLinearConfig::new(d1, d2);
    let inner = candle_nn::linear(d1, d2, vb.clone())?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    let mut this = TracedLoraLinear {
        inner: Arc::new(inner),
        span,
    };
    if merge {
        this.get_merged_lora_model(
            lora_config,
            &vb.pp("traced_lora_linear"),
            Some(linear_config),
            None,
            None,
            None,
        );
    } else {
        this.get_lora_model(
            lora_config,
            &vb.pp("traced_lora_linear"),
            Some(linear_config),
            None,
            None,
            None,
        );
    }
    Ok(this)
}

pub fn linear_no_bias(
    d1: usize,
    d2: usize,
    vb: VarBuilder,
    merge: bool,
    lora_config: LoraConfig,
) -> Result<TracedLoraLinear> {
    let linear_config = LoraLinearConfig::new(d1, d2);
    let inner = candle_nn::linear_no_bias(d1, d2, vb.clone())?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    let mut this = TracedLoraLinear {
        inner: Arc::new(inner),
        span,
    };
    if merge {
        this.get_merged_lora_model(
            lora_config,
            &vb.pp("traced_lora_linear"),
            Some(linear_config),
            None,
            None,
            None,
        );
    } else {
        this.get_lora_model(
            lora_config,
            &vb.pp("traced_lora_linear"),
            Some(linear_config),
            None,
            None,
            None,
        );
    }
    Ok(this)
}

impl Module for TracedLoraLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// Wrap the conv2d op to provide some tracing.
#[derive(Debug, Clone)]
pub struct TracedLoraConv2d {
    inner: Conv2d,
    span: tracing::Span,
}

impl Module for TracedLoraConv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vs: candle_nn::VarBuilder,
) -> Result<TracedLoraConv2d> {
    let span = tracing::span!(tracing::Level::TRACE, "conv2d");
    let inner = candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vs)?;
    Ok(TracedLoraConv2d { inner, span })
}

// QMatMul wrapper adding some tracing.
#[derive(Clone)]
pub struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    pub fn new(
        out_dim: usize,
        in_dim: usize,
        vb: candle_transformers::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get((in_dim, out_dim), "weight")?;
        let inner = candle_core::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }
}

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

impl std::fmt::Debug for QMatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul")
    }
}
