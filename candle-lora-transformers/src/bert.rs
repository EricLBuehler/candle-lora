//! The BERT model.

use candle_core::{DType, Device, Result, Tensor};
use candle_lora::{
    EmbeddingLayerLike, LinearLayerLike, LoraConfig, LoraEmbeddingConfig, LoraLinearConfig,
};
use candle_lora_macro::{replace_layer_fields, AutoLoraConvert};
use candle_nn::{Embedding, Linear, VarBuilder};
use serde::Deserialize;
use std::ops::Deref;
use std::sync::Arc;

pub const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum HiddenAct {
    Gelu,
    Relu,
}

struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

#[derive(Debug, AutoLoraConvert)]
#[replace_layer_fields]
pub struct BertLinear {
    inner: Linear,
    span: tracing::Span,
}

impl BertLinear {
    pub fn new(
        vb: VarBuilder,
        weight: Tensor,
        bias: Option<Tensor>,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        let dims = weight.dims2().unwrap();
        let linear_config = LoraLinearConfig::new(dims.1, dims.0);
        let mut this = Self {
            inner: Arc::new(Linear::new(weight, bias)),
            span,
        };

        if merge {
            this.get_merged_lora_model(
                lora_config,
                &vb.pp("lora_linear"),
                Some(linear_config),
                None,
                None,
                None,
            )
        } else {
            this.get_lora_model(
                lora_config,
                &vb.pp("lora_linear"),
                Some(linear_config),
                None,
                None,
                None,
            )
        }

        this
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

#[derive(Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
        Self {
            weight,
            bias,
            eps,
            span,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let (_bsize, _seq_len, hidden_size) = x.dims3()?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum_keepdim(2)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = (x.sqr()?.sum_keepdim(2)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?;
        Ok(x)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/configuration_bert.py#L1
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    hidden_act: HiddenAct,
    hidden_dropout_prob: f64,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    initializer_range: f64,
    layer_norm_eps: f64,
    pad_token_id: usize,
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    use_cache: bool,
    classifier_dropout: Option<f64>,
    model_type: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        }
    }
}

impl Config {
    fn _all_mini_lm_l6_v2() -> Self {
        // https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/config.json
        Self {
            vocab_size: 30522,
            hidden_size: 384,
            num_hidden_layers: 6,
            num_attention_heads: 12,
            intermediate_size: 1536,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        }
    }
}

fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}

fn linear(
    size1: usize,
    size2: usize,
    vb: VarBuilder,
    merge: bool,
    lora_config: LoraConfig,
) -> Result<BertLinear> {
    let weight = vb.get((size2, size1), "weight")?;
    let bias = vb.get(size2, "bias")?;
    Ok(BertLinear::new(
        vb.pp("lora_linear"),
        weight,
        Some(bias),
        merge,
        lora_config,
    ))
}

struct Dropout {
    #[allow(dead_code)]
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let (weight, bias) = match (vb.get(size, "weight"), vb.get(size, "bias")) {
        (Ok(weight), Ok(bias)) => (weight, bias),
        (Err(err), _) | (_, Err(err)) => {
            if let (Ok(weight), Ok(bias)) = (vb.get(size, "gamma"), vb.get(size, "beta")) {
                (weight, bias)
            } else {
                return Err(err);
            }
        }
    };
    Ok(LayerNorm::new(weight, bias, eps))
}

#[replace_layer_fields]
#[derive(AutoLoraConvert)]
struct BertEmbedding {
    inner: Embedding,
}

impl Deref for BertEmbedding {
    type Target = Arc<dyn EmbeddingLayerLike>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl BertEmbedding {
    fn new(
        vb: VarBuilder,
        vocab_size: usize,
        hidden_size: usize,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<BertEmbedding> {
        let mut this = Self {
            inner: Arc::new(embedding(vocab_size, hidden_size, vb.clone())?),
        };

        let embed_config = LoraEmbeddingConfig::new(vocab_size, hidden_size);

        if merge {
            this.get_merged_lora_model(
                lora_config,
                &vb.pp("lora_embed"),
                None,
                None,
                None,
                Some(embed_config),
            )
        } else {
            this.get_lora_model(
                lora_config,
                &vb.pp("lora_embed"),
                None,
                None,
                None,
                Some(embed_config),
            )
        }

        Ok(this)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L180
struct BertEmbeddings {
    word_embeddings: BertEmbedding,
    position_embeddings: Option<BertEmbedding>,
    token_type_embeddings: BertEmbedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder, config: &Config, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let word_embeddings = BertEmbedding::new(
            vb.pp("word_embeddings"),
            config.vocab_size,
            config.hidden_size,
            merge,
            lora_config.clone(),
        )?;
        let position_embeddings = BertEmbedding::new(
            vb.pp("position_embeddings"),
            config.max_position_embeddings,
            config.hidden_size,
            merge,
            lora_config.clone(),
        )?;
        let token_type_embeddings = BertEmbedding::new(
            vb.pp("token_type_embeddings"),
            config.type_vocab_size,
            config.hidden_size,
            merge,
            lora_config.clone(),
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_bsize, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (&input_embeddings + token_type_embeddings)?;
        if let Some(position_embeddings) = &self.position_embeddings {
            // TODO: Proper absolute positions?
            let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
            let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?
        }
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;
        Ok(embeddings)
    }
}

struct BertSelfAttention {
    query: BertLinear,
    key: BertLinear,
    value: BertLinear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
    span_softmax: tracing::Span,
}

impl BertSelfAttention {
    fn load(vb: VarBuilder, config: &Config, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = linear(
            hidden_size,
            all_head_size,
            vb.pp("query"),
            merge,
            lora_config.clone(),
        )?;
        let value = linear(
            hidden_size,
            all_head_size,
            vb.pp("value"),
            merge,
            lora_config.clone(),
        )?;
        let key = linear(
            hidden_size,
            all_head_size,
            vb.pp("key"),
            merge,
            lora_config.clone(),
        )?;
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_probs = {
            let _enter_sm = self.span_softmax.enter();
            candle_nn::ops::softmax(&attention_scores, candle_core::D::Minus1)?
        };
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle_core::D::Minus2)?;
        Ok(context_layer)
    }
}

struct BertSelfOutput {
    dense: BertLinear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl BertSelfOutput {
    fn load(vb: VarBuilder, config: &Config, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let dense = linear(
            config.hidden_size,
            config.hidden_size,
            vb.pp("dense"),
            merge,
            lora_config,
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "self-out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L392
struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
    span: tracing::Span,
}

impl BertAttention {
    fn load(vb: VarBuilder, config: &Config, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let self_attention =
            BertSelfAttention::load(vb.pp("self"), config, merge, lora_config.clone())?;
        let self_output = BertSelfOutput::load(vb.pp("output"), config, merge, lora_config)?;
        Ok(Self {
            self_attention,
            self_output,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_outputs = self.self_attention.forward(hidden_states)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L441
struct BertIntermediate {
    dense: BertLinear,
    intermediate_act: HiddenActLayer,
    span: tracing::Span,
}

impl BertIntermediate {
    fn load(vb: VarBuilder, config: &Config, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let dense = linear(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("dense"),
            merge,
            lora_config,
        )?;
        Ok(Self {
            dense,
            intermediate_act: HiddenActLayer::new(config.hidden_act),
            span: tracing::span!(tracing::Level::TRACE, "inter"),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L456
struct BertOutput {
    dense: BertLinear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl BertOutput {
    fn load(vb: VarBuilder, config: &Config, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let dense = linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("dense"),
            merge,
            lora_config,
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L470
struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
    span: tracing::Span,
}

impl BertLayer {
    fn load(vb: VarBuilder, config: &Config, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let attention =
            BertAttention::load(vb.pp("attention"), config, merge, lora_config.clone())?;
        let intermediate =
            BertIntermediate::load(vb.pp("intermediate"), config, merge, lora_config.clone())?;
        let output = BertOutput::load(vb.pp("output"), config, merge, lora_config.clone())?;
        Ok(Self {
            attention,
            intermediate,
            output,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attention_output = self.attention.forward(hidden_states)?;
        // TODO: Support cross-attention?
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L523
        // TODO: Support something similar to `apply_chunking_to_forward`?
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L556
struct BertEncoder {
    layers: Vec<BertLayer>,
    span: tracing::Span,
}

impl BertEncoder {
    fn load(vb: VarBuilder, config: &Config, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                BertLayer::load(
                    vb.pp(format!("layer.{index}")),
                    config,
                    merge,
                    lora_config.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(BertEncoder { layers, span })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut hidden_states = hidden_states.clone();
        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states)?
        }
        Ok(hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L874
pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pub device: Device,
    span: tracing::Span,
}

impl BertModel {
    pub fn load(
        vb: VarBuilder,
        config: &Config,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let (embeddings, encoder) = match (
            BertEmbeddings::load(vb.pp("embeddings"), config, merge, lora_config.clone()),
            BertEncoder::load(vb.pp("encoder"), config, merge, lora_config.clone()),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder)) = (
                        BertEmbeddings::load(
                            vb.pp(format!("{model_type}.embeddings")),
                            config,
                            merge,
                            lora_config.clone(),
                        ),
                        BertEncoder::load(
                            vb.pp(format!("{model_type}.encoder")),
                            config,
                            merge,
                            lora_config,
                        ),
                    ) {
                        (embeddings, encoder)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        let sequence_output = self.encoder.forward(&embedding_output)?;
        Ok(sequence_output)
    }
}
