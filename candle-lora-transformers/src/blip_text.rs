use super::with_tracing::{linear, TracedLoraEmbedding, TracedLoraLinear};
use candle_core::{Module, Result, Tensor, D};
use candle_lora::LoraConfig;
use candle_nn::{layer_norm, LayerNorm, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub encoder_hidden_size: usize,
    pub intermediate_size: usize,
    pub projection_dim: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_act: candle_nn::Activation,
    pub layer_norm_eps: f64,
    pub is_decoder: bool,
}

#[derive(Debug)]
struct TextEmbeddings {
    word_embedddings: TracedLoraEmbedding,
    position_embeddings: TracedLoraEmbedding,
    layer_norm: LayerNorm,
    position_ids: Tensor,
}

impl TextEmbeddings {
    fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let word_embedddings = TracedLoraEmbedding::new(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("word_embeddings"),
            merge,
            lora_config.clone(),
        )?;
        let position_embeddings = TracedLoraEmbedding::new(
            cfg.max_position_embeddings,
            cfg.hidden_size,
            vb.pp("position_embeddings"),
            merge,
            lora_config,
        )?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        let position_ids =
            Tensor::arange(0, cfg.max_position_embeddings as u32, vb.device())?.unsqueeze(0)?;
        Ok(Self {
            word_embedddings,
            position_embeddings,
            layer_norm,
            position_ids,
        })
    }

    fn forward(&self, xs: &Tensor, past_kv_len: usize) -> Result<Tensor> {
        let seq_len = xs.dim(1)?;
        let position_ids = self.position_ids.narrow(1, past_kv_len, seq_len)?;
        let embeddings = self.word_embedddings.forward(xs)?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        (embeddings + position_embeddings)?.apply(&self.layer_norm)
    }
}

#[derive(Debug)]
struct TextSelfAttention {
    query: TracedLoraLinear,
    key: TracedLoraLinear,
    value: TracedLoraLinear,
    attention_head_size: usize,
    num_attention_heads: usize,
    attention_scale: f64,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl TextSelfAttention {
    fn new(
        cfg: &Config,
        is_cross_attention: bool,
        vb: VarBuilder,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let num_attention_heads = cfg.num_attention_heads;
        let attention_head_size = cfg.hidden_size / num_attention_heads;
        let all_head_size = cfg.num_attention_heads * attention_head_size;
        let query = linear(
            cfg.hidden_size,
            all_head_size,
            vb.pp("query"),
            merge,
            lora_config.clone(),
        )?;
        let in_size = if is_cross_attention {
            cfg.encoder_hidden_size
        } else {
            cfg.hidden_size
        };
        let key = linear(
            in_size,
            all_head_size,
            vb.pp("key"),
            merge,
            lora_config.clone(),
        )?;
        let value = linear(in_size, all_head_size, vb.pp("value"), merge, lora_config)?;
        let attention_scale = 1f64 / (attention_head_size as f64).sqrt();
        Ok(Self {
            query,
            key,
            value,
            attention_head_size,
            num_attention_heads,
            attention_scale,
            kv_cache: None,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, _) = xs.dims3()?;
        xs.reshape((
            b_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?
        .permute((0, 2, 1, 3))
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache = None
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let query = self
            .transpose_for_scores(&self.query.forward(xs)?)?
            .contiguous()?;
        let (key, value) = match encoder_hidden_states {
            None => {
                let key = self.transpose_for_scores(&self.key.forward(xs)?)?;
                let value = self.transpose_for_scores(&self.value.forward(xs)?)?;
                let (key, value) = match &self.kv_cache {
                    None => (key, value),
                    Some((prev_key, prev_value)) => {
                        let key = Tensor::cat(&[prev_key, &key], 2)?;
                        let value = Tensor::cat(&[prev_value, &value], 2)?;
                        (key, value)
                    }
                };
                self.kv_cache = Some((key.clone(), value.clone()));
                (key, value)
            }
            Some(xs) => {
                let key = self.transpose_for_scores(&self.key.forward(xs)?)?;
                let value = self.transpose_for_scores(&self.value.forward(xs)?)?;
                // no kv-cache in this case, but the results could probably be memoized.
                (key, value)
            }
        };
        let key = key.contiguous()?;
        let value = value.contiguous()?;
        let attention_scores = query.matmul(&key.t()?)?;
        let attention_scores = (attention_scores * self.attention_scale)?;
        let attention_scores = match attention_mask {
            Some(mask) => attention_scores.broadcast_add(mask)?,
            None => attention_scores,
        };
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        attention_probs
            .matmul(&value)?
            .permute((0, 2, 1, 3))?
            .flatten_from(D::Minus2)
    }
}

#[derive(Debug)]
struct TextSelfOutput {
    dense: TracedLoraLinear,
    layer_norm: LayerNorm,
}

impl TextSelfOutput {
    fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let dense = linear(
            cfg.hidden_size,
            cfg.hidden_size,
            vb.pp("dense"),
            merge,
            lora_config,
        )?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, xs: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        (xs.apply(&self.dense) + input_tensor)?.apply(&self.layer_norm)
    }
}

#[derive(Debug)]
struct TextAttention {
    self_: TextSelfAttention,
    output: TextSelfOutput,
}

impl TextAttention {
    fn new(
        cfg: &Config,
        is_cross_attention: bool,
        vb: VarBuilder,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let self_ = TextSelfAttention::new(
            cfg,
            is_cross_attention,
            vb.pp("self"),
            merge,
            lora_config.clone(),
        )?;
        let output = TextSelfOutput::new(cfg, vb.pp("output"), merge, lora_config)?;
        Ok(Self { self_, output })
    }

    fn reset_kv_cache(&mut self) {
        self.self_.reset_kv_cache()
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let self_outputs = self
            .self_
            .forward(xs, encoder_hidden_states, attention_mask)?;
        self.output.forward(&self_outputs, xs)
    }
}

#[derive(Debug)]
struct TextIntermediate {
    dense: TracedLoraLinear,
    intermediate_act_fn: candle_nn::Activation,
}

impl TextIntermediate {
    fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let dense = linear(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("dense"),
            merge,
            lora_config,
        )?;
        Ok(Self {
            dense,
            intermediate_act_fn: cfg.hidden_act,
        })
    }
}

impl Module for TextIntermediate {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.dense)?.apply(&self.intermediate_act_fn)
    }
}

#[derive(Debug)]
struct TextOutput {
    dense: TracedLoraLinear,
    layer_norm: LayerNorm,
}

impl TextOutput {
    fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let dense = linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("dense"),
            merge,
            lora_config,
        )?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, xs: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        (xs.apply(&self.dense)? + input_tensor)?.apply(&self.layer_norm)
    }
}

#[derive(Debug)]
struct TextLayer {
    attention: TextAttention,
    cross_attention: Option<TextAttention>,
    intermediate: TextIntermediate,
    output: TextOutput,
}

impl TextLayer {
    fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let attention =
            TextAttention::new(cfg, false, vb.pp("attention"), merge, lora_config.clone())?;
        let cross_attention = if cfg.is_decoder {
            Some(TextAttention::new(
                cfg,
                true,
                vb.pp("crossattention"),
                merge,
                lora_config.clone(),
            )?)
        } else {
            None
        };
        let intermediate =
            TextIntermediate::new(cfg, vb.pp("intermediate"), merge, lora_config.clone())?;
        let output = TextOutput::new(cfg, vb.pp("output"), merge, lora_config.clone())?;
        Ok(Self {
            attention,
            cross_attention,
            intermediate,
            output,
        })
    }

    fn reset_kv_cache(&mut self) {
        self.attention.reset_kv_cache();
        if let Some(ca) = &mut self.cross_attention {
            ca.reset_kv_cache()
        }
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        encoder_hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let attention_output = self.attention.forward(xs, None, Some(attention_mask))?;
        let attention_output = match &mut self.cross_attention {
            Some(ca) => ca.forward(&attention_output, Some(encoder_hidden_states), None)?,
            None => candle_core::bail!("expected some cross-attn"),
        };
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(&intermediate_output, &attention_output)
    }
}

#[derive(Debug)]
struct TextEncoder {
    layers: Vec<TextLayer>,
}

impl TextEncoder {
    fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let vb = vb.pp("layer");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = TextLayer::new(cfg, vb.pp(i), merge, lora_config.clone())?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    fn reset_kv_cache(&mut self) {
        self.layers.iter_mut().for_each(|l| l.reset_kv_cache())
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        encoder_hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, encoder_hidden_states, attention_mask)?
        }
        Ok(xs)
    }
}

#[derive(Debug)]
pub struct TextPooler {
    dense: TracedLoraLinear,
}

impl TextPooler {
    pub fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let dense = linear(
            cfg.hidden_size,
            cfg.hidden_size,
            vb.pp("dense"),
            merge,
            lora_config,
        )?;
        Ok(Self { dense })
    }
}

impl Module for TextPooler {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.narrow(D::Minus1, 0, 1)?
            .squeeze(D::Minus1)?
            .apply(&self.dense)?
            .tanh()
    }
}

#[derive(Debug)]
struct TextPredictionHeadTransform {
    dense: TracedLoraLinear,
    transform_act_fn: candle_nn::Activation,
    layer_norm: LayerNorm,
}

impl TextPredictionHeadTransform {
    fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let dense = linear(
            cfg.hidden_size,
            cfg.hidden_size,
            vb.pp("dense"),
            merge,
            lora_config,
        )?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self {
            dense,
            transform_act_fn: cfg.hidden_act,
            layer_norm,
        })
    }
}

impl Module for TextPredictionHeadTransform {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.dense)?
            .apply(&self.transform_act_fn)?
            .apply(&self.layer_norm)
    }
}

#[derive(Debug)]
struct TextLMPredictionHead {
    transform: TextPredictionHeadTransform,
    decoder: TracedLoraLinear,
}

impl TextLMPredictionHead {
    fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let transform =
            TextPredictionHeadTransform::new(cfg, vb.pp("transform"), merge, lora_config.clone())?;
        let weight = vb.get((cfg.vocab_size, cfg.hidden_size), "decoder.weight")?;
        let bias = vb.get(cfg.vocab_size, "bias")?;
        let decoder = TracedLoraLinear::from_weights(
            weight,
            Some(bias),
            vb.pp("decoder"),
            merge,
            lora_config,
        );
        Ok(Self { transform, decoder })
    }
}

impl Module for TextLMPredictionHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.transform)?.apply(&self.decoder)
    }
}

#[derive(Debug)]
struct TextOnlyMLMHead {
    predictions: TextLMPredictionHead,
}

impl TextOnlyMLMHead {
    fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let predictions = TextLMPredictionHead::new(cfg, vb.pp("predictions"), merge, lora_config)?;
        Ok(Self { predictions })
    }
}

impl Module for TextOnlyMLMHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.predictions.forward(xs)
    }
}

#[derive(Debug)]
struct TextModel {
    embeddings: TextEmbeddings,
    encoder: TextEncoder,
    past_kv_len: usize,
    // We do not need the pooler for caption generation
}

impl TextModel {
    pub fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let embeddings = TextEmbeddings::new(cfg, vb.pp("embeddings"), merge, lora_config.clone())?;
        let encoder = TextEncoder::new(cfg, vb.pp("encoder"), merge, lora_config)?;
        Ok(Self {
            embeddings,
            encoder,
            past_kv_len: 0,
        })
    }

    fn forward(
        &mut self,
        input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let embedding_output = self.embeddings.forward(input_ids, self.past_kv_len)?;
        let sequence_output =
            self.encoder
                .forward(&embedding_output, encoder_hidden_states, attention_mask)?;
        self.past_kv_len += seq_len;
        // We're interested in the sequence-output rather than the pooled-output.
        Ok(sequence_output)
    }

    fn reset_kv_cache(&mut self) {
        self.past_kv_len = 0;
        self.encoder.reset_kv_cache();
    }
}

#[derive(Debug)]
pub struct TextLMHeadModel {
    bert: TextModel,
    cls: TextOnlyMLMHead,
}

impl TextLMHeadModel {
    pub fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let bert = TextModel::new(cfg, vb.pp("bert"), merge, lora_config.clone())?;
        let cls = TextOnlyMLMHead::new(cfg, vb.pp("cls"), merge, lora_config)?;
        Ok(Self { bert, cls })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (seq_len, seq_len), input_ids.device())?;
        let sequence_output = self.bert.forward(input_ids, encoder_hidden_states, &mask)?;
        let prediction_scores = self.cls.forward(&sequence_output)?;
        // return_logits is false so we don't discard the last sequence element.
        Ok(prediction_scores)
    }

    pub fn reset_kv_cache(&mut self) {
        self.bert.reset_kv_cache()
    }
}
