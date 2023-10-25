use crate::{
    model::{Cache, Config, Llama},
    weights::TransformerWeights,
};
use candle_core::{Result, Var};
use candle_llm_dataset::{LLMDataset, LLMDatasetIter};
use candle_nn::{Optimizer, VarMap};
use candle_transformers::generation::LogitsProcessor;
use plotly::common::Title;
use plotly::layout::{Axis, Layout};
use plotly::{Plot, Scatter};

const EOS_TOKEN: &str = "</s>";
const EPOCHS: usize = 100;
const SEED: u64 = 42;

pub fn run(args: &crate::TrainingCmd, common_args: &crate::Args) -> Result<()> {
    let tokenizer = common_args.tokenizer().unwrap();

    let config_path = match &args.config {
        Some(config) => std::path::PathBuf::from(config),
        None => {
            let api = hf_hub::api::sync::Api::new().unwrap();
            println!("loading the model weights from {}", args.model_id);
            let api = api.model(args.model_id.clone());
            api.get(&args.which_model).unwrap()
        }
    };

    let device = candle_examples::device(common_args.cpu)?;

    let is_safetensors = config_path
        .extension()
        .map_or(false, |v| v == "safetensors");

    let (config, map) = if is_safetensors {
        let config = Config::tiny();
        let tensors = candle_core::safetensors::load(config_path, &device)?;
        let map = VarMap::new();

        let mut ws = map.data().lock().unwrap();
        ws.extend(
            tensors
                .iter()
                .map(|(k, v)| (k.clone(), Var::from_tensor(v).unwrap())),
        );
        drop(ws);

        (config, map)
    } else {
        let mut file = std::fs::File::open(config_path)?;
        let config = Config::from_reader(&mut file).unwrap();
        println!("{config:?}");
        let weights = TransformerWeights::from_reader(&mut file, &config, &device).unwrap();
        let map = weights.var_builder(&config, &device).unwrap();
        (config, map)
    };

    let vb = candle_nn::VarBuilder::from_varmap(&map, candle_core::DType::F32, &device);

    let cache = Cache::new(false, &config, vb.pp("rot"))?;
    let model = Llama::load(vb, &cache, config, true)?;

    let params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(map.all_vars(), params)?;

    let mut dataset: LLMDataset = LLMDataset::new(vec![], device, tokenizer.clone());
    dataset
        .add_line(
            "What is oxygen good for? Oxygen is good for breathing.".into(),
            true,
            None,
            Some(EOS_TOKEN.to_string()),
        )
        .unwrap();
    dataset
        .add_line(
            "Why are leaves beautiful? Leaves might be beautiful.".into(),
            true,
            None,
            Some(EOS_TOKEN.to_string()),
        )
        .unwrap();
    dataset
        .add_line(
            "What is Kelvin? A unit of temperature.".into(),
            true,
            None,
            Some(EOS_TOKEN.to_string()),
        )
        .unwrap();

    let mut logits_processor = LogitsProcessor::new(SEED, args.temperature, args.top_p);

    let mut losses = Vec::new();
    for epoch in 0..EPOCHS {
        let batch_iter = LLMDatasetIter::new_shuffled(&dataset, 1);
        for (batch_index, batch) in batch_iter.enumerate() {
            let (inp, tgt) = (batch.input.ids, batch.target.ids);
            let logits = model.forward(&inp, 0)?;
            let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
            opt.backward_step(&loss)?;

            let ids = inp.squeeze(0).unwrap().to_vec1().unwrap();
            let logits = logits.detach().unwrap().squeeze(0).unwrap();
            let mut logit_ids = Vec::new();
            for i in 0..logits.dim(0).unwrap() {
                logit_ids.push(logits_processor.sample(&logits.get(i).unwrap()).unwrap());
            }
            let scalar_loss = loss.to_scalar::<f32>().unwrap();
            println!("input = {:?}", tokenizer.decode(&ids, false).unwrap());
            println!(
                "output = {:?}",
                tokenizer.decode(&logit_ids, false).unwrap()
            );
            println!();

            losses.push(scalar_loss);

            if batch_index > 0 && batch_index % 1000 == 0 {
                map.save("checkpoint.safetensors")?
            }
        }

        if epoch > 0 && epoch % 10 == 0 {
            let trace = Scatter::new((0..losses.len()).collect::<Vec<_>>(), losses.clone());

            let layout = Layout::new()
                .x_axis(Axis::new().title(Title::from("Epoch")))
                .y_axis(Axis::new().title(Title::from("Loss")))
                .title(Title::from("Loss graph"));

            let mut plot = Plot::new();
            plot.add_trace(trace);
            plot.set_layout(layout);
            plot.write_html("examples/llama/loss.html");

            map.save(format!("examples/llama/epoch_{epoch}.safetensors"))?;
        }
    }

    println!("Done training!");

    Ok(())
}
