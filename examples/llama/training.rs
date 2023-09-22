use crate::{
    model::{Cache, Config, Llama},
    weights::TransformerWeights,
};
use candle_core::{Result, Var};
use candle_nn::{Optimizer, VarMap};

pub fn run(args: &crate::TrainingCmd, common_args: &crate::Args) -> Result<()> {
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
    let _model = Llama::load(vb, &cache, config, true)?;

    let params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let opt = candle_nn::AdamW::new(map.all_vars(), params)?;

    println!("{:?}", opt);

    Ok(())
}
