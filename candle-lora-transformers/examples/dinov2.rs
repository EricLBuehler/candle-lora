//! DINOv2: Learning Robust Visual Features without Supervision
//! https://github.com/facebookresearch/dinov2

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_lora::LoraConfig;
use clap::Parser;

use candle_core::{DType, IndexOp, D};
use candle_lora_transformers::{dinov2, varbuilder_utils::from_mmaped_safetensors};
use candle_nn::Module;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;

    let image = candle_examples::imagenet::load_image224(args.image)?;
    println!("loaded image {image:?}");

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/candle-dino-v2".into());
            api.get("dinov2_vits14.safetensors")?
        }
        Some(model) => model.into(),
    };
    let vb = from_mmaped_safetensors(&[model_file], DType::F32, &device, false)?;
    let loraconfig = LoraConfig::new(1, 1., None);
    let model = dinov2::vit_small(vb, true, loraconfig)?;
    println!("model built");
    let logits = model.forward(&image.unsqueeze(0)?)?;
    let prs = candle_nn::ops::softmax(&logits, D::Minus1)?
        .i(0)?
        .to_vec1::<f32>()?;
    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for &(category_idx, pr) in prs.iter().take(5) {
        println!(
            "{:24}: {:.2}%",
            candle_examples::imagenet::CLASSES[category_idx],
            100. * pr
        );
    }
    Ok(())
}
