use std::path::Path;

use candle_core::{DType, Device, Error, Var};
use candle_nn::{
    var_builder::{SimpleBackend, VarBuilderArgs},
    VarBuilder, VarMap,
};

/// Load tensors into a VarBuilder backed by a VarMap using MmapedSafetensors.
pub fn from_mmaped_safetensors<'a, P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>, Error> {
    let map = VarMap::new();
    {
        let mut ws = map.data().lock().unwrap();

        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(paths)? };
        for (name, _) in tensors.tensors() {
            let tensor = tensors
                .load(&name, device)?
                .to_device(device)?
                .to_dtype(dtype)?;
            ws.insert(name, Var::from_tensor(&tensor)?);
        }
    }

    Ok(VarBuilder::from_varmap(&map, dtype, device))
}

/// Load tensors into a VarBuilder backed by a VarMap using NpzTensors.
pub fn from_npz_tensors<'a, P: AsRef<Path>>(
    path: P,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>, Error> {
    let map = VarMap::new();
    {
        let mut ws = map.data().lock().unwrap();

        let tensors = candle_core::npy::NpzTensors::new(path)?;
        for name in tensors.names() {
            let tensor = tensors
                .get(name)?
                .expect("Expect Some(_) tensor.")
                .to_device(device)?
                .to_dtype(dtype)?;
            ws.insert(name.to_string(), Var::from_tensor(&tensor)?);
        }
    }

    Ok(VarBuilder::from_varmap(&map, dtype, device))
}

/// Load tensors into a VarBuilder backed by a VarMap using PthTensors.
pub fn from_pth_tensors<'a, P: AsRef<Path>>(
    path: P,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>, Error> {
    let map = VarMap::new();
    {
        let mut ws = map.data().lock().unwrap();

        let tensors = candle_core::pickle::PthTensors::new(path)?;
        for name in tensors.tensor_infos().keys() {
            let tensor = tensors
                .get(name)?
                .expect("Tensor not found")
                .to_device(device)?
                .to_dtype(dtype)?;
            ws.insert(name.to_string(), Var::from_tensor(&tensor)?);
        }
    }

    Ok(VarBuilder::from_varmap(&map, dtype, device))
}
