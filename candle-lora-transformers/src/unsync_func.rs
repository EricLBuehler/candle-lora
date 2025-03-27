//! Layers defined by closures, but not Sync.
use candle_core::Module;
use candle_core::{Result, Tensor};
use std::rc::Rc;

/// A layer defined by a simple closure.
#[derive(Clone)]
pub struct UnsyncFunc<'a> {
    #[allow(clippy::type_complexity)]
    f: Rc<dyn 'a + Fn(&Tensor) -> Result<Tensor>>,
}

impl std::fmt::Debug for UnsyncFunc<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "func")
    }
}

pub fn func<'a, F>(f: F) -> UnsyncFunc<'a>
where
    F: 'a + Fn(&Tensor) -> Result<Tensor>,
{
    UnsyncFunc { f: Rc::new(f) }
}

impl Module for UnsyncFunc<'_> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (*self.f)(xs)
    }
}

impl<'a> UnsyncFunc<'a> {
    pub fn new<F>(f: F) -> Self
    where
        F: 'a + Fn(&Tensor) -> Result<Tensor>,
    {
        Self { f: Rc::new(f) }
    }
}
