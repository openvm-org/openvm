use std::sync::Arc;

use openvm_stark_backend::AirRef;

use crate::SC;

pub mod deferral;
pub mod nonroot;
pub mod root;
pub mod subair;

pub const CONSTRAINT_EVAL_CACHED_INDEX: usize = 0;

// TODO: move to stark-backend-v2
pub trait Circuit {
    fn airs(&self) -> Vec<AirRef<SC>>;
}

impl<C: Circuit> Circuit for Arc<C> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        self.as_ref().airs()
    }
}
