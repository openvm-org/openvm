use openvm_circuit::arch::execution_mode::{MeteredCostCtx, MeteredCtx};
#[cfg(not(feature = "rvr"))]
use openvm_circuit::arch::{execution_mode::ExecutionCtx, InterpretedInstance};

use crate::F;

cfg_if::cfg_if! {
    if #[cfg(feature = "rvr")] {
        pub use openvm_circuit::arch::rvr::RvrPureInstance as CompiledExePure;
        pub use openvm_circuit::arch::rvr::RvrMeteredInstance as MeteredInstance;
        pub use openvm_circuit::arch::rvr::RvrMeteredCostInstance as MeteredCostInstance;
    } else if #[cfg(feature = "aot")] {
        use openvm_circuit::arch::AotInstance;
        pub type CompiledExePure<F> = AotInstance<F, ExecutionCtx>;
        pub type MeteredInstance<F> = AotInstance<F, MeteredCtx>;
        // AOT has no dedicated metered-cost backend; fall back to the interpreter.
        pub type MeteredCostInstance<F> = InterpretedInstance<F, MeteredCostCtx>;
    } else {
        pub type CompiledExePure<F> = InterpretedInstance<F, ExecutionCtx>;
        pub type MeteredInstance<F> = InterpretedInstance<F, MeteredCtx>;
        pub type MeteredCostInstance<F> = InterpretedInstance<F, MeteredCostCtx>;
    }
}

/// Bundles a [`MeteredInstance`] with a precomputed [`MeteredCtx`] so each execution
/// just clones the ctx instead of rebuilding from the proving key.
pub struct CompiledExeMetered {
    pub instance: MeteredInstance<F>,
    pub ctx: MeteredCtx,
}

pub struct CompiledExeMeteredCost {
    pub instance: MeteredCostInstance<F>,
    pub ctx: MeteredCostCtx,
}

#[cfg(feature = "rvr")]
impl CompiledExeMetered {
    /// Persist the compiled shared library into `dir`. The `MeteredCtx` is
    /// not persisted — it is rebuilt from the proving key on load via
    /// [`Sdk::load_compiled_metered`](crate::Sdk::load_compiled_metered).
    pub fn save(
        &self,
        dir: &std::path::Path,
    ) -> Result<(), openvm_circuit::arch::rvr::CompileError> {
        self.instance.save(dir)
    }
}

#[cfg(feature = "rvr")]
impl CompiledExeMeteredCost {
    /// Persist the compiled shared library into `dir`. The `MeteredCostCtx`
    /// is not persisted — it is rebuilt on load via
    /// [`Sdk::load_compiled_metered_cost`](crate::Sdk::load_compiled_metered_cost).
    pub fn save(
        &self,
        dir: &std::path::Path,
    ) -> Result<(), openvm_circuit::arch::rvr::CompileError> {
        self.instance.save(dir)
    }
}
