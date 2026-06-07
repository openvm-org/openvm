use openvm_circuit::arch::execution_mode::{MeteredCostCtx, MeteredCtx};
#[cfg(not(feature = "rvr"))]
use openvm_circuit::arch::{execution_mode::ExecutionCtx, InterpretedInstance};

use crate::F;

cfg_if::cfg_if! {
    if #[cfg(feature = "rvr")] {
        use openvm_circuit::arch::rvr::{
            RvrMeteredCostInstance, RvrMeteredInstance, RvrPureInstance,
        };
        pub type CompiledExePure<'a, F> = RvrPureInstance<'a, F>;
        pub type MeteredInstance<'a, F> = RvrMeteredInstance<'a, F>;
        pub type MeteredCostInstance<'a, F> = RvrMeteredCostInstance<'a, F>;
    } else if #[cfg(feature = "aot")] {
        use openvm_circuit::arch::AotInstance;
        pub type CompiledExePure<'a, F> = AotInstance<'a, F, ExecutionCtx>;
        pub type MeteredInstance<'a, F> = AotInstance<'a, F, MeteredCtx>;
        // AOT has no dedicated metered-cost backend; fall back to the interpreter.
        pub type MeteredCostInstance<'a, F> = InterpretedInstance<'a, F, MeteredCostCtx>;
    } else {
        pub type CompiledExePure<'a, F> = InterpretedInstance<'a, F, ExecutionCtx>;
        pub type MeteredInstance<'a, F> = InterpretedInstance<'a, F, MeteredCtx>;
        pub type MeteredCostInstance<'a, F> = InterpretedInstance<'a, F, MeteredCostCtx>;
    }
}

/// Bundles a [`MeteredInstance`] with a precomputed [`MeteredCtx`] so each execution
/// just clones the ctx instead of rebuilding from the proving key.
pub struct CompiledExeMetered<'a> {
    pub instance: MeteredInstance<'a, F>,
    pub ctx: MeteredCtx,
}

pub struct CompiledExeMeteredCost<'a> {
    pub instance: MeteredCostInstance<'a, F>,
    pub ctx: MeteredCostCtx,
}

#[cfg(feature = "rvr")]
impl CompiledExeMetered<'_> {
    /// Persist the compiled shared library into `dir`. Returns the path of
    /// the copied `.so`/`.dylib`. The `MeteredCtx` is not persisted — it is
    /// rebuilt from the proving key on load via
    /// [`Sdk::load_compiled_metered`](crate::Sdk::load_compiled_metered).
    pub fn save(
        &self,
        dir: &std::path::Path,
    ) -> Result<std::path::PathBuf, openvm_circuit::arch::rvr::CompileError> {
        self.instance.save(dir)
    }
}

#[cfg(feature = "rvr")]
impl CompiledExeMeteredCost<'_> {
    /// Persist the compiled shared library into `dir`. Returns the path of
    /// the copied `.so`/`.dylib`. The `MeteredCostCtx` is not persisted — it
    /// is rebuilt on load via
    /// [`Sdk::load_compiled_metered_cost`](crate::Sdk::load_compiled_metered_cost).
    pub fn save(
        &self,
        dir: &std::path::Path,
    ) -> Result<std::path::PathBuf, openvm_circuit::arch::rvr::CompileError> {
        self.instance.save(dir)
    }
}
