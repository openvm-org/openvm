//! Metered cost execution: per-chip trace cost tracking matching OpenVM's `MeteredCostCtx`.

use std::path::{Path, PathBuf};

use rvr_openvm_lift::RvrRuntimeExtension;

use super::{
    bridge::map_rvr_execute_error, execute::execute_metered_cost, RvrCompiled, RvrInitialImage,
};
#[cfg(feature = "metrics")]
use crate::arch::execution_metrics::{ExecutionMetric, ExecutionMetricTimer};
use crate::{
    arch::{execution_mode::MeteredCostCtx, ExecutionError, Streams, SystemConfig, VmState},
    system::memory::online::GuestMemory,
};

pub(super) struct RvrMeteredCostResult {
    pub(super) instret: u64,
    pub(super) cost: u64,
}

pub struct RvrMeteredCostInstance<'a> {
    pub(crate) system_config: &'a SystemConfig,
    pub(crate) initial_image: RvrInitialImage,
    pub(crate) runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
    pub(crate) compiled: RvrCompiled,
}

/// C-compatible state for metered-cost execution.
///
/// Layout must exactly match the generated C `MeteredCostState` struct.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeteredCostState {
    pub instret: u64,
    pub cost: u64,
}

impl Default for MeteredCostState {
    fn default() -> Self {
        Self {
            instret: 0,
            cost: 0,
        }
    }
}

impl RvrMeteredCostInstance<'_> {
    /// Persist the compiled shared library into `dir`. Returns the path to
    /// the copied artifact. The user must re-supply `exe`, `executor_idx_to_air_idx`,
    /// and `widths` when loading.
    pub fn save(&self, dir: &Path) -> Result<PathBuf, super::CompileError> {
        let dest_lib = self
            .compiled
            .lib_file_name_with_suffix(self.compiled.execution_kind().artifact_suffix())?;
        self.compiled.save_artifact(&dir.join(dest_lib))
    }

    pub fn execute_metered_cost(
        &self,
        inputs: impl Into<Streams>,
        ctx: MeteredCostCtx,
    ) -> Result<(MeteredCostCtx, VmState<GuestMemory>), ExecutionError> {
        let vm_state = self
            .initial_image
            .create_vm_state(self.system_config, inputs);
        self.execute_metered_cost_from_state(vm_state, ctx)
    }

    pub fn execute_metered_cost_from_state(
        &self,
        mut vm_state: VmState<GuestMemory>,
        ctx: MeteredCostCtx,
    ) -> Result<(MeteredCostCtx, VmState<GuestMemory>), ExecutionError> {
        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::MeteredCost);
        let result = tracing::info_span!("execute_metered_cost")
            .in_scope(|| execute_metered_cost(&self.compiled, &self.runtime_hooks, &mut vm_state))
            .map_err(map_rvr_execute_error)?;
        #[cfg(feature = "metrics")]
        {
            metrics.record(result.instret);
        }

        let mut output_ctx = ctx;
        output_ctx.instret = result.instret;
        output_ctx.cost = result.cost;

        Ok((output_ctx, vm_state))
    }
}
