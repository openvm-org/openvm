//! Metered cost execution: per-chip trace cost tracking matching OpenVM's `MeteredCostCtx`.

use std::sync::Arc;

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_lift::ExtensionRegistry;

use super::{
    bridge::map_rvr_execute_error,
    execute_metered_cost,
    state::{MeteredCostState, TracerPayload, TracerPtr},
    RvrCompiled,
};
#[cfg(feature = "metrics")]
use crate::arch::execution_metrics::{ExecutionMetric, ExecutionMetricTimer};
use crate::{
    arch::{execution_mode::MeteredCostCtx, ExecutionError, Streams, SystemConfig, VmState},
    system::memory::online::GuestMemory,
};

pub struct RvrMeteredCostResult {
    pub state: MeteredCostState,
    pub cost: u64,
}

pub struct RvrMeteredCostInstance<F: PrimeField32> {
    pub(crate) system_config: SystemConfig,
    pub(crate) exe: Arc<VmExe<F>>,
    pub(crate) extensions: ExtensionRegistry<F>,
    pub(crate) compiled: RvrCompiled,
    /// Must match the widths baked into `compiled` so per-block constants and
    /// extension `trace_chip` calls compute cost against the same values.
    pub(crate) widths: Vec<u64>,
}

/// C-compatible metered cost meter data.
///
/// Layout must exactly match the C `Tracer` struct in `openvm_tracer_metered_cost.h`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeteredCostData {
    pub cost: u64,
    pub chip_widths: *const u64,
}

impl Default for MeteredCostData {
    fn default() -> Self {
        Self {
            cost: 0,
            chip_widths: std::ptr::null(),
        }
    }
}

impl TracerPayload for MeteredCostData {
    const KIND: u32 = 10;
}

pub type MeteredCostMeter = TracerPtr<MeteredCostData>;

/// C-compatible pure tracer data.
///
/// Layout must exactly match the C `Tracer` struct in `openvm_tracer_pure.h`.
/// All tracing is no-op; suspension is handled by RvState's target_instret.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PureTracerData;

impl TracerPayload for PureTracerData {
    const KIND: u32 = 12;
}

pub type PureTracer = TracerPtr<PureTracerData>;

impl<F: PrimeField32> RvrMeteredCostInstance<F> {
    pub fn execute_metered_cost(
        &self,
        inputs: impl Into<Streams<F>>,
        ctx: MeteredCostCtx,
    ) -> Result<(MeteredCostCtx, VmState<F, GuestMemory>), ExecutionError> {
        let vm_state = VmState::initial(
            &self.system_config,
            &self.exe.init_memory,
            self.exe.pc_start,
            inputs,
        );
        self.execute_metered_cost_from_state(vm_state, ctx)
    }

    pub fn execute_metered_cost_from_state(
        &self,
        mut vm_state: VmState<F, GuestMemory>,
        ctx: MeteredCostCtx,
    ) -> Result<(MeteredCostCtx, VmState<F, GuestMemory>), ExecutionError> {
        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::MeteredCost);
        let result = tracing::info_span!("execute_metered_cost")
            .in_scope(|| {
                execute_metered_cost(
                    &self.compiled,
                    &self.extensions,
                    &mut vm_state,
                    &self.widths,
                )
            })
            .map_err(map_rvr_execute_error)?;
        #[cfg(feature = "metrics")]
        {
            let insns = result.state.instret;
            metrics.record(insns);
        }

        let mut output_ctx = ctx;
        output_ctx.instret = result.state.instret;
        output_ctx.cost = result.cost;

        Ok((output_ctx, vm_state))
    }
}
