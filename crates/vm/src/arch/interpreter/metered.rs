use super::{check_exit_code, check_termination, InterpretedInstance};
#[cfg(feature = "metrics")]
use crate::arch::execution_metrics::{ExecutionMetric, ExecutionMetricTimer};
use crate::{
    arch::{
        execution_mode::{MeteredCostCtx, MeteredCtx, Segment},
        ExecutionError, Streams, VmExecState, VmState,
    },
    system::memory::online::GuestMemory,
};

impl InterpretedInstance<'_, MeteredCtx> {
    /// Executes the program from its initial state and returns its segments and final state.
    pub fn execute_metered(
        &self,
        inputs: impl Into<Streams>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<GuestMemory>), ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_metered_from_state(vm_state, ctx)
    }

    /// Executes from `from_state` and returns the resulting segments and final state.
    pub fn execute_metered_from_state(
        &self,
        from_state: VmState<GuestMemory>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<GuestMemory>), ExecutionError> {
        let mut exec_state = VmExecState::new(from_state, ctx);

        loop {
            exec_state = self.execute_metered_until_suspend(exec_state)?;
            let exit_code = std::mem::replace(&mut exec_state.exit_code, Ok(None))?;
            if exit_code.is_some() {
                exec_state.exit_code = Ok(exit_code);
                break;
            }
        }
        check_termination(exec_state.exit_code)?;
        let VmExecState { vm_state, ctx, .. } = exec_state;
        Ok((ctx.into_segments(), vm_state))
    }

    /// Executes until the metered context suspends or the program terminates.
    pub fn execute_metered_until_suspend(
        &self,
        mut exec_state: VmExecState<GuestMemory, MeteredCtx>,
    ) -> Result<VmExecState<GuestMemory, MeteredCtx>, ExecutionError> {
        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Metered);
        #[cfg(feature = "metrics")]
        let start_instret = exec_state.ctx.segmentation_ctx.instret;

        run!("execute_metered", self, exec_state, MeteredCtx);

        #[cfg(feature = "metrics")]
        {
            let insns = exec_state.ctx.segmentation_ctx.instret - start_instret;
            metrics.record(insns);
        }
        Ok(exec_state)
    }
}

impl InterpretedInstance<'_, MeteredCostCtx> {
    /// Executes the program from its initial state and returns its cost and final state.
    pub fn execute_metered_cost(
        &self,
        inputs: impl Into<Streams>,
        ctx: MeteredCostCtx,
    ) -> Result<(MeteredCostCtx, VmState<GuestMemory>), ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_metered_cost_from_state(vm_state, ctx)
    }

    /// Executes from `from_state` and returns its cost and final state.
    pub fn execute_metered_cost_from_state(
        &self,
        from_state: VmState<GuestMemory>,
        ctx: MeteredCostCtx,
    ) -> Result<(MeteredCostCtx, VmState<GuestMemory>), ExecutionError> {
        let mut exec_state = VmExecState::new(from_state, ctx);

        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::MeteredCost);
        #[cfg(feature = "metrics")]
        let start_instret = exec_state.ctx.instret;

        run!("execute_metered_cost", self, exec_state, MeteredCostCtx);

        #[cfg(feature = "metrics")]
        {
            let insns = exec_state.ctx.instret - start_instret;
            metrics.record(insns);
        }

        check_exit_code(exec_state.exit_code)?;
        let VmExecState { ctx, vm_state, .. } = exec_state;
        Ok((ctx, vm_state))
    }
}
