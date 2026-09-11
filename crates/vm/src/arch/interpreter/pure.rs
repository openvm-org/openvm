use super::{check_exit_code, check_termination, InterpretedInstance};
#[cfg(feature = "metrics")]
use crate::arch::execution_metrics::{ExecutionMetric, ExecutionMetricTimer};
use crate::{
    arch::{
        execution_mode::ExecutionCtx, ExecutionError, ExecutionOutcome, Streams, VmExecState,
        VmState,
    },
    system::memory::online::GuestMemory,
};

impl InterpretedInstance<'_, ExecutionCtx> {
    /// Execute from the program's initial state until successful termination.
    pub fn execute(
        &self,
        inputs: impl Into<Streams>,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        let vm_state =
            VmState::initial(self.system_config, &self.init_memory, self.pc_start, inputs);
        self.execute_from_state(vm_state)
    }

    /// Execute for at most `num_insns` and report whether the program terminated or suspended.
    pub fn execute_for(
        &self,
        inputs: impl Into<Streams>,
        num_insns: u64,
    ) -> Result<ExecutionOutcome<VmState<GuestMemory>>, ExecutionError> {
        let vm_state =
            VmState::initial(self.system_config, &self.init_memory, self.pc_start, inputs);
        self.execute_from_state_for(vm_state, num_insns)
    }

    /// Continue from `from_state` until successful termination.
    pub fn execute_from_state(
        &self,
        from_state: VmState<GuestMemory>,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        match self.execute_from_state_inner(from_state, None)? {
            ExecutionOutcome::Terminated(state) => Ok(state),
            ExecutionOutcome::Suspended(_) => {
                unreachable!("unbounded interpreter execution cannot suspend")
            }
        }
    }

    /// Continue from `from_state` for at most `num_insns` and report whether the program
    /// terminated or suspended.
    pub fn execute_from_state_for(
        &self,
        from_state: VmState<GuestMemory>,
        num_insns: u64,
    ) -> Result<ExecutionOutcome<VmState<GuestMemory>>, ExecutionError> {
        self.execute_from_state_inner(from_state, Some(num_insns))
    }

    fn execute_from_state_inner(
        &self,
        from_state: VmState<GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<ExecutionOutcome<VmState<GuestMemory>>, ExecutionError> {
        let ctx = ExecutionCtx::new(num_insns);
        let mut exec_state = VmExecState::new(from_state, ctx);

        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Pure);
        #[cfg(feature = "metrics")]
        let start_instret_left = exec_state.ctx.instret_left;

        run!("execute_pure", self, exec_state, ExecutionCtx);

        #[cfg(feature = "metrics")]
        {
            let insns = start_instret_left - exec_state.ctx.instret_left;
            metrics.record(insns);
        }
        tracing::debug!("pc: {}", exec_state.vm_state.pc());
        tracing::debug!("interpreter exit code {:?}", exec_state.exit_code);
        tracing::debug!("num_insns {:?}", num_insns);

        let terminated = matches!(exec_state.exit_code.as_ref(), Ok(Some(_)));
        if num_insns.is_some() {
            check_exit_code(exec_state.exit_code)?;
        } else {
            check_termination(exec_state.exit_code)?;
        }
        Ok(if terminated {
            ExecutionOutcome::Terminated(exec_state.vm_state)
        } else {
            ExecutionOutcome::Suspended(exec_state.vm_state)
        })
    }
}
