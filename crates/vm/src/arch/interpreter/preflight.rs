use std::sync::Arc;

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;

use super::InterpretedInstance;
#[cfg(feature = "metrics")]
use crate::arch::execution_metrics::{ExecutionMetric, ExecutionMetricTimer};
use crate::{
    arch::{
        execution_mode::{PreflightCtx, Segment},
        ExecutionError, Executor, ExecutorInventory, ExitCode, PreflightOutput, StaticProgramError,
        Streams, VmExecState, VmState,
    },
    system::memory::online::GuestMemory,
};

type PreflightCtxFactory = fn(&GuestMemory, Option<u64>) -> PreflightCtx;

impl InterpretedInstance<'_, PreflightCtx> {
    /// Executes exactly one metered segment from its architectural start state.
    pub fn execute_segment<F: PrimeField32>(
        &self,
        state: VmState<GuestMemory>,
        segment: &Segment,
    ) -> Result<PreflightOutput, ExecutionError> {
        self.execute_segment_with_ctx_factory(state, segment, PreflightCtx::new_for_field::<F>)
    }

    fn execute_segment_with_ctx_factory(
        &self,
        state: VmState<GuestMemory>,
        segment: &Segment,
        ctx_factory: PreflightCtxFactory,
    ) -> Result<PreflightOutput, ExecutionError> {
        let mut output = self.execute_preflight_from_state_with_ctx_factory(
            state,
            Some(segment.num_insns),
            ctx_factory,
        )?;
        if let Some(exit_code) = output.exit_code {
            if exit_code != ExitCode::Success as u32 {
                return Err(ExecutionError::FailedWithExitCode(exit_code));
            }
        }
        output.mark_written_pages();
        Ok(output)
    }

    /// Execute while recording the append-only history needed by postflight.
    pub fn execute_preflight_from_state<F: PrimeField32>(
        &self,
        from_state: VmState<GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<PreflightOutput, ExecutionError> {
        self.execute_preflight_from_state_with_ctx_factory(
            from_state,
            num_insns,
            PreflightCtx::new_for_field::<F>,
        )
    }

    fn execute_preflight_from_state_with_ctx_factory(
        &self,
        from_state: VmState<GuestMemory>,
        num_insns: Option<u64>,
        ctx_factory: PreflightCtxFactory,
    ) -> Result<PreflightOutput, ExecutionError> {
        let ctx = ctx_factory(&from_state.memory, num_insns);
        let mut exec_state = VmExecState::new(from_state, ctx);
        let start_instret_left = exec_state.ctx.instret_left;

        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Preflight);

        run!("execute_preflight", self, exec_state, PreflightCtx);

        let retired = start_instret_left - exec_state.ctx.instret_left;
        #[cfg(feature = "metrics")]
        metrics.record(retired);

        if let Some(expected) = num_insns {
            if retired != expected {
                return Err(ExecutionError::RetiredInstructionCountMismatch {
                    expected,
                    actual: retired,
                });
            }
        }

        let exit_code = exec_state.exit_code?;
        if num_insns.is_none() && exit_code.is_none() {
            return Err(ExecutionError::DidNotTerminate);
        }
        let pc = exec_state.vm_state.pc();
        let history = exec_state.ctx.finish(pc);
        Ok(PreflightOutput {
            history,
            state: exec_state.vm_state,
            exit_code,
        })
    }
}

/// Owned interpreter instance for append-only preflight execution.
///
/// The ordinary interpreter borrows its executor inventory because generated
/// pre-compute data may point into executors. Proving instances own both the VM
/// and this interpreter, so this wrapper keeps the shared inventory alive for
/// exactly as long as the borrowed interpreter.
pub struct PreflightInterpretedInstance<E> {
    // Drop the interpreter before releasing the inventory that backs its
    // pre-compute pointers.
    inner: InterpretedInstance<'static, PreflightCtx>,
    _inventory: Arc<ExecutorInventory<E>>,
    ctx_factory: PreflightCtxFactory,
}

impl<E> PreflightInterpretedInstance<E>
where
    E: Executor + 'static,
{
    pub fn new(
        exe: &VmExe,
        inventory: Arc<ExecutorInventory<E>>,
    ) -> Result<Self, StaticProgramError> {
        Self::new_with_ctx_factory(exe, inventory, PreflightCtx::new)
    }

    pub fn new_for_field<F: PrimeField32>(
        exe: &VmExe,
        inventory: Arc<ExecutorInventory<E>>,
    ) -> Result<Self, StaticProgramError> {
        Self::new_with_ctx_factory(exe, inventory, PreflightCtx::new_for_field::<F>)
    }

    fn new_with_ctx_factory(
        exe: &VmExe,
        inventory: Arc<ExecutorInventory<E>>,
        ctx_factory: PreflightCtxFactory,
    ) -> Result<Self, StaticProgramError> {
        let inventory_ref = unsafe {
            // SAFETY:
            // - `inventory` is stored in the returned wrapper and therefore outlives `inner`.
            // - `inner` is declared before `_inventory`, so it is dropped first.
            // - moving the Arc does not move its allocation.
            &*Arc::as_ptr(&inventory)
        };
        let inner = InterpretedInstance::new(inventory_ref, exe)?;
        Ok(Self {
            inner,
            _inventory: inventory,
            ctx_factory,
        })
    }

    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.inner.create_initial_vm_state(inputs)
    }

    /// Executes exactly one metered segment from its architectural start state.
    pub fn execute_segment(
        &self,
        state: VmState<GuestMemory>,
        segment: &Segment,
    ) -> Result<PreflightOutput, ExecutionError> {
        self.inner
            .execute_segment_with_ctx_factory(state, segment, self.ctx_factory)
    }

    /// Low-level interpreter entry point.
    ///
    /// `None` runs until termination and may retain an unbounded history. Normal
    /// proving code should use [`Self::execute_segment`] with a metered bound.
    pub fn execute_preflight_from_state(
        &self,
        state: VmState<GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<PreflightOutput, ExecutionError> {
        self.inner
            .execute_preflight_from_state_with_ctx_factory(state, num_insns, self.ctx_factory)
    }
}
