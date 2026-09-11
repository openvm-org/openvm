use std::{marker::PhantomData, sync::Arc};

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

impl InterpretedInstance<'_, PreflightCtx> {
    /// Executes exactly one metered segment from its architectural start state.
    pub fn execute_segment<F: PrimeField32>(
        &self,
        state: VmState<GuestMemory>,
        segment: &Segment,
    ) -> Result<PreflightOutput, ExecutionError> {
        let mut output = self.execute_preflight_from_state::<F>(state, Some(segment.num_insns))?;
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
        let ctx = PreflightCtx::new::<F>(&from_state.memory, num_insns);
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
pub struct PreflightInterpretedInstance<F, E> {
    // Drop the interpreter before releasing the inventory that backs its
    // pre-compute pointers.
    inner: InterpretedInstance<'static, PreflightCtx>,
    _inventory: Arc<ExecutorInventory<E>>,
    _field: PhantomData<fn() -> F>,
}

impl<F, E> PreflightInterpretedInstance<F, E>
where
    F: PrimeField32,
    E: Executor<F> + 'static,
{
    pub fn new(
        exe: &VmExe,
        inventory: Arc<ExecutorInventory<E>>,
    ) -> Result<Self, StaticProgramError> {
        let inventory_ref = unsafe {
            // SAFETY:
            // - `inventory` is stored in the returned wrapper and therefore outlives `inner`.
            // - `inner` is declared before `_inventory`, so it is dropped first.
            // - moving the Arc does not move its allocation.
            &*Arc::as_ptr(&inventory)
        };
        let inner = InterpretedInstance::new::<F, _>(inventory_ref, exe)?;
        Ok(Self {
            inner,
            _inventory: inventory,
            _field: PhantomData,
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
        self.inner.execute_segment::<F>(state, segment)
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
            .execute_preflight_from_state::<F>(state, num_insns)
    }
}
