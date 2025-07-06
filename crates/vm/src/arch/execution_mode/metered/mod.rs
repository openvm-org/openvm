pub mod ctx;
pub mod memory_ctx;
pub mod segment_ctx;

pub use ctx::MeteredCtx;
use openvm_stark_backend::p3_field::PrimeField32;
pub use segment_ctx::Segment;

use crate::{
    arch::{execution_control::ExecutionControl, ExecutionError, InsExecutorE1, VmSegmentState},
    system::{memory::online::GuestMemory, program::ProgramHandler},
};

#[derive(Debug, derive_new::new)]
pub struct MeteredExecutionControl {
    executor_idx_to_air_idx: Vec<usize>,
}

impl<F, Executor> ExecutionControl<F, Executor> for MeteredExecutionControl
where
    F: PrimeField32,
    Executor: InsExecutorE1<F>,
{
    type Memory = GuestMemory;
    type Ctx = MeteredCtx;

    #[inline(always)]
    fn should_suspend(&self, _state: &mut VmSegmentState<F, GuestMemory, Self::Ctx>) -> bool {
        false
    }

    #[inline(always)]
    fn on_suspend_or_terminate(
        &self,
        state: &mut VmSegmentState<F, GuestMemory, Self::Ctx>,
        _exit_code: Option<u32>,
    ) {
        state
            .ctx
            .segmentation_ctx
            .add_final_segment(state.instret, &state.ctx.trace_heights);
    }

    /// Execute a single instruction
    #[inline(always)]
    fn execute_instruction(
        &self,
        state: &mut VmSegmentState<F, GuestMemory, Self::Ctx>,
        handler: &mut ProgramHandler<F, Executor>,
    ) -> Result<(), ExecutionError> {
        // Check if segmentation needs to happen
        state.ctx.check_and_segment(state.instret);

        let (executor, pc_entry) = handler.get_executor(state.pc)?;
        // SAFETY: executor idx is guaranteed to be within bounds in construction of ProgramHandler
        let air_id = unsafe {
            *self
                .executor_idx_to_air_idx
                .get_unchecked(pc_entry.executor_idx as usize)
        };
        executor.execute_metered(&mut state.state_mut(), &pc_entry.insn, air_id)?;

        Ok(())
    }
}
