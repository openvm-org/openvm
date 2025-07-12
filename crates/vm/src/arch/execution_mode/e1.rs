use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{
        execution_control::ExecutionControl, execution_mode::E1E2ExecutionCtx, ExecutionError,
        InsExecutorE1, VmSegmentState,
    },
    system::{memory::online::GuestMemory, program::PcEntry},
};

#[derive(Default, derive_new::new)]
pub struct E1Ctx {
    pub instret_end: Option<u64>,
}

impl E1E2ExecutionCtx for E1Ctx {
    fn on_memory_operation(&mut self, _address_space: u32, _ptr: u32, _size: u32) {}
}

/// Implementation of the ExecutionControl trait using the old segmentation strategy
#[derive(Default)]
pub struct E1ExecutionControl;

impl<F, Executor> ExecutionControl<F, Executor> for E1ExecutionControl
where
    F: PrimeField32,
    Executor: InsExecutorE1<F>,
{
    type Memory = GuestMemory;
    type Ctx = E1Ctx;

    fn should_suspend(&self, state: &mut VmSegmentState<F, GuestMemory, Self::Ctx>) -> bool {
        state
            .ctx
            .instret_end
            .is_some_and(|instret_end| state.instret >= instret_end)
    }

    fn on_suspend_or_terminate(
        &self,
        _state: &mut VmSegmentState<F, GuestMemory, Self::Ctx>,
        _exit_code: Option<u32>,
    ) {
    }

    /// Execute a single instruction
    #[inline(always)]
    fn execute_instruction(
        &self,
        state: &mut VmSegmentState<F, GuestMemory, Self::Ctx>,
        executor: &mut Executor,
        pc_entry: &PcEntry<F>,
    ) -> Result<(), ExecutionError> {
        executor.execute_e1(&mut state.state_mut(), &pc_entry.insn)?;

        Ok(())
    }
}
