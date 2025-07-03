use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::arch::{
    execution_control::ExecutionControl, execution_mode::E1ExecutionCtx, ExecutionError,
    InsExecutorE1, VmChipComplex, VmConfig, VmSegmentState,
};

#[derive(Default, derive_new::new)]
pub struct E1Ctx {
    pub instret_end: Option<u64>,
}

impl E1ExecutionCtx for E1Ctx {
    fn on_memory_operation(&mut self, _address_space: u32, _ptr: u32, _size: u32) {}
    fn should_suspend<F>(vm_state: &VmSegmentState<F, Self>) -> bool {
        if let Some(instret_end) = vm_state.ctx.instret_end {
            vm_state.pc as u64 >= instret_end
        } else {
            false
        }
    }
}

/// Implementation of the ExecutionControl trait using the old segmentation strategy
#[derive(Default)]
pub struct E1ExecutionControl;

impl<F, VC> ExecutionControl<F, VC> for E1ExecutionControl
where
    F: PrimeField32,
    VC: VmConfig<F>,
    VC::Executor: InsExecutorE1<F>,
{
    type Ctx = E1Ctx;

    fn initialize_context(&self) -> Self::Ctx {
        E1Ctx { instret_end: None }
    }

    fn should_suspend(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        _chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        state
            .ctx
            .instret_end
            .is_some_and(|instret_end| state.instret >= instret_end)
    }

    fn on_start(
        &self,
        _state: &mut VmSegmentState<F, Self::Ctx>,
        _chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
    }

    fn on_suspend_or_terminate(
        &self,
        _state: &mut VmSegmentState<F, Self::Ctx>,
        _chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        _exit_code: Option<u32>,
    ) {
    }

    /// Execute a single instruction
    fn execute_instruction(
        &self,
        _state: &mut VmSegmentState<F, Self::Ctx>,
        _instruction: &Instruction<F>,
        _chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
    {
        todo!()
    }
}
