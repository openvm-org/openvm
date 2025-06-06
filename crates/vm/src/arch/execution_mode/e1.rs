use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::arch::{
    execution_control::ExecutionControl, execution_mode::E1E2ExecutionCtx, ExecutionError,
    InsExecutorE1, VmChipComplex, VmConfig, VmSegmentState, VmStateMut,
};

pub type E1Ctx = ();

impl E1E2ExecutionCtx for E1Ctx {
    fn on_memory_operation(&mut self, _address_space: u32, _ptr: u32, _size: u32) {}
}

/// Implementation of the ExecutionControl trait using the old segmentation strategy
#[derive(Default, derive_new::new)]
pub struct E1ExecutionControl {
    pub clk_end: Option<u64>,
}

impl<F, VC> ExecutionControl<F, VC> for E1ExecutionControl
where
    F: PrimeField32,
    VC: VmConfig<F>,
    VC::Executor: InsExecutorE1<F>,
{
    type Ctx = E1Ctx;

    fn initialize_context(&self) -> Self::Ctx {
        ()
    }

    fn should_suspend(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        _chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        if let Some(clk_end) = self.clk_end {
            state.clk >= clk_end
        } else {
            false
        }
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
        state: &mut VmSegmentState<F, Self::Ctx>,
        instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
    {
        let &Instruction { opcode, .. } = instruction;

        if let Some(executor) = chip_complex.inventory.get_mut_executor(&opcode) {
            let mut vm_state = VmStateMut {
                pc: &mut state.pc,
                memory: state.memory.as_mut().unwrap(),
                streams: &mut state.streams,
                ctx: &mut state.ctx,
            };
            executor.execute_e1(&mut vm_state, instruction)?;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: state.pc,
                opcode,
            });
        };

        Ok(())
    }
}
