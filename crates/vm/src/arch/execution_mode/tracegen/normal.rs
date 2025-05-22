use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{
        execution_control::ExecutionControl, ExecutionError, ExecutionState, InstructionExecutor,
        VmChipComplex, VmConfig, VmSegmentState,
    },
    system::memory::{MemoryImage, INITIAL_TIMESTAMP},
};

pub type TracegenCtx = ();

/// Implementation of the ExecutionControl trait using the old segmentation strategy
pub struct TracegenExecutionControl {
    // State
    pub clk_end: u64,
    // TODO(ayush): do we need this if only executing one segment?
    pub final_memory: Option<MemoryImage>,
}

impl TracegenExecutionControl {
    pub fn new(clk_end: u64) -> Self {
        Self {
            clk_end,
            final_memory: None,
        }
    }
}

impl<F, VC> ExecutionControl<F, VC> for TracegenExecutionControl
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    type Ctx = TracegenCtx;

    fn should_suspend(
        &mut self,
        state: &mut VmSegmentState<Self::Ctx>,
        _chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        state.clk >= self.clk_end
    }

    fn on_start(
        &mut self,
        state: &mut VmSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        chip_complex
            .connector_chip_mut()
            .begin(ExecutionState::new(state.pc, INITIAL_TIMESTAMP + 1));
    }

    fn on_suspend_or_terminate(
        &mut self,
        state: &mut VmSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        exit_code: Option<u32>,
    ) {
        // TODO(ayush): this should ideally not be here
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());

        let timestamp = chip_complex.memory_controller().timestamp();
        chip_complex
            .connector_chip_mut()
            .end(ExecutionState::new(state.pc, timestamp), exit_code);
    }

    /// Execute a single instruction
    fn execute_instruction(
        &mut self,
        state: &mut VmSegmentState<Self::Ctx>,
        instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
    {
        let timestamp = chip_complex.memory_controller().timestamp();

        let &Instruction { opcode, .. } = instruction;

        if let Some(executor) = chip_complex.inventory.get_mut_executor(&opcode) {
            let memory_controller = &mut chip_complex.base.memory_controller;
            let new_state = executor.execute(
                memory_controller,
                instruction,
                ExecutionState::new(state.pc, timestamp),
            )?;
            state.pc = new_state.pc;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: state.pc,
                opcode,
            });
        };
        state.clk += 1;

        Ok(())
    }
}
