use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{
        execution_control::ExecutionControl, ExecutionError, ExecutionState, InstructionExecutor,
        VmChipComplex, VmConfig, VmSegmentState,
    },
    system::memory::INITIAL_TIMESTAMP,
};

pub type TracegenCtx = ();

#[derive(Default, derive_new::new)]
pub struct TracegenExecutionControl {
    pub clk_end: Option<u64>,
}

impl<F, VC> ExecutionControl<F, VC> for TracegenExecutionControl
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    type Ctx = TracegenCtx;

    fn initialize_context(&self) -> Self::Ctx {}

    fn should_suspend(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        _chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        self.clk_end.is_some_and(|clk_end| state.clk >= clk_end)
    }

    fn on_start(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        chip_complex
            .connector_chip_mut()
            .begin(ExecutionState::new(state.pc, INITIAL_TIMESTAMP + 1));
    }

    fn on_suspend_or_terminate(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        exit_code: Option<u32>,
    ) {
        let timestamp = chip_complex.memory_controller().timestamp();
        chip_complex
            .connector_chip_mut()
            .end(ExecutionState::new(state.pc, timestamp), exit_code);
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
        let timestamp = chip_complex.memory_controller().timestamp();

        let &Instruction { opcode, .. } = instruction;

        if let Some(executor) = chip_complex.inventory.get_mut_executor(&opcode) {
            let memory_controller = &mut chip_complex.base.memory_controller;
            let new_state = executor.execute(
                memory_controller,
                &mut state.streams,
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

        Ok(())
    }
}
