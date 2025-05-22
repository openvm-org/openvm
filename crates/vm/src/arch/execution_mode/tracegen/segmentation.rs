use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{
        execution_control::ExecutionControl, ExecutionError, ExecutionState, InstructionExecutor,
        VmChipComplex, VmConfig, VmSegmentState,
    },
    system::memory::{MemoryImage, INITIAL_TIMESTAMP},
};

use super::TracegenCtx;

/// Check segment every 100 instructions.
const SEGMENT_CHECK_INTERVAL: usize = 100;

// TODO(ayush): fix this name since it's a mouthful
/// Implementation of the ExecutionControl trait using the old segmentation strategy
pub struct TracegenExecutionControlWithSegmentation {
    // Constant
    air_names: Vec<String>,
    // State
    pub since_last_segment_check: usize,
    pub final_memory: Option<MemoryImage>,
}

impl TracegenExecutionControlWithSegmentation {
    pub fn new(air_names: Vec<String>) -> Self {
        Self {
            since_last_segment_check: 0,
            air_names,
            final_memory: None,
        }
    }
}

impl<F, VC> ExecutionControl<F, VC> for TracegenExecutionControlWithSegmentation
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    type Ctx = TracegenCtx;

    fn should_suspend(
        &mut self,
        _state: &mut VmSegmentState<Self::Ctx>,
        chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        // Avoid checking segment too often.
        if self.since_last_segment_check != SEGMENT_CHECK_INTERVAL {
            self.since_last_segment_check += 1;
            return false;
        }
        self.since_last_segment_check = 0;
        chip_complex.config().segmentation_strategy.should_segment(
            &self.air_names,
            &chip_complex.dynamic_trace_heights().collect::<Vec<_>>(),
            &chip_complex.current_trace_cells(),
        )
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
