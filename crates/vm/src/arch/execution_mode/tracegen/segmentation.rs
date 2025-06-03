use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use super::TracegenCtx;
use crate::{
    arch::{
        execution_control::ExecutionControl, ExecutionError, ExecutionState, InstructionExecutor,
        VmChipComplex, VmConfig, VmSegmentState,
    },
    system::memory::INITIAL_TIMESTAMP,
};

/// Check segment every 100 instructions.
const SEGMENT_CHECK_INTERVAL: usize = 100;

// TODO(ayush): fix this name since it's a mouthful
/// Implementation of the ExecutionControl trait using the old segmentation strategy
pub struct TracegenExecutionControlWithSegmentation {
    // Constant
    air_names: Vec<String>,
}

impl TracegenExecutionControlWithSegmentation {
    pub fn new(air_names: Vec<String>) -> Self {
        Self { air_names }
    }
}

impl<F, VC> ExecutionControl<F, VC> for TracegenExecutionControlWithSegmentation
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    type Ctx = TracegenCtx;

    fn initialize_context(&self) -> Self::Ctx {
        Self::Ctx {
            since_last_segment_check: 0,
        }
    }
    fn should_suspend(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        // Avoid checking segment too often.
        if state.ctx.since_last_segment_check != SEGMENT_CHECK_INTERVAL {
            state.ctx.since_last_segment_check += 1;
            return false;
        }
        state.ctx.since_last_segment_check = 0;
        chip_complex.config().segmentation_strategy.should_segment(
            &self.air_names,
            &chip_complex.dynamic_trace_heights().collect::<Vec<_>>(),
            &chip_complex.current_trace_cells(),
        )
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
        state.clk += 1;

        Ok(())
    }
}
