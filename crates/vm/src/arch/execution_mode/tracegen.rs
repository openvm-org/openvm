use std::marker::PhantomData;

use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{
        execution_control::ExecutionControl, ExecutionError, ExecutionState, InstructionExecutor,
        MatrixRecordArena, RowMajorMatrixArena, TraceStep, VmChipComplex, VmConfig, VmSegmentState,
        VmStateMut,
    },
    system::memory::INITIAL_TIMESTAMP,
};

#[derive(Default)]
pub struct TracegenCtx<F> {
    pub arenas: Vec<MatrixRecordArena<F>>,
    pub instret_end: Option<u64>,
}

impl<F> TracegenCtx<F>
where
    F: PrimeField32,
{
    pub fn new_with_capacity(capacities: &[(usize, usize)], instret_end: Option<u64>) -> Self {
        let arenas = capacities
            .iter()
            .map(|&(height, width)| MatrixRecordArena::with_capacity(height, width))
            .collect();

        Self {
            arenas,
            instret_end,
        }
    }
}

pub struct TracegenExecutionControl;

impl Default for TracegenExecutionControl {
    fn default() -> Self {
        Self {}
    }
}

impl<F, VC> ExecutionControl<F, VC> for TracegenExecutionControl
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    type Ctx = TracegenCtx<F>;

    fn initialize_context(&self) -> Self::Ctx {
        unreachable!()
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
        let &Instruction { opcode, .. } = instruction;

        if let Some(executor) = chip_complex.inventory.get_mut_executor(&opcode) {
            let memory = &mut chip_complex.base.memory_controller;
            let from_state = ExecutionState {
                pc: state.pc,
                timestamp: memory.timestamp(),
            };
            let to_state = executor.execute(
                memory,
                &mut state.streams,
                &mut state.rng,
                instruction,
                from_state,
            )?;
            state.pc = to_state.pc;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: state.pc,
                opcode,
            });
        };

        Ok(())
    }
}
