use std::marker::PhantomData;

use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{
        execution_control::ExecutionControl, ExecutionError, ExecutionState, MatrixRecordArena,
        RowMajorMatrixArena, TraceStep, VmChipComplex, VmConfig, VmSegmentState, VmStateMut,
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
    pub fn new(count: usize) -> Self {
        let arenas = (0..count)
            .map(|_| MatrixRecordArena::with_capacity(0, 0))
            .collect();
        Self {
            arenas,
            instret_end: None,
        }
    }

    pub fn new_with_capacity(capacities: &[(usize, usize)]) -> Self {
        let arenas = capacities
            .iter()
            .map(|&(height, width)| MatrixRecordArena::with_capacity(height, width))
            .collect();

        Self {
            arenas,
            instret_end: None,
        }
    }
}

pub struct TracegenExecutionControl<F> {
    phantom: PhantomData<F>,
}

impl<F> Default for TracegenExecutionControl<F> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F, VC> ExecutionControl<F, VC> for TracegenExecutionControl<F>
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
        let timestamp = chip_complex.memory_controller().timestamp();

        let &Instruction { opcode, .. } = instruction;

        if let Some(executor) = chip_complex.inventory.get_mut_executor(&opcode) {
            let memory = &mut chip_complex.base.memory_controller.memory;
            let mut pc = state.pc;
            let chip_index = executor.chip_index;
            let arena = &mut state.ctx.arenas[chip_index];
            let state_mut = VmStateMut {
                pc: &mut pc,
                memory,
                streams: &mut state.streams,
                rng: &mut state.rng,
                ctx: arena,
            };
            executor.step.execute(state_mut, instruction)?;
            state.pc = pc;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: state.pc,
                opcode,
            });
        };

        Ok(())
    }
}
