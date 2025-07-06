use std::marker::PhantomData;

use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{
        execution_control::ExecutionControl, ExecutionError, InstructionExecutor, VmSegmentState,
        VmStateMut,
    },
    system::{memory::online::TracingMemory, program::ProgramHandler},
};

pub struct TracegenCtx<RA> {
    pub arenas: Vec<RA>,
    pub instret_end: Option<u64>,
}

impl<RA> TracegenCtx<RA> {
    // /// `capacities` is list of `(height, width)` dimensions for each arena.
    // pub fn new_with_capacity(capacities: &[(usize, usize)], instret_end: Option<u64>) -> Self {
    //     let arenas = capacities
    //         .iter()
    //         .map(|&(height, width)| {
    //             MatrixRecordArena::with_capacity(next_power_of_two_or_zero(height), width)
    //         })
    //         .collect();

    //     Self {
    //         arenas,
    //         instret_end,
    //     }
    // }
}

pub struct TracegenExecutionControl<RA> {
    executor_idx_to_air_idx: Vec<usize>,
    phantom: PhantomData<RA>,
}

impl<RA> TracegenExecutionControl<RA> {
    pub fn new(executor_idx_to_air_idx: Vec<usize>) -> Self {
        Self {
            executor_idx_to_air_idx,
            phantom: PhantomData,
        }
    }
}

impl<F, RA, Executor> ExecutionControl<F, Executor> for TracegenExecutionControl<RA>
where
    F: PrimeField32,
    Executor: InstructionExecutor<F, RA>,
{
    type Memory = TracingMemory<F>;
    type Ctx = TracegenCtx<RA>;

    fn should_suspend(&self, state: &mut VmSegmentState<F, Self::Memory, Self::Ctx>) -> bool {
        state
            .ctx
            .instret_end
            .is_some_and(|instret_end| state.instret >= instret_end)
    }

    fn on_suspend_or_terminate(
        &self,
        _state: &mut VmSegmentState<F, Self::Memory, Self::Ctx>,
        _exit_code: Option<u32>,
    ) {
        // let timestamp = state.memory.timestamp();
        // chip_complex
        //     .connector_chip_mut()
        //     .end(ExecutionState::new(state.pc, timestamp), exit_code);
    }

    /// Execute a single instruction
    #[inline(always)]
    fn execute_instruction(
        &self,
        state: &mut VmSegmentState<F, Self::Memory, Self::Ctx>,
        handler: &mut ProgramHandler<F, Executor>,
    ) -> Result<(), ExecutionError> {
        let (executor, pc_entry) = handler.get_executor(state.pc)?;
        let arena = unsafe {
            // SAFETY: executor_idx is guarantee to be within bounds by ProgramHandler constructor
            let air_idx = *self
                .executor_idx_to_air_idx
                .get_unchecked(pc_entry.executor_idx as usize);
            // SAFETY: air_idx is a valid AIR index in the vkey, and always construct arenas with
            // length equal to num_airs
            state.ctx.arenas.get_unchecked_mut(air_idx)
        };
        let state_mut = VmStateMut {
            pc: &mut state.pc,
            memory: &mut state.memory,
            streams: &mut state.streams,
            rng: &mut state.rng,
            ctx: arena,
        };
        executor.execute(state_mut, &pc_entry.insn)?;

        Ok(())
    }
}
