use std::marker::PhantomData;

use crate::{
    arch::{Arena, ExecutionError, InstructionExecutor, VmSegmentState, VmStateMut},
    system::{memory::online::TracingMemory, program::PcEntry},
};

pub struct TracegenCtx<RA> {
    pub arenas: Vec<RA>,
    pub instret_end: Option<u64>,
}

impl<RA: Arena> TracegenCtx<RA> {
    /// `capacities` is list of `(height, width)` dimensions for each arena, indexed by AIR index.
    /// The length of `capacities` must equal the number of AIRs.
    /// Here `height` will always mean an overestimate of the trace height for that AIR, while
    /// `width` may have different meanings depending on the `RA` type.
    pub fn new_with_capacity(capacities: &[(usize, usize)], instret_end: Option<u64>) -> Self {
        let arenas = capacities
            .iter()
            .map(|&(height, main_width)| RA::with_capacity(height, main_width))
            .collect();

        Self {
            arenas,
            instret_end,
        }
    }
}

pub struct TracegenExecutionControl<F> {
    executor_idx_to_air_idx: Vec<usize>,
    phantom: PhantomData<F>,
}

impl<F> TracegenExecutionControl<F> {
    pub fn new(executor_idx_to_air_idx: Vec<usize>) -> Self {
        Self {
            executor_idx_to_air_idx,
            phantom: PhantomData,
        }
    }
}

impl<F> TracegenExecutionControl<F> {
    #[inline(always)]
    pub fn should_suspend<RA>(
        &self,
        state: &mut VmSegmentState<F, TracingMemory, TracegenCtx<RA>>,
    ) -> bool {
        state
            .ctx
            .instret_end
            .is_some_and(|instret_end| state.instret >= instret_end)
    }

    #[inline(always)]
    pub fn on_suspend_or_terminate<RA>(
        &self,
        _state: &mut VmSegmentState<F, TracingMemory, TracegenCtx<RA>>,
        _exit_code: Option<u32>,
    ) {
    }

    #[inline(always)]
    pub fn on_suspend<RA>(&self, state: &mut VmSegmentState<F, TracingMemory, TracegenCtx<RA>>) {
        self.on_suspend_or_terminate(state, None);
    }

    #[inline(always)]
    pub fn on_terminate<RA>(
        &self,
        state: &mut VmSegmentState<F, TracingMemory, TracegenCtx<RA>>,
        exit_code: u32,
    ) {
        self.on_suspend_or_terminate(state, Some(exit_code));
    }

    /// Execute a single instruction
    #[inline(always)]
    pub fn execute_instruction<RA, Executor>(
        &self,
        state: &mut VmSegmentState<F, TracingMemory, TracegenCtx<RA>>,
        executor: &mut Executor,
        pc_entry: &PcEntry<F>,
    ) -> Result<(), ExecutionError>
    where
        Executor: InstructionExecutor<F, RA>,
    {
        tracing::trace!(
            "opcode: {} | timestamp: {}",
            executor.get_opcode_name(pc_entry.insn.opcode.as_usize()),
            state.memory.timestamp()
        );
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
            pc: &mut state.vm_state.pc,
            memory: &mut state.vm_state.memory,
            streams: &mut state.vm_state.streams,
            rng: &mut state.vm_state.rng,
            ctx: arena,
            #[cfg(feature = "metrics")]
            metrics: &mut state.vm_state.metrics,
        };
        executor.execute(state_mut, &pc_entry.insn)?;

        Ok(())
    }
}
