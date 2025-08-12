use openvm_stark_backend::p3_field::PrimeField32;

use super::ExecutionError;
use crate::{
    arch::{
        execution_mode::PreflightCtx, instructions::*, Arena, PreflightExecutor, VmExecState,
        VmStateMut,
    },
    system::{memory::online::TracingMemory, program::ProgramHandler},
};

pub struct PreflightInterpretedInstance<F, E> {
    pub handler: ProgramHandler<F, E>,
    executor_idx_to_air_idx: Vec<usize>,
}

impl<F, E> PreflightInterpretedInstance<F, E>
where
    F: PrimeField32,
{
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(handler: ProgramHandler<F, E>, executor_idx_to_air_idx: Vec<usize>) -> Self {
        Self {
            handler,
            executor_idx_to_air_idx,
        }
    }

    /// Stopping is triggered by should_stop() or if VM is terminated
    pub fn execute_from_state<RA>(
        &mut self,
        state: &mut VmExecState<F, TracingMemory, PreflightCtx<RA>>,
    ) -> Result<(), ExecutionError>
    where
        RA: Arena,
        E: PreflightExecutor<F, RA>,
    {
        loop {
            if let Ok(Some(_)) = state.exit_code {
                // should terminate
                break;
            }
            if state
                .ctx
                .instret_end
                .is_some_and(|instret_end| state.instret >= instret_end)
            {
                // should suspend
                break;
            }

            // Fetch, decode and execute single instruction
            self.execute_instruction(state)?;
            state.instret += 1;
        }

        Ok(())
    }

    /// Executes a single instruction and updates VM state
    #[inline(always)]
    fn execute_instruction<RA>(
        &mut self,
        state: &mut VmExecState<F, TracingMemory, PreflightCtx<RA>>,
    ) -> Result<(), ExecutionError>
    where
        RA: Arena,
        E: PreflightExecutor<F, RA>,
    {
        let pc = state.pc;
        let (executor, pc_entry) = self.handler.get_executor(pc)?;
        tracing::trace!("pc: {pc:#x} | {:?}", pc_entry.insn);

        let opcode = pc_entry.insn.opcode;
        let c = pc_entry.insn.c;
        // Handle termination instruction
        if opcode.as_usize() == SystemOpcode::CLASS_OFFSET + SystemOpcode::TERMINATE as usize {
            state.exit_code = Ok(Some(c.as_canonical_u32()));
            return Ok(());
        }

        // Execute the instruction using the control implementation
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
            custom_pvs: &mut state.vm_state.custom_pvs,
            ctx: arena,
            #[cfg(feature = "metrics")]
            metrics: &mut state.vm_state.metrics,
        };
        executor.execute(state_mut, &pc_entry.insn)?;

        #[cfg(feature = "metrics")]
        {
            crate::metrics::update_instruction_metrics(state, executor, pc, pc_entry);
        }

        Ok(())
    }
}

/// Macro for executing and emitting metrics for instructions/s and number of instructions executed.
/// Does not include any tracing span.
#[macro_export]
macro_rules! execute_spanned {
    ($name:literal, $executor:expr, $state:expr) => {{
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        #[cfg(feature = "metrics")]
        let start_instret = $state.instret;

        let result = $executor.execute_from_state($state);

        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed();
            let insns = $state.instret - start_instret;
            metrics::counter!("insns").absolute(insns);
            metrics::gauge!(concat!($name, "_insn_mi/s"))
                .set(insns as f64 / elapsed.as_micros() as f64);
        }
        result
    }};
}
