use std::{iter::repeat_n, sync::Arc};

#[cfg(not(feature = "parallel"))]
use itertools::Itertools;
use openvm_instructions::{instruction::Instruction, program::Program, LocalOpcode, SystemOpcode};
use openvm_stark_backend::{
    p3_field::{Field, PrimeField32},
    p3_maybe_rayon::prelude::*,
};

use crate::{
    arch::{
        execution_mode::PreflightCtx, interpreter::get_pc_index, Arena, ExecutionError, ExecutorId,
        ExecutorInventory, PreflightExecutor, StaticProgramError, VmExecState,
    },
    system::memory::online::TracingMemory,
};

/// VM preflight executor (E3 executor) for use with trace generation.
/// Note: This executor doesn't hold any VM state and can be used for multiple execution.
pub struct PreflightInterpretedInstance<F, E> {
    // NOTE[jpw]: we use an Arc so that VmInstance can hold both VirtualMachine and
    // PreflightInterpretedInstance. All we really need is to borrow `executors: &'a [E]`.
    inventory: Arc<ExecutorInventory<E>>,

    /// This is a map from (pc - pc_base) / pc_step -> [PcEntry].
    /// We will set `executor_idx` to `u32::MAX` in the [PcEntry] if the program has no instruction
    /// at that pc.
    // PERF[jpw/ayush]: We could map directly to the raw pointer(u64) for executor, but storing the
    // u32 may be better for cache efficiency.
    pc_handler: Vec<PcEntry<F>>,
    // pc_handler, execution_frequencies will all have the same length, which equals
    // `Program::len()`
    execution_frequencies: Vec<u32>,
    pc_base: u32,

    pub(super) executor_idx_to_air_idx: Vec<usize>,
}

#[repr(C)]
#[derive(Clone)]
pub struct PcEntry<F> {
    // NOTE[jpw]: revisit storing only smaller `precompute` for better cache locality. Currently
    // VmOpcode is usize so align=8 and there are 7 u32 operands so we store ExecutorId(u32) after
    // to avoid padding. This means PcEntry has align=8 and size=40 bytes, which is too big
    pub insn: Instruction<F>,
    pub executor_idx: ExecutorId,
}

impl<F: Field, E> PreflightInterpretedInstance<F, E> {
    /// Creates a new interpreter instance for preflight execution.
    /// Rewrites the program into an internal table specialized for enum dispatch.
    ///
    /// ## Assumption
    /// There are less than `u32::MAX` total AIRs.
    pub fn new(
        program: &Program<F>,
        inventory: Arc<ExecutorInventory<E>>,
        executor_idx_to_air_idx: Vec<usize>,
    ) -> Result<Self, StaticProgramError> {
        if inventory.executors().len() > u32::MAX as usize {
            // This would mean we cannot use u32::MAX as an "undefined" executor index
            return Err(StaticProgramError::TooManyExecutors);
        }
        let len = program.instructions_and_debug_infos.len();
        let pc_base = program.pc_base;
        let base_idx = get_pc_index(pc_base);
        let mut pc_handler = Vec::with_capacity(base_idx + len);
        pc_handler.extend(repeat_n(PcEntry::undefined(), base_idx));
        for insn_and_debug_info in &program.instructions_and_debug_infos {
            if let Some((insn, _)) = insn_and_debug_info {
                let insn = insn.clone();
                let executor_idx = if insn.opcode == SystemOpcode::TERMINATE.global_opcode() {
                    // The execution loop will always branch to terminate before using this executor
                    0
                } else {
                    *inventory.instruction_lookup.get(&insn.opcode).ok_or(
                        StaticProgramError::ExecutorNotFound {
                            opcode: insn.opcode,
                        },
                    )?
                };
                assert!(
                    (executor_idx as usize) < inventory.executors.len(),
                    "ExecutorInventory ensures executor_idx is in bounds"
                );
                let pc_entry = PcEntry { insn, executor_idx };
                pc_handler.push(pc_entry);
            } else {
                pc_handler.push(PcEntry::undefined());
            }
        }
        Ok(Self {
            inventory,
            execution_frequencies: vec![0u32; base_idx + len],
            pc_base,
            pc_handler,
            executor_idx_to_air_idx,
        })
    }

    pub fn executors(&self) -> &[E] {
        &self.inventory.executors
    }

    pub fn filtered_execution_frequencies(&self) -> Vec<u32> {
        let base_idx = get_pc_index(self.pc_base);
        self.pc_handler
            .par_iter()
            .zip_eq(&self.execution_frequencies)
            .skip(base_idx)
            .filter_map(|(entry, freq)| entry.is_some().then_some(*freq))
            .collect()
    }

    pub fn reset_execution_frequencies(&mut self) {
        self.execution_frequencies.fill(0);
    }
}

impl<F: PrimeField32, E> PreflightInterpretedInstance<F, E> {
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
                .is_some_and(|instret_end| state.instret() >= instret_end)
            {
                // should suspend
                break;
            }

            // Fetch, decode and execute single instruction
            self.execute_instruction(state)?;
            *state.instret_mut() += 1;
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
        let pc = state.pc();
        let pc_idx = get_pc_index(pc);
        let pc_entry = self
            .pc_handler
            .get(pc_idx)
            .ok_or_else(|| ExecutionError::PcOutOfBounds(pc))?;
        // SAFETY: `execution_frequencies` has the same length as `pc_handler` so `get_pc_entry`
        // already does the bounds check
        unsafe {
            *self.execution_frequencies.get_unchecked_mut(pc_idx) += 1;
        };
        // SAFETY: the `executor_idx` comes from ExecutorInventory, which ensures that
        // `executor_idx` is within bounds
        let executor = unsafe {
            self.inventory
                .executors
                .get_unchecked(pc_entry.executor_idx as usize)
        };
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
        let vm_state_mut = state.vm_state.into_mut(arena);
        executor.execute(vm_state_mut, &pc_entry.insn)?;

        #[cfg(feature = "metrics")]
        {
            crate::metrics::update_instruction_metrics(state, executor, pc, pc_entry);
        }

        Ok(())
    }
}

impl<F> PcEntry<F> {
    pub fn is_some(&self) -> bool {
        self.executor_idx != u32::MAX
    }
}

impl<F: Default> PcEntry<F> {
    fn undefined() -> Self {
        Self {
            insn: Instruction::default(),
            executor_idx: u32::MAX,
        }
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
        let start_instret = $state.instret();

        let result = $executor.execute_from_state($state);

        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed();
            let insns = $state.instret() - start_instret;
            tracing::info!("instructions_executed={insns}");
            metrics::counter!(concat!($name, "_insns")).absolute(insns);
            metrics::gauge!(concat!($name, "_insn_mi/s"))
                .set(insns as f64 / elapsed.as_micros() as f64);
        }
        result
    }};
}
