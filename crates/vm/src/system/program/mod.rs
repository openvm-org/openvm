use openvm_instructions::{
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode, SystemOpcode,
};
use openvm_stark_backend::{
    config::StarkGenericConfig,
    p3_field::Field,
    p3_maybe_rayon::prelude::*,
    prover::{cpu::CpuBackend, types::CommittedTraceData},
};

use crate::arch::{ExecutionError, ExecutorId, ExecutorInventory, StaticProgramError};

#[cfg(test)]
pub mod tests;

mod air;
mod bus;
pub mod trace;

pub use air::*;
pub use bus::*;

const EXIT_CODE_FAIL: usize = 1;

#[repr(C)]
pub struct PcEntry<F> {
    // TODO[jpw]: revisit storing only smaller `precompute` for better cache locality. Currently
    // VmOpcode is usize so align=8 and there are 7 u32 operands so we store ExecutorId(u32) after
    // to avoid padding. This means PcEntry has align=8 and size=40 bytes, which is too big
    pub insn: Instruction<F>,
    pub executor_idx: ExecutorId,
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

// pc_handler, execution_frequencies, debug_infos will all have the same length, which equals
// `Program::len()`
pub struct ProgramHandler<F, E> {
    pub(crate) executors: Vec<E>,
    /// This is a map from (pc - pc_base) / pc_step -> [PcEntry].
    /// We will map to `u32::MAX` if the program has no instruction at that pc.
    // Perf[jpw/ayush]: We could map directly to the raw pointer(u64) for executor, but storing the
    // u32 may be better for cache efficiency.
    pc_handler: Vec<PcEntry<F>>,
    execution_frequencies: Vec<u32>,
    pc_base: u32,
}

impl<F: Field, E> ProgramHandler<F, E> {
    /// Rewrite the program into compiled handlers.
    ///
    /// ## Assumption
    /// There are less than `u32::MAX` total AIRs.
    // @dev: We need to clone the executors because they are not completely stateless
    pub fn new(
        program: &Program<F>,
        inventory: &ExecutorInventory<E>,
    ) -> Result<Self, StaticProgramError>
    where
        E: Clone,
    {
        if inventory.executors().len() > u32::MAX as usize {
            // This would mean we cannot use u32::MAX as an "undefined" executor index
            return Err(StaticProgramError::TooManyExecutors);
        }
        let len = program.instructions_and_debug_infos.len();
        let mut pc_handler = Vec::with_capacity(len);
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
        let executors = inventory.executors.clone();

        Ok(Self {
            execution_frequencies: vec![0u32; len],
            executors,
            pc_handler,
            pc_base: program.pc_base,
        })
    }

    #[inline(always)]
    fn get_pc_index(&self, pc: u32) -> usize {
        let pc_base = self.pc_base;
        ((pc - pc_base) / DEFAULT_PC_STEP) as usize
    }

    /// Returns `(executor, pc_entry, pc_idx)`.
    #[inline(always)]
    pub fn get_executor(&mut self, pc: u32) -> Result<(&mut E, &PcEntry<F>), ExecutionError> {
        let pc_idx = self.get_pc_index(pc);
        let entry = self
            .pc_handler
            .get(pc_idx)
            .ok_or_else(|| ExecutionError::PcOutOfBounds {
                pc,
                pc_base: self.pc_base,
                program_len: self.pc_handler.len(),
            })?;
        // SAFETY: `execution_frequencies` has the same length as `pc_handler` so `get_pc_entry`
        // already does the bounds check
        unsafe {
            *self.execution_frequencies.get_unchecked_mut(pc_idx) += 1;
        };
        // SAFETY: the `executor_idx` comes from ExecutorInventory, which ensures that
        // `executor_idx` is within bounds
        let executor = unsafe {
            self.executors
                .get_unchecked_mut(entry.executor_idx as usize)
        };

        Ok((executor, entry))
    }

    pub fn filtered_execution_frequencies(&self) -> Vec<u32>
    where
        E: Sync,
    {
        self.pc_handler
            .par_iter()
            .enumerate()
            .filter_map(|(i, entry)| entry.is_some().then(|| self.execution_frequencies[i]))
            .collect()
    }
}

// For CPU backend only
pub struct ProgramChip<SC: StarkGenericConfig> {
    /// `i` -> frequency of instruction in `i`th row of trace matrix. This requires filtering
    /// `program.instructions_and_debug_infos` to remove gaps.
    pub(super) filtered_exec_frequencies: Vec<u32>,
    pub(super) cached: Option<CommittedTraceData<CpuBackend<SC>>>,
}

impl<SC: StarkGenericConfig> ProgramChip<SC> {
    pub(super) fn unloaded() -> Self {
        Self {
            filtered_exec_frequencies: Vec::new(),
            cached: None,
        }
    }
}
