use openvm_instructions::{
    instruction::{DebugInfo, Instruction},
    program::Program,
    VmOpcode,
};
use openvm_stark_backend::{
    config::StarkGenericConfig,
    p3_field::Field,
    p3_maybe_rayon::prelude::*,
    prover::{cpu::CpuBackend, types::CommittedTraceData},
};
use thiserror::Error;

use crate::arch::{ExecutionError, ExecutorId, ExecutorInventory};

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
    executors: Vec<E>,
    /// This is a map from (pc - pc_base) / pc_step -> [PcEntry].
    /// We will map to `u32::MAX` if the program has no instruction at that pc.
    // Perf[jpw/ayush]: We could map directly to the raw pointer(u64) for executor, but storing the
    // u32 may be better for cache efficiency.
    pc_handler: Vec<PcEntry<F>>,
    execution_frequencies: Vec<u32>,
    debug_infos: Vec<Option<DebugInfo>>,
    pc_base: u32,
    step: u32,
}

impl<F: Field, E> ProgramHandler<F, E> {
    /// Rewrite the program into compiled handlers.
    // @dev: We need to clone the executors because they are not completely stateless
    pub fn new(
        program: Program<F>,
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
        let mut debug_infos = Vec::with_capacity(len);
        for insn_and_debug_info in program.instructions_and_debug_infos {
            if let Some((insn, debug_info)) = insn_and_debug_info {
                let executor_idx = *inventory.instruction_lookup.get(&insn.opcode).ok_or(
                    StaticProgramError::ExecutorNotFound {
                        opcode: insn.opcode,
                    },
                )?;
                assert!(
                    (executor_idx as usize) < inventory.executors.len(),
                    "ExecutorInventory ensures executor_idx is in bounds"
                );
                let pc_entry = PcEntry { insn, executor_idx };
                pc_handler.push(pc_entry);
                debug_infos.push(debug_info);
            } else {
                pc_handler.push(PcEntry::undefined());
                debug_infos.push(None);
            }
        }

        Ok(Self {
            execution_frequencies: vec![0u32; len],
            executors: inventory.executors.clone(),
            pc_handler,
            debug_infos,
            pc_base: program.pc_base,
            step: program.step,
        })
    }

    #[inline(always)]
    fn get_pc_index(&self, pc: u32) -> usize {
        let step = self.step;
        let pc_base = self.pc_base;
        ((pc - pc_base) / step) as usize
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
                step: self.step,
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

    pub fn get_debug_info(&self, pc: u32) -> Result<&Option<DebugInfo>, ExecutionError> {
        let pc_idx = self.get_pc_index(pc);
        self.debug_infos
            .get(pc_idx)
            .ok_or_else(|| ExecutionError::PcOutOfBounds {
                pc: pc_idx as u32 * self.step + self.pc_base,
                step: self.step,
                pc_base: self.pc_base,
                program_len: self.pc_handler.len(),
            })
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

/// Errors in the program that can be statically analyzed before runtime.
#[derive(Error, Debug)]
pub enum StaticProgramError {
    #[error("Too many executors")]
    TooManyExecutors,
    #[error("Executor not found for opcode {opcode}")]
    ExecutorNotFound { opcode: VmOpcode },
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
