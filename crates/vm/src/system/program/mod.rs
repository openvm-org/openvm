use openvm_instructions::{
    instruction::{DebugInfo, Instruction},
    program::Program,
};
use openvm_stark_backend::{
    config::StarkGenericConfig,
    p3_field::PrimeField64,
    p3_maybe_rayon::prelude::*,
    prover::{cpu::CpuBackend, types::CommittedTraceData},
    ChipUsageGetter,
};

use crate::{arch::ExecutionError, system::program::trace::padding_instruction};

#[cfg(test)]
pub mod tests;

mod air;
mod bus;
pub mod trace;

pub use air::*;
pub use bus::*;

const EXIT_CODE_FAIL: usize = 1;

// TODO[jpw]: will this still be needed with rewriting of ins lookups?
#[derive(Debug)]
pub struct ProgramHandler<F> {
    pub air: ProgramAir,
    pub program: Program<F>,
    pub true_program_length: usize,
    pub execution_frequencies: Vec<u32>,
}

impl<F: PrimeField64> ProgramHandler<F> {
    pub fn new(bus: ProgramBus) -> Self {
        Self {
            execution_frequencies: vec![],
            program: Program::default(),
            true_program_length: 0,
            air: ProgramAir { bus },
        }
    }

    pub fn new_with_program(program: Program<F>, bus: ProgramBus) -> Self {
        let mut ret = Self::new(bus);
        ret.set_program(program);
        ret
    }

    pub fn set_program(&mut self, mut program: Program<F>) {
        let true_program_length = program.len();
        let mut number_actual_instructions = program.num_defined_instructions();
        while !number_actual_instructions.is_power_of_two() {
            program.push_instruction(padding_instruction());
            number_actual_instructions += 1;
        }
        self.true_program_length = true_program_length;
        self.execution_frequencies = vec![0; program.len()];
        self.program = program;
    }

    fn get_pc_index(&self, pc: u32) -> Result<usize, ExecutionError> {
        let step = self.program.step;
        let pc_base = self.program.pc_base;
        let pc_index = ((pc - pc_base) / step) as usize;
        if !(0..self.true_program_length).contains(&pc_index) {
            return Err(ExecutionError::PcOutOfBounds {
                pc,
                step,
                pc_base,
                program_len: self.true_program_length,
            });
        }
        Ok(pc_index)
    }

    pub fn get_instruction(
        &mut self,
        pc: u32,
    ) -> Result<&(Instruction<F>, Option<DebugInfo>), ExecutionError> {
        let pc_index = self.get_pc_index(pc)?;
        self.execution_frequencies[pc_index] += 1;
        self.program
            .get_instruction_and_debug_info(pc_index)
            .ok_or(ExecutionError::PcNotFound {
                pc,
                step: self.program.step,
                pc_base: self.program.pc_base,
                program_len: self.program.len(),
            })
    }

    pub fn filtered_execution_frequencies(&self) -> Vec<u32> {
        self.program
            .instructions_and_debug_infos
            .par_iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.is_some().then(|| self.execution_frequencies[i]))
            .collect()
    }
}

impl<F: PrimeField64> ChipUsageGetter for ProgramHandler<F> {
    fn air_name(&self) -> String {
        "ProgramChip".to_string()
    }

    fn constant_trace_height(&self) -> Option<usize> {
        Some(self.true_program_length.next_power_of_two())
    }

    fn current_trace_height(&self) -> usize {
        self.true_program_length
    }

    fn trace_width(&self) -> usize {
        1
    }
}

// For CPU backend only
pub struct ProgramChip<SC: StarkGenericConfig> {
    /// `i` -> frequency of instruction in `i`th row of trace matrix. This requires filtering
    /// `program.instructions_and_debug_infos` to remove gaps.
    pub filtered_exec_frequencies: Vec<u32>,
    pub cached: CommittedTraceData<CpuBackend<SC>>,
}
