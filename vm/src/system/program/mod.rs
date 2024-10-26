use std::{error::Error, fmt::Display};

use afs_stark_backend::ChipUsageGetter;
pub use air::*;
use axvm_instructions::{
    instruction::{DebugInfo, Instruction},
    program::Program,
};
pub use bus::*;
use p3_field::PrimeField64;

use crate::system::{program::trace::padding_instruction, vm::chip_set::READ_INSTRUCTION_BUS};

#[cfg(test)]
pub mod tests;

mod air;
mod bus;
pub mod trace;
pub mod util;

const EXIT_CODE_FAIL: usize = 1;

#[derive(Debug)]
pub enum ExecutionError {
    Fail(u32),
    PcOutOfBounds(u32, u32, u32, usize),
    DisabledOperation(u32, usize),
    HintOutOfBounds(u32),
    EndOfInputStream(u32),
    PublicValueIndexOutOfBounds(u32, usize, usize),
    PublicValueNotEqual(u32, usize, usize, usize),
}

impl Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionError::Fail(pc) => write!(f, "execution failed at pc = {}", pc),
            ExecutionError::PcOutOfBounds(pc, step, pc_base, program_len) => write!(
                f,
                "pc = {} out of bounds for program of length {}, with pc_base = {} and step = {}",
                pc, program_len, pc_base, step
            ),
            ExecutionError::DisabledOperation(pc, op) => {
                write!(f, "at pc = {}, opcode {:?} was not enabled", pc, op)
            }
            ExecutionError::HintOutOfBounds(pc) => write!(f, "at pc = {}", pc),
            ExecutionError::EndOfInputStream(pc) => write!(f, "at pc = {}", pc),
            ExecutionError::PublicValueIndexOutOfBounds(
                pc,
                num_public_values,
                public_value_index,
            ) => write!(
                f,
                "at pc = {}, tried to publish into index {} when num_public_values = {}",
                pc, public_value_index, num_public_values
            ),
            ExecutionError::PublicValueNotEqual(
                pc,
                public_value_index,
                existing_value,
                new_value,
            ) => write!(
                f,
                "at pc = {}, tried to publish value {} into index {}, but already had {}",
                pc, new_value, public_value_index, existing_value
            ),
        }
    }
}

impl Error for ExecutionError {}

#[derive(Debug)]
pub struct ProgramChip<F> {
    pub air: ProgramAir,
    pub program: Program<F>,
    pub true_program_length: usize,
    pub execution_frequencies: Vec<usize>,
}

impl<F: PrimeField64> Default for ProgramChip<F> {
    fn default() -> Self {
        Self {
            execution_frequencies: vec![],
            program: Program::default(),
            true_program_length: 0,
            air: ProgramAir {
                bus: ProgramBus(READ_INSTRUCTION_BUS),
            },
        }
    }
}

impl<F: PrimeField64> ProgramChip<F> {
    pub fn new_with_program(program: Program<F>) -> Self {
        let mut ret = Self::default();
        ret.set_program(program);
        ret
    }

    pub fn set_program(&mut self, mut program: Program<F>) {
        let true_program_length = program.len();
        while !program.len().is_power_of_two() {
            program.instructions_and_debug_infos.insert(
                program.pc_base + program.len() as u32 * program.step,
                (padding_instruction(), None),
            );
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
            return Err(ExecutionError::PcOutOfBounds(
                pc,
                step,
                pc_base,
                self.true_program_length,
            ));
        }
        Ok(pc_index)
    }

    pub fn get_instruction(
        &mut self,
        pc: u32,
    ) -> Result<(Instruction<F>, Option<DebugInfo>), ExecutionError> {
        let pc_index = self.get_pc_index(pc)?;
        self.execution_frequencies[pc_index] += 1;
        Ok(self.program.instructions_and_debug_infos[&pc].clone())
    }
}

impl<F: PrimeField64> ChipUsageGetter for ProgramChip<F> {
    fn air_name(&self) -> String {
        "ProgramChip".to_string()
    }

    fn current_trace_height(&self) -> usize {
        self.true_program_length
    }

    fn trace_width(&self) -> usize {
        1
    }
}
