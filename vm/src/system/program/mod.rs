use std::{error::Error, fmt::Display};

use ax_stark_backend::ChipUsageGetter;
use axvm_instructions::{
    instruction::{DebugInfo, Instruction},
    program::Program,
};
use p3_field::PrimeField64;

use crate::system::program::trace::padding_instruction;

#[cfg(test)]
pub mod tests;

mod air;
mod bus;
pub mod trace;
pub mod util;

pub use air::*;
pub use bus::*;

const EXIT_CODE_FAIL: usize = 1;

#[derive(Debug)]
pub enum ExecutionError {
    /// pc
    Fail(u32),
    /// pc, step, pc_base, program_len
    PcNotFound(u32, u32, u32, usize),
    /// pc, step, pc_base, program_len
    PcOutOfBounds(u32, u32, u32, usize),
    /// pc, phantom_repr
    InvalidPhantomInstruction(u32, u16),
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
            ExecutionError::PcNotFound(pc, step, pc_base, program_len) => write!(
                f,
                "pc = {} not found for program of length {}, with pc_base = {} and step = {}",
                pc, program_len, pc_base, step
            ),
            ExecutionError::PcOutOfBounds(pc, step, pc_base, program_len) => write!(
                f,
                "pc = {} out of bounds for program of length {}, with pc_base = {} and step = {}",
                pc, program_len, pc_base, step
            ),
            ExecutionError::InvalidPhantomInstruction(pc, phantom_repr) => write!(
                f,
                "at pc = {}, invalid phantom instruction {:?}",
                pc, phantom_repr
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

impl<F: PrimeField64> ProgramChip<F> {
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
        while !program.len().is_power_of_two() {
            program.push_instruction(padding_instruction());
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
        self.program
            .get_instruction_and_debug_info(pc_index)
            .ok_or(ExecutionError::PcNotFound(
                pc,
                self.program.step,
                self.program.pc_base,
                self.program.len(),
            ))
    }
}

impl<F: PrimeField64> ChipUsageGetter for ProgramChip<F> {
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
