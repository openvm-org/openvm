use p3_field::PrimeField64;

use crate::cpu::trace::ExecutionError::PcOutOfBounds;
use crate::cpu::trace::{ExecutionError, Instruction};
use crate::cpu::OpCode::FAIL;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub struct ProgramAir<F> {
    pub program: Vec<Instruction<F>>,
}

pub struct ProgramChip<F> {
    pub air: ProgramAir<F>,
    pub true_program_length: usize,
    pub execution_frequencies: Vec<usize>,
}

impl<F: PrimeField64> ProgramChip<F> {
    pub fn new(mut program: Vec<Instruction<F>>) -> Self {
        let true_program_length = program.len();
        while !program.len().is_power_of_two() {
            program.push(Instruction::from_isize(FAIL, 0, 0, 0, 0, 0));
        }
        Self {
            execution_frequencies: vec![0; program.len()],
            true_program_length,
            air: ProgramAir { program },
        }
    }

    pub fn get_instruction(&mut self, pc: usize) -> Result<Instruction<F>, ExecutionError> {
        if !(0..self.true_program_length).contains(&pc) {
            return Err(PcOutOfBounds(pc, self.true_program_length));
        }
        self.execution_frequencies[pc] += 1;
        Ok(self.air.program[pc])
    }
}
