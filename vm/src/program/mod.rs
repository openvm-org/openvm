use p3_field::PrimeField64;

use crate::cpu::trace::Instruction;
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
    pub execution_frequencies: Vec<usize>,
}

impl<F: PrimeField64> ProgramChip<F> {
    pub fn new(mut program: Vec<Instruction<F>>) -> Self {
        while !program.len().is_power_of_two() {
            program.push(Instruction::from_isize(FAIL, 0, 0, 0, 0, 0));
        }
        Self {
            execution_frequencies: vec![0; program.len()],
            air: ProgramAir { program },
        }
    }

    pub fn get_instruction(&mut self, pc: usize) -> Instruction<F> {
        self.execution_frequencies[pc] += 1;
        self.air.program[pc]
    }
}
