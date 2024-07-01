use p3_field::PrimeField64;

use crate::cpu::trace::Instruction;
use crate::cpu::OpCode::FAIL;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub struct ProgramAir<T> {
    pub program: Vec<Instruction<T>>,
}

impl<F: PrimeField64> ProgramAir<F> {
    pub fn new(mut program: Vec<Instruction<F>>) -> Self {
        // in order to make program length a power of 2,
        // add instructions that jump to themselves
        // so that any program that tries to jump to instructions that shouldn't exist
        // will enter an infinite loop (so their termination cannot be proven)
        while !program.len().is_power_of_two() {
            // op_c, as_c never matter in JAL
            // op_a doesn't matter here (random address to write garbage to)
            // op_b is the offset, needs to be 0 so we jump to self
            // as_b should be nonzero so we don't write to immediate (may be unsupported)
            program.push(Instruction::from_isize(FAIL, 0, 0, 0, 0, 0));
        }

        Self { program }
    }
}
