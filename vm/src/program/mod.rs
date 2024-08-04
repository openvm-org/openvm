use backtrace::Backtrace;
use p3_field::PrimeField64;

use crate::cpu::{
    trace::{ExecutionError, ExecutionError::PcOutOfBounds, Instruction},
    OpCode::FAIL,
};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Debug, Clone, Default)]
pub struct DebugInfo {
    pub dsl_instruction: String,
    pub trace: Option<Backtrace>,
}

impl DebugInfo {
    pub fn new(dsl_instruction: String, trace: Option<Backtrace>) -> Self {
        Self {
            dsl_instruction,
            trace,
        }
    }
}

#[derive(Clone)]
pub struct Program<F> {
    pub isa_instructions: Vec<Instruction<F>>,
    pub debug_info_vec: Vec<Option<DebugInfo>>,
}

impl<F> Program<F> {
    pub fn len(&self) -> usize {
        self.isa_instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.isa_instructions.is_empty()
    }
}

pub struct ProgramAir<F> {
    pub program: Program<F>,
}

pub struct ProgramChip<F> {
    pub air: ProgramAir<F>,
    pub true_program_length: usize,
    pub execution_frequencies: Vec<usize>,
}

impl<F: PrimeField64> ProgramChip<F> {
    pub fn new(mut program: Program<F>) -> Self {
        let true_program_length = program.len();
        while !program.len().is_power_of_two() {
            program
                .isa_instructions
                .push(Instruction::from_isize(FAIL, 0, 0, 0, 0, 0));
            program.debug_info_vec.push(None);
        }
        Self {
            execution_frequencies: vec![0; program.len()],
            true_program_length,
            air: ProgramAir { program },
        }
    }

    pub fn get_instruction(
        &mut self,
        pc: usize,
    ) -> Result<(Instruction<F>, Option<DebugInfo>), ExecutionError> {
        if !(0..self.true_program_length).contains(&pc) {
            return Err(PcOutOfBounds(pc, self.true_program_length));
        }
        self.execution_frequencies[pc] += 1;
        Ok((
            self.air.program.isa_instructions[pc].clone(),
            self.air.program.debug_info_vec[pc].clone(),
        ))
    }
}
