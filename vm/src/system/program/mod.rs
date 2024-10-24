use std::{collections::HashMap, error::Error, fmt::Display, sync::Arc};

use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    prover::{helper::AirProofInputTestHelper, types::AirProofInput},
};
use axvm_instructions::{TerminateOpcode, UsizeOpcode};
use backtrace::Backtrace;
use itertools::Itertools;
use p3_field::{Field, PrimeField64};

use crate::{arch::NUM_OPERANDS, kernels::core::READ_INSTRUCTION_BUS};

#[cfg(test)]
pub mod tests;

mod air;
mod bus;
mod trace;
pub mod util;

pub use air::*;
pub use bus::*;

use super::PC_BITS;

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug, PartialEq, Eq, derive_new::new)]
pub struct Instruction<F> {
    pub opcode: usize,
    pub a: F,
    pub b: F,
    pub c: F,
    pub d: F,
    pub e: F,
    pub f: F,
    pub g: F,
    pub debug: String,
}

pub fn isize_to_field<F: Field>(value: isize) -> F {
    if value < 0 {
        return F::neg_one() * F::from_canonical_usize(value.unsigned_abs());
    }
    F::from_canonical_usize(value as usize)
}

impl<F: Field> Instruction<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn from_isize(opcode: usize, a: isize, b: isize, c: isize, d: isize, e: isize) -> Self {
        Self {
            opcode,
            a: isize_to_field::<F>(a),
            b: isize_to_field::<F>(b),
            c: isize_to_field::<F>(c),
            d: isize_to_field::<F>(d),
            e: isize_to_field::<F>(e),
            f: isize_to_field::<F>(0),
            g: isize_to_field::<F>(0),
            debug: String::new(),
        }
    }

    pub fn from_usize<const N: usize>(opcode: usize, operands: [usize; N]) -> Self {
        let mut operands = operands.to_vec();
        while operands.len() < NUM_OPERANDS {
            operands.push(0);
        }
        let operands = operands
            .into_iter()
            .map(F::from_canonical_usize)
            .collect_vec();
        Self {
            opcode,
            a: operands[0],
            b: operands[1],
            c: operands[2],
            d: operands[3],
            e: operands[4],
            f: operands[5],
            g: operands[6],
            debug: String::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn large_from_isize(
        opcode: usize,
        a: isize,
        b: isize,
        c: isize,
        d: isize,
        e: isize,
        f: isize,
        g: isize,
    ) -> Self {
        Self {
            opcode,
            a: isize_to_field::<F>(a),
            b: isize_to_field::<F>(b),
            c: isize_to_field::<F>(c),
            d: isize_to_field::<F>(d),
            e: isize_to_field::<F>(e),
            f: isize_to_field::<F>(f),
            g: isize_to_field::<F>(g),
            debug: String::new(),
        }
    }

    pub fn debug(opcode: usize, debug: &str) -> Self {
        Self {
            opcode,
            a: F::zero(),
            b: F::zero(),
            c: F::zero(),
            d: F::zero(),
            e: F::zero(),
            f: F::zero(),
            g: F::zero(),
            debug: String::from(debug),
        }
    }
}

impl<T: Default> Default for Instruction<T> {
    fn default() -> Self {
        Self {
            opcode: 0, // there is no real default opcode, this field must always be set
            a: T::default(),
            b: T::default(),
            c: T::default(),
            d: T::default(),
            e: T::default(),
            f: T::default(),
            g: T::default(),
            debug: String::new(),
        }
    }
}

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

#[derive(Clone, Debug, Default)]
pub struct Program<F> {
    /// A map from program counter to instruction.
    /// Sometimes the instructions are enumerated as 0, 4, 8, etc.
    /// Maybe at some point we will replace this with a struct that would have a `Vec` under the hood and divide the incoming `pc` by whatever given.
    pub instructions_and_debug_infos: HashMap<u32, (Instruction<F>, Option<DebugInfo>)>,
    pub step: u32,

    // these two are needed to calculate the index for execution_frequencies
    pub pc_start: u32,
    pub pc_base: u32,
}

const MAX_ALLOWED_PC: u32 = (1 << PC_BITS) - 1;

impl<F> Program<F> {
    pub fn from_instructions_and_step(
        instructions: &[Instruction<F>],
        step: u32,
        pc_start: u32,
        pc_base: u32,
    ) -> Self
    where
        F: Clone,
    {
        assert!(
            instructions.is_empty()
                || pc_base + (instructions.len() as u32 - 1) * step <= MAX_ALLOWED_PC
        );
        Self {
            instructions_and_debug_infos: instructions
                .iter()
                .enumerate()
                .map(|(index, instruction)| {
                    (
                        index as u32 * step + pc_base,
                        ((*instruction).clone(), None),
                    )
                })
                .collect(),
            step,
            pc_start,
            pc_base,
        }
    }

    // We assume that pc_start = pc_base = 0 everywhere except the RISC-V programs, until we need otherwise
    pub fn from_instructions_and_debug_infos(
        instructions: &[Instruction<F>],
        debug_infos: &[Option<DebugInfo>],
    ) -> Self
    where
        F: Clone,
    {
        assert!(instructions.is_empty() || instructions.len() as u32 - 1 <= MAX_ALLOWED_PC);
        Self {
            instructions_and_debug_infos: instructions
                .iter()
                .zip(debug_infos.iter())
                .enumerate()
                .map(|(index, (instruction, debug_info))| {
                    (
                        index as u32,
                        ((*instruction).clone(), (*debug_info).clone()),
                    )
                })
                .collect(),
            step: 1,
            pc_start: 0,
            pc_base: 0,
        }
    }

    pub fn from_instructions(instructions: &[Instruction<F>]) -> Self
    where
        F: Clone,
    {
        Self::from_instructions_and_step(instructions, 1, 0, 0)
    }

    pub fn len(&self) -> usize {
        self.instructions_and_debug_infos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions_and_debug_infos.is_empty()
    }

    pub fn instructions(&self) -> Vec<Instruction<F>>
    where
        F: Clone,
    {
        self.instructions_and_debug_infos
            .iter()
            .sorted_by_key(|(pc, _)| *pc)
            .map(|(_, (instruction, _))| instruction)
            .cloned()
            .collect()
    }

    pub fn debug_infos(&self) -> Vec<Option<DebugInfo>> {
        self.instructions_and_debug_infos
            .iter()
            .sorted_by_key(|(pc, _)| *pc)
            .map(|(_, (_, debug_info))| debug_info)
            .cloned()
            .collect()
    }
}

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
        const EXIT_CODE_FAIL: usize = 1;
        while !program.len().is_power_of_two() {
            program.instructions_and_debug_infos.insert(
                program.pc_base + program.len() as u32 * program.step,
                (
                    Instruction::from_usize(
                        TerminateOpcode::TERMINATE.with_default_offset(),
                        [0, 0, EXIT_CODE_FAIL],
                    ),
                    None,
                ),
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

impl<SC: StarkGenericConfig> From<ProgramChip<Val<SC>>> for AirProofInput<SC>
where
    Val<SC>: PrimeField64,
{
    fn from(program_chip: ProgramChip<Val<SC>>) -> Self {
        let air = program_chip.air.clone();
        let cached_trace = program_chip.generate_cached_trace();
        let common_trace = program_chip.generate_trace();
        AirProofInput::cached_traces_no_pis(Arc::new(air), vec![cached_trace], common_trace)
    }
}
