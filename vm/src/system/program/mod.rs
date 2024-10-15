use std::{collections::HashMap, error::Error, fmt::Display, sync::Arc};

use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    prover::{helper::AirProofInputTestHelper, types::AirProofInput},
};
use backtrace::Backtrace;
use bridge::ProgramBus;
use itertools::Itertools;
use p3_field::{Field, PrimeField64};

use crate::{
    arch::{
        instructions::CoreOpcode::{FAIL, NOP},
        NUM_OPERANDS,
    },
    kernels::core::READ_INSTRUCTION_BUS,
};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;
pub mod util;

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug, PartialEq, Eq, derive_new::new)]
pub struct Instruction<F> {
    pub opcode: usize,
    pub op_a: F,
    pub op_b: F,
    pub op_c: F,
    pub d: F,
    pub e: F,
    pub op_f: F,
    pub op_g: F,
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
    pub fn from_isize(
        opcode: usize,
        op_a: isize,
        op_b: isize,
        op_c: isize,
        d: isize,
        e: isize,
    ) -> Self {
        Self {
            opcode,
            op_a: isize_to_field::<F>(op_a),
            op_b: isize_to_field::<F>(op_b),
            op_c: isize_to_field::<F>(op_c),
            d: isize_to_field::<F>(d),
            e: isize_to_field::<F>(e),
            op_f: isize_to_field::<F>(0),
            op_g: isize_to_field::<F>(0),
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
            op_a: operands[0],
            op_b: operands[1],
            op_c: operands[2],
            d: operands[3],
            e: operands[4],
            op_f: operands[5],
            op_g: operands[6],
            debug: String::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn large_from_isize(
        opcode: usize,
        op_a: isize,
        op_b: isize,
        op_c: isize,
        d: isize,
        e: isize,
        op_f: isize,
        op_g: isize,
    ) -> Self {
        Self {
            opcode,
            op_a: isize_to_field::<F>(op_a),
            op_b: isize_to_field::<F>(op_b),
            op_c: isize_to_field::<F>(op_c),
            d: isize_to_field::<F>(d),
            e: isize_to_field::<F>(e),
            op_f: isize_to_field::<F>(op_f),
            op_g: isize_to_field::<F>(op_g),
            debug: String::new(),
        }
    }

    pub fn debug(opcode: usize, debug: &str) -> Self {
        Self {
            opcode,
            op_a: F::zero(),
            op_b: F::zero(),
            op_c: F::zero(),
            d: F::zero(),
            e: F::zero(),
            op_f: F::zero(),
            op_g: F::zero(),
            debug: String::from(debug),
        }
    }
}

impl<T: Default> Default for Instruction<T> {
    fn default() -> Self {
        Self {
            opcode: NOP as usize,
            op_a: T::default(),
            op_b: T::default(),
            op_c: T::default(),
            d: T::default(),
            e: T::default(),
            op_f: T::default(),
            op_g: T::default(),
            debug: String::new(),
        }
    }
}

#[derive(Debug)]
pub enum ExecutionError {
    Fail(usize),
    PcOutOfBounds(usize, usize),
    DisabledOperation(usize, usize),
    HintOutOfBounds(usize),
    EndOfInputStream(usize),
    PublicValueIndexOutOfBounds(usize, usize, usize),
    PublicValueNotEqual(usize, usize, usize, usize),
}

impl Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionError::Fail(pc) => write!(f, "execution failed at pc = {}", pc),
            ExecutionError::PcOutOfBounds(pc, program_len) => write!(
                f,
                "pc = {} out of bounds for program of length {}",
                pc, program_len
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

#[derive(Clone, Debug)]
pub struct Program<F> {
    /// A map from program counter to instruction.
    /// Sometimes the instructions are enumerated as 0, 4, 8, etc.
    /// Maybe at some point we will replace this with a struct that would have a `Vec` under the hood and divide the incoming `pc` by whatever given.
    pub instructions_and_debug_infos: HashMap<usize, (Instruction<F>, Option<DebugInfo>)>,
    pub step: usize,
}

impl<F> Program<F> {
    pub fn from_instructions_and_step(instructions: &[Instruction<F>], step: usize) -> Self
    where
        F: Clone,
    {
        Self {
            instructions_and_debug_infos: instructions
                .iter()
                .enumerate()
                .map(|(index, instruction)| (index * step, ((*instruction).clone(), None)))
                .collect(),
            step,
        }
    }

    pub fn from_instructions_and_debug_infos(
        instructions: &[Instruction<F>],
        debug_infos: &[Option<DebugInfo>],
    ) -> Self
    where
        F: Clone,
    {
        Self {
            instructions_and_debug_infos: instructions
                .iter()
                .zip(debug_infos.iter())
                .enumerate()
                .map(|(index, (instruction, debug_info))| {
                    (index, ((*instruction).clone(), (*debug_info).clone()))
                })
                .collect(),
            step: 1,
        }
    }

    pub fn from_instructions(instructions: &[Instruction<F>]) -> Self
    where
        F: Clone,
    {
        Self::from_instructions_and_step(instructions, 1)
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

#[derive(Clone, Debug)]
pub struct ProgramAir {
    bus: ProgramBus,
}

#[derive(Debug)]
pub struct ProgramChip<F> {
    pub air: ProgramAir,
    pub program: Program<F>,
    pub true_program_length: usize,
    pub execution_frequencies: Vec<usize>,
}

impl<F: PrimeField64> ProgramChip<F> {
    pub fn new(mut program: Program<F>) -> Self {
        let true_program_length = program.len();
        while !program.len().is_power_of_two() {
            program.instructions_and_debug_infos.insert(
                program.len() * program.step,
                (Instruction::from_isize(FAIL as usize, 0, 0, 0, 0, 0), None),
            );
        }
        Self {
            execution_frequencies: vec![0; program.len()],
            program,
            true_program_length,
            air: ProgramAir {
                bus: ProgramBus(READ_INSTRUCTION_BUS),
            },
        }
    }

    pub fn get_instruction(
        &mut self,
        pc: usize,
    ) -> Result<(Instruction<F>, Option<DebugInfo>), ExecutionError> {
        if !(0..self.true_program_length).contains(&pc) {
            return Err(ExecutionError::PcOutOfBounds(pc, self.true_program_length));
        }
        self.execution_frequencies[pc] += 1;
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
