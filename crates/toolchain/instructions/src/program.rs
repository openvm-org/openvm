use std::{fmt, fmt::Display};

use itertools::Itertools;
use openvm_stark_backend::p3_field::Field;
use serde::{Deserialize, Serialize};

use crate::instruction::{DebugInfo, Instruction};

pub const PC_BITS: usize = 30;
/// We use default PC step of 4 whenever possible for consistency with RISC-V, where 4 comes
/// from the fact that each standard RISC-V instruction is 32-bits = 4 bytes.
pub const DEFAULT_PC_STEP: u32 = 4;
pub const DEFAULT_MAX_NUM_PUBLIC_VALUES: usize = 32;
const MAX_ALLOWED_PC: u32 = (1 << PC_BITS) - 1;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Program<F> {
    /// A map from program counter to instruction.
    /// Sometimes the instructions are enumerated as 0, 4, 8, etc.
    /// Maybe at some point we will replace this with a struct that would have a `Vec` under the hood and divide the incoming `pc` by whatever given.
    instructions_and_debug_infos: Vec<Option<(Instruction<F>, Option<DebugInfo>)>>,
    pub step: u32,
    pub pc_base: u32,
    /// The upper bound of the number of public values the program would publish.
    /// Currently, this won't result any constraint. But users should always be aware of the limit
    /// of public values when they write programs.
    pub max_num_public_values: usize,
}

impl<F: Field> Program<F> {
    pub fn new_empty(step: u32, pc_base: u32, max_num_public_values: usize) -> Self {
        Self {
            instructions_and_debug_infos: vec![],
            step,
            pc_base,
            max_num_public_values,
        }
    }

    pub fn new_without_debug_infos(
        instructions: &[Instruction<F>],
        step: u32,
        pc_base: u32,
        max_num_public_values: usize,
    ) -> Self {
        assert!(
            instructions.is_empty()
                || pc_base + (instructions.len() as u32 - 1) * step <= MAX_ALLOWED_PC
        );
        Self {
            instructions_and_debug_infos: instructions
                .iter()
                .map(|instruction| Some((instruction.clone(), None)))
                .collect(),
            step,
            pc_base,
            max_num_public_values,
        }
    }

    /// We assume that pc_start = pc_base = 0 everywhere except the RISC-V programs, until we need otherwise
    /// We use [DEFAULT_PC_STEP] for consistency with RISC-V
    pub fn from_instructions_and_debug_infos(
        instructions: &[Instruction<F>],
        debug_infos: &[Option<DebugInfo>],
    ) -> Self {
        assert!(instructions.is_empty() || instructions.len() as u32 - 1 <= MAX_ALLOWED_PC);
        Self {
            instructions_and_debug_infos: instructions
                .iter()
                .zip_eq(debug_infos.iter())
                .map(|(instruction, debug_info)| Some((instruction.clone(), debug_info.clone())))
                .collect(),
            step: DEFAULT_PC_STEP,
            pc_base: 0,
            max_num_public_values: DEFAULT_MAX_NUM_PUBLIC_VALUES,
        }
    }

    pub fn strip_debug_infos(self) -> Self {
        Self {
            instructions_and_debug_infos: self
                .instructions_and_debug_infos
                .into_iter()
                .map(|opt| opt.map(|(ins, _)| (ins, None)))
                .collect(),
            ..self
        }
    }

    pub fn from_instructions(instructions: &[Instruction<F>]) -> Self {
        Self::new_without_debug_infos(
            instructions,
            DEFAULT_PC_STEP,
            0,
            DEFAULT_MAX_NUM_PUBLIC_VALUES,
        )
    }

    pub fn len(&self) -> usize {
        self.instructions_and_debug_infos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions_and_debug_infos.is_empty()
    }

    pub fn instructions(&self) -> Vec<Instruction<F>> {
        self.instructions_and_debug_infos
            .iter()
            .flatten()
            .map(|(instruction, _)| instruction.clone())
            .collect()
    }

    pub fn debug_infos(&self) -> Vec<Option<DebugInfo>> {
        self.instructions_and_debug_infos
            .iter()
            .flatten()
            .map(|(_, debug_info)| debug_info.clone())
            .collect()
    }

    pub fn enumerate_by_pc(&self) -> Vec<(u32, Instruction<F>, Option<DebugInfo>)> {
        self.instructions_and_debug_infos
            .iter()
            .enumerate()
            .flat_map(|(index, option)| {
                option.clone().map(|(instruction, debug_info)| {
                    (
                        self.pc_base + (self.step * (index as u32)),
                        instruction,
                        debug_info,
                    )
                })
            })
            .collect()
    }

    // such that pc = pc_base + (step * index)
    pub fn get_instruction_and_debug_info(
        &self,
        index: usize,
    ) -> Option<(Instruction<F>, Option<DebugInfo>)> {
        self.instructions_and_debug_infos
            .get(index)
            .cloned()
            .flatten()
    }

    pub fn push_instruction_and_debug_info(
        &mut self,
        instruction: Instruction<F>,
        debug_info: Option<DebugInfo>,
    ) {
        self.instructions_and_debug_infos
            .push(Some((instruction, debug_info)));
    }

    pub fn push_instruction(&mut self, instruction: Instruction<F>) {
        self.push_instruction_and_debug_info(instruction, None);
    }

    pub fn append(&mut self, other: Program<F>) {
        self.instructions_and_debug_infos
            .extend(other.instructions_and_debug_infos);
    }
}
impl<F: Field> Display for Program<F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for instruction in self.instructions().iter() {
            let Instruction {
                opcode,
                a,
                b,
                c,
                d,
                e,
                f,
                g,
            } = instruction;
            write!(
                formatter,
                "{:?} {} {} {} {} {} {} {}",
                opcode, a, b, c, d, e, f, g,
            )?;
        }
        Ok(())
    }
}

pub fn display_program_with_pc<F: Field>(program: &Program<F>) {
    for (pc, instruction) in program.instructions().iter().enumerate() {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
        } = instruction;
        println!(
            "{} | {:?} {} {} {} {} {} {} {}",
            pc, opcode, a, b, c, d, e, f, g
        );
    }
}
