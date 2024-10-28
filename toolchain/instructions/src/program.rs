use std::{collections::HashMap, fmt, fmt::Display};

use itertools::Itertools;
use p3_field::Field;

use crate::instruction::{DebugInfo, Instruction};

pub const PC_BITS: usize = 30;
/// We use default PC step of 4 whenever possible for consistency with RISC-V, where 4 comes
/// from the fact that each standard RISC-V instruction is 32-bits = 4 bytes.
pub const DEFAULT_PC_STEP: u32 = 4;

const MAX_ALLOWED_PC: u32 = (1 << PC_BITS) - 1;

#[derive(Clone, Debug, Default)]
pub struct Program<F: Field> {
    /// A map from program counter to instruction.
    /// Sometimes the instructions are enumerated as 0, 4, 8, etc.
    /// Maybe at some point we will replace this with a struct that would have a `Vec` under the hood and divide the incoming `pc` by whatever given.
    pub instructions_and_debug_infos: HashMap<u32, (Instruction<F>, Option<DebugInfo>)>,
    pub step: u32,
    pub pc_base: u32,
}

impl<F: Field> Program<F> {
    pub fn from_instructions_and_step(
        instructions: &[Instruction<F>],
        step: u32,
        pc_base: u32,
    ) -> Self {
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
            pc_base,
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
                .zip(debug_infos.iter())
                .enumerate()
                .map(|(index, (instruction, debug_info))| {
                    (
                        (index as u32) * DEFAULT_PC_STEP,
                        ((*instruction).clone(), (*debug_info).clone()),
                    )
                })
                .collect(),
            step: DEFAULT_PC_STEP,
            pc_base: 0,
        }
    }

    pub fn from_instructions(instructions: &[Instruction<F>]) -> Self {
        Self::from_instructions_and_step(instructions, DEFAULT_PC_STEP, 0)
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
                debug,
            } = instruction;
            write!(
                formatter,
                "{:?} {} {} {} {} {} {} {} {}",
                opcode, a, b, c, d, e, f, g, debug,
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
            debug,
        } = instruction;
        println!(
            "{} | {:?} {} {} {} {} {} {} {} {}",
            pc, opcode, a, b, c, d, e, f, g, debug
        );
    }
}
