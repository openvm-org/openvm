use std::{fmt, fmt::Display};

use itertools::Itertools;
use openvm_stark_backend::p3_field::Field;
use serde::{de::Deserializer, Deserialize, Serialize, Serializer};

use crate::instruction::{DebugInfo, Instruction};

pub const PC_BITS: usize = 30;
/// We use default PC step of 4 whenever possible for consistency with RISC-V, where 4 comes
/// from the fact that each standard RISC-V instruction is 32-bits = 4 bytes.
pub const DEFAULT_PC_STEP: u32 = 4;
pub const MAX_ALLOWED_PC: u32 = (1 << PC_BITS) - 1;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct Program<F> {
    /// A map from program counter to instruction.
    /// Sometimes the instructions are enumerated as 0, 4, 8, etc.
    /// Maybe at some point we will replace this with a struct that would have a `Vec` under the
    /// hood and divide the incoming `pc` by whatever given.
    #[serde(
        serialize_with = "serialize_instructions_and_debug_infos",
        deserialize_with = "deserialize_instructions_and_debug_infos"
    )]
    pub instructions_and_debug_infos: Vec<Option<(Instruction<F>, Option<DebugInfo>)>>,
    pub step: u32,
    pub pc_base: u32,
}

impl<F: Field> Program<F> {
    pub fn new_empty(step: u32, pc_base: u32) -> Self {
        Self {
            instructions_and_debug_infos: vec![],
            step,
            pc_base,
        }
    }

    pub fn new_without_debug_infos(
        instructions: &[Instruction<F>],
        step: u32,
        pc_base: u32,
    ) -> Self {
        Self {
            instructions_and_debug_infos: instructions
                .iter()
                .map(|instruction| Some((instruction.clone(), None)))
                .collect(),
            step,
            pc_base,
        }
    }

    pub fn new_without_debug_infos_with_option(
        instructions: &[Option<Instruction<F>>],
        step: u32,
        pc_base: u32,
    ) -> Self {
        Self {
            instructions_and_debug_infos: instructions
                .iter()
                .map(|instruction| instruction.clone().map(|instruction| (instruction, None)))
                .collect(),
            step,
            pc_base,
        }
    }

    /// We assume that pc_start = pc_base = 0 everywhere except the RISC-V programs, until we need
    /// otherwise We use [DEFAULT_PC_STEP] for consistency with RISC-V
    pub fn from_instructions_and_debug_infos(
        instructions: &[Instruction<F>],
        debug_infos: &[Option<DebugInfo>],
    ) -> Self {
        Self {
            instructions_and_debug_infos: instructions
                .iter()
                .zip_eq(debug_infos.iter())
                .map(|(instruction, debug_info)| Some((instruction.clone(), debug_info.clone())))
                .collect(),
            step: DEFAULT_PC_STEP,
            pc_base: 0,
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
        Self::new_without_debug_infos(instructions, DEFAULT_PC_STEP, 0)
    }

    pub fn len(&self) -> usize {
        self.instructions_and_debug_infos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions_and_debug_infos.is_empty()
    }

    pub fn defined_instructions(&self) -> Vec<Instruction<F>> {
        self.instructions_and_debug_infos
            .iter()
            .flatten()
            .map(|(instruction, _)| instruction.clone())
            .collect()
    }

    // if this is being called a lot, we may want to optimize this later
    pub fn num_defined_instructions(&self) -> usize {
        self.defined_instructions().len()
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
    ) -> Option<&(Instruction<F>, Option<DebugInfo>)> {
        self.instructions_and_debug_infos
            .get(index)
            .and_then(|x| x.as_ref())
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
        for instruction in self.defined_instructions().iter() {
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
            writeln!(
                formatter,
                "{:?} {} {} {} {} {} {} {}",
                opcode, a, b, c, d, e, f, g,
            )?;
        }
        Ok(())
    }
}

pub fn display_program_with_pc<F: Field>(program: &Program<F>) {
    for (pc, instruction) in program.defined_instructions().iter().enumerate() {
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

// `debug_info` is stick with a specific binary. Serializ
fn serialize_instructions_and_debug_infos<F: Serialize, S: Serializer>(
    data: &[Option<(Instruction<F>, Option<DebugInfo>)>],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let ins_data: Vec<_> = data
        .iter()
        .map(|o| o.as_ref().map(|(ins, _)| ins))
        .collect();
    ins_data.serialize(serializer)
}

#[allow(clippy::type_complexity)]
fn deserialize_instructions_and_debug_infos<'de, F: Deserialize<'de>, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<Option<(Instruction<F>, Option<DebugInfo>)>>, D::Error> {
    let inst_data: Vec<Option<Instruction<F>>> = Deserialize::deserialize(deserializer)?;
    Ok(inst_data
        .into_iter()
        .map(|ins_opt| ins_opt.map(|ins| (ins, None)))
        .collect())
}
