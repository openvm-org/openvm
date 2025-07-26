use std::{
    fmt::{self, Display},
    ops::Deref,
    sync::Arc,
};

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
    pub pc_base: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ProgramDebugInfo {
    inner: Arc<Vec<Option<DebugInfo>>>,
    pc_base: u32,
}

impl<F: Field> Program<F> {
    pub fn new_empty(pc_base: u32) -> Self {
        Self {
            instructions_and_debug_infos: vec![],
            pc_base,
        }
    }

    pub fn new_without_debug_infos(instructions: &[Instruction<F>], pc_base: u32) -> Self {
        Self {
            instructions_and_debug_infos: instructions
                .iter()
                .map(|instruction| Some((instruction.clone(), None)))
                .collect(),
            pc_base,
        }
    }

    pub fn new_without_debug_infos_with_option(
        instructions: &[Option<Instruction<F>>],
        pc_base: u32,
    ) -> Self {
        Self {
            instructions_and_debug_infos: instructions
                .iter()
                .map(|instruction| instruction.clone().map(|instruction| (instruction, None)))
                .collect(),
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
        Self::new_without_debug_infos(instructions, 0)
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

    pub fn enumerate_by_pc(&self) -> Vec<(u32, Instruction<F>, Option<DebugInfo>)> {
        self.instructions_and_debug_infos
            .iter()
            .enumerate()
            .flat_map(|(index, option)| {
                option.clone().map(|(instruction, debug_info)| {
                    (
                        self.pc_base + (DEFAULT_PC_STEP * (index as u32)),
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

impl<F> Program<F> {
    pub fn debug_infos(&self) -> ProgramDebugInfo {
        let debug_infos = self
            .instructions_and_debug_infos
            .iter()
            .map(|opt| opt.as_ref().and_then(|(_, debug_info)| debug_info.clone()))
            .collect();
        ProgramDebugInfo {
            inner: Arc::new(debug_infos),
            pc_base: self.pc_base,
        }
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

impl ProgramDebugInfo {
    /// ## Panics
    /// If `pc` is out of bounds.
    pub fn get(&self, pc: u32) -> &Option<DebugInfo> {
        let pc_base = self.pc_base;
        let pc_idx = ((pc - pc_base) / DEFAULT_PC_STEP) as usize;
        &self.inner[pc_idx]
    }
}

impl Deref for ProgramDebugInfo {
    type Target = [Option<DebugInfo>];

    fn deref(&self) -> &Self::Target {
        &self.inner
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

// `debug_info` is based on the symbol table of the binary. Usually serializing `debug_info` is not
// meaningful because the program is executed by another binary. So here we only serialize
// instructions.
fn serialize_instructions_and_debug_infos<F: Serialize, S: Serializer>(
    data: &[Option<(Instruction<F>, Option<DebugInfo>)>],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut ins_data = Vec::with_capacity(data.len());
    let total_len = data.len() as u32;
    for (i, o) in data.iter().enumerate() {
        if let Some(o) = o {
            ins_data.push((&o.0, i as u32));
        }
    }
    (ins_data, total_len).serialize(serializer)
}

#[allow(clippy::type_complexity)]
fn deserialize_instructions_and_debug_infos<'de, F: Deserialize<'de>, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<Option<(Instruction<F>, Option<DebugInfo>)>>, D::Error> {
    let (inst_data, total_len): (Vec<(Instruction<F>, u32)>, u32) =
        Deserialize::deserialize(deserializer)?;
    let mut ret: Vec<Option<(Instruction<F>, Option<DebugInfo>)>> = Vec::new();
    ret.resize_with(total_len as usize, || None);
    for (inst, i) in inst_data {
        ret[i as usize] = Some((inst, None));
    }
    Ok(ret)
}

#[cfg(test)]
mod tests {
    use itertools::izip;
    use p3_baby_bear::BabyBear;

    use super::*;
    use crate::VmOpcode;

    type F = BabyBear;

    #[test]
    fn test_program_serde() {
        let mut program = Program::<F>::new_empty(0);
        program.instructions_and_debug_infos.push(Some((
            Instruction::from_isize(VmOpcode::from_usize(113), 1, 2, 3, 4, 5),
            None,
        )));
        program.instructions_and_debug_infos.push(None);
        program.instructions_and_debug_infos.push(None);
        program.instructions_and_debug_infos.push(Some((
            Instruction::from_isize(VmOpcode::from_usize(145), 10, 20, 30, 40, 50),
            None,
        )));
        program.instructions_and_debug_infos.push(Some((
            Instruction::from_isize(VmOpcode::from_usize(145), 10, 20, 30, 40, 50),
            None,
        )));
        program.instructions_and_debug_infos.push(None);
        let bytes = bitcode::serialize(&program).unwrap();
        let de_program: Program<F> = bitcode::deserialize(&bytes).unwrap();
        for (expected_ins, ins) in izip!(
            &program.instructions_and_debug_infos,
            &de_program.instructions_and_debug_infos
        ) {
            match (expected_ins, ins) {
                (Some(expected_ins), Some(ins)) => {
                    assert_eq!(expected_ins.0, ins.0);
                }
                (None, None) => {}
                _ => {
                    panic!("Different instructions after serialization");
                }
            }
        }
    }
}
