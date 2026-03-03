use eyre::Result;
use openvm_deferral_guest::{COMMIT_NUM_BYTES, DEFERRAL_FUNCT3, MAX_DEF_CIRCUITS, OPCODE};
use openvm_instructions::{
    exe::SparseMemoryImage,
    instruction::Instruction,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode, DEFERRAL_AS,
};
use openvm_instructions_derive::LocalOpcode;
use openvm_transpiler::{TranspilerExtension, TranspilerOutput};
use p3_field::PrimeField32;
use rrs_lib::instruction_formats::IType;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x800]
#[repr(usize)]
pub enum DeferralOpcode {
    CALL,
    OUTPUT,
}

#[derive(Default)]
pub struct DeferralTranspilerExtension {
    def_vk_commits: Vec<[u8; COMMIT_NUM_BYTES]>,
}

impl DeferralTranspilerExtension {
    pub fn new(def_vk_commits: Vec<[u8; COMMIT_NUM_BYTES]>) -> Self {
        assert!(def_vk_commits.len() <= MAX_DEF_CIRCUITS as usize);
        Self { def_vk_commits }
    }
}

impl<F: PrimeField32> TranspilerExtension<F> for DeferralTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }

        let instruction_u32 = instruction_stream[0];
        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        if opcode != OPCODE {
            return None;
        }
        if funct3 != DEFERRAL_FUNCT3 {
            return None;
        }

        // Deferral immediates are encoded as [def_idx(10 bits) | imm_code(2 bits)],
        // where imm_code determines which DeferralOpcode is being called.
        let imm12 = ((instruction_u32 >> 20) & 0xfff) as usize;
        let imm_code = imm12 & 0b11;
        let def_idx = imm12 >> 2;
        if def_idx >= MAX_DEF_CIRCUITS as usize {
            return None;
        }

        let dec_insn = IType::new(instruction_u32);
        let def_opcode = DeferralOpcode::from_repr(imm_code)?;

        let instruction = match def_opcode {
            DeferralOpcode::CALL => Instruction::from_usize(
                DeferralOpcode::CALL.global_opcode(),
                [
                    RV32_REGISTER_NUM_LIMBS * dec_insn.rd,
                    RV32_REGISTER_NUM_LIMBS * dec_insn.rs1,
                    def_idx,
                    RV32_REGISTER_AS as usize,
                    RV32_MEMORY_AS as usize,
                ],
            ),
            DeferralOpcode::OUTPUT => Instruction::from_usize(
                DeferralOpcode::OUTPUT.global_opcode(),
                [
                    RV32_REGISTER_NUM_LIMBS * dec_insn.rd,
                    RV32_REGISTER_NUM_LIMBS * dec_insn.rs1,
                    def_idx,
                    RV32_REGISTER_AS as usize,
                    RV32_MEMORY_AS as usize,
                ],
            ),
        };

        Some(TranspilerOutput::one_to_one(instruction))
    }

    fn modify_initial_memory(&self, init_memory: &mut SparseMemoryImage) -> Result<()> {
        const F_NUM_BYTES: usize = 4;
        const COMMIT_SIZE: usize = COMMIT_NUM_BYTES / F_NUM_BYTES;

        // Each input_acc starts at cell 2 * def_idx * COMMIT_SIZE, and each output_acc
        // immediately follows it. The initial input_acc must be the def_vk_commit, and
        // the initial output_acc must be all 0 (i.e. untouched).
        for (def_idx, commit) in self.def_vk_commits.iter().enumerate() {
            let start_cell = 2 * def_idx * COMMIT_SIZE;
            let start_byte = start_cell * F_NUM_BYTES;

            for (byte_offset, b) in commit.iter().copied().enumerate() {
                init_memory.insert((DEFERRAL_AS, (start_byte + byte_offset) as u32), b);
            }
        }

        Ok(())
    }
}
