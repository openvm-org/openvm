use std::marker::PhantomData;

use openvm_instructions::{
    instruction::Instruction, riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode, PhantomDiscriminant,
    SystemOpcode,
};
use openvm_rv64im_guest::{
    PhantomImm, CSRRW_FUNCT3, CSR_OPCODE, HINT_BUFFER_IMM, HINT_FUNCT3, HINT_STORED_IMM,
    NATIVE_STORED_FUNCT3, NATIVE_STORED_FUNCT7, OPCODE_OP_32, OPCODE_OP_IMM_32, PHANTOM_FUNCT3,
    REVEAL_FUNCT3, RV64M_FUNCT7, RV64_ALU_OPCODE, SYSTEM_OPCODE, TERMINATE_FUNCT3,
};
pub use openvm_rv64im_guest::{MAX_HINT_BUFFER_DWORDS, MAX_HINT_BUFFER_DWORDS_BITS};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{
    util::{
        from_i_type_rv64, from_i_type_shamt_rv64, from_load_rv64, from_r_type_rv64,
        from_s_type_rv64, nop, unimp,
    },
    TranspilerExtension, TranspilerOutput,
};
use rrs::InstructionTranspiler;
use rrs_lib::{
    instruction_formats::{IType, ITypeShamt, RType, SType},
    process_instruction,
};

mod instructions;
pub mod rrs;
pub use instructions::*;

// RISC-V opcode fields for RV64-specific encodings
const OPCODE_LOAD: u8 = 0b0000011;
const OPCODE_STORE: u8 = 0b0100011;
const OPCODE_OP_IMM: u8 = 0b0010011;

// funct3 values for loads/stores
const FUNCT3_LD: u8 = 0b011;
const FUNCT3_LWU: u8 = 0b110;
const FUNCT3_SD: u8 = 0b011;

// funct3 values for shifts
const FUNCT3_SLL: u8 = 0b001;
const FUNCT3_SRL_SRA: u8 = 0b101;

// funct3 values for OP-32 / OP-IMM-32
const FUNCT3_ADDW_SUBW: u8 = 0b000;
const FUNCT3_SLLW: u8 = 0b001;
const FUNCT3_SRLW_SRAW: u8 = 0b101;

// funct7 for sub/sra
const FUNCT7_SUB_SRA: u8 = 0b0100000;

// funct3 values for M-extension OP-32
const FUNCT3_MULW: u8 = 0b000;
const FUNCT3_DIVW: u8 = 0b100;
const FUNCT3_DIVUW: u8 = 0b101;
const FUNCT3_REMW: u8 = 0b110;
const FUNCT3_REMUW: u8 = 0b111;

#[derive(Default)]
pub struct Rv64ITranspilerExtension;

#[derive(Default)]
pub struct Rv64MTranspilerExtension;

#[derive(Default)]
pub struct Rv64IoTranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for Rv64ITranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        let mut transpiler = InstructionTranspiler::<F>(PhantomData);
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];

        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        let instruction = match opcode {
            // --- CSR instructions ---
            CSR_OPCODE => {
                let dec_insn = IType::new(instruction_u32);
                if dec_insn.funct3 as u8 == CSRRW_FUNCT3 && dec_insn.rs1 == 0 && dec_insn.rd == 0 {
                    return Some(TranspilerOutput::one_to_one(nop()));
                }
                eprintln!(
                    "Transpiling system / CSR instruction: {instruction_u32:b} (opcode = {opcode:07b}, funct3 = {funct3:03b}) to unimp"
                );
                return Some(TranspilerOutput::one_to_one(unimp()));
            }

            // --- SYSTEM opcode: TERMINATE and PHANTOM ---
            SYSTEM_OPCODE => match funct3 {
                TERMINATE_FUNCT3 => {
                    let dec_insn = IType::new(instruction_u32);
                    Some(Instruction {
                        opcode: SystemOpcode::TERMINATE.global_opcode(),
                        c: F::from_canonical_u8(
                            dec_insn.imm.try_into().expect("exit code must be byte"),
                        ),
                        ..Default::default()
                    })
                }
                PHANTOM_FUNCT3 => {
                    let dec_insn = IType::new(instruction_u32);
                    PhantomImm::from_repr(dec_insn.imm as u16).map(|phantom| match phantom {
                        PhantomImm::HintInput => Instruction::phantom(
                            PhantomDiscriminant(Rv64Phantom::HintInput as u16),
                            F::ZERO,
                            F::ZERO,
                            0,
                        ),
                        PhantomImm::HintRandom => Instruction::phantom(
                            PhantomDiscriminant(Rv64Phantom::HintRandom as u16),
                            F::from_canonical_usize(RV64_REGISTER_NUM_LIMBS * dec_insn.rd),
                            F::ZERO,
                            0,
                        ),
                        PhantomImm::PrintStr => Instruction::phantom(
                            PhantomDiscriminant(Rv64Phantom::PrintStr as u16),
                            F::from_canonical_usize(RV64_REGISTER_NUM_LIMBS * dec_insn.rd),
                            F::from_canonical_usize(RV64_REGISTER_NUM_LIMBS * dec_insn.rs1),
                            0,
                        ),
                        PhantomImm::HintLoadByKey => Instruction::phantom(
                            PhantomDiscriminant(Rv64Phantom::HintLoadByKey as u16),
                            F::from_canonical_usize(RV64_REGISTER_NUM_LIMBS * dec_insn.rd),
                            F::from_canonical_usize(RV64_REGISTER_NUM_LIMBS * dec_insn.rs1),
                            0,
                        ),
                    })
                }
                _ => None,
            },

            // --- RV64 base ALU (OP) — exclude M-extension (funct7=0x01) ---
            RV64_ALU_OPCODE => {
                let dec_insn = RType::new(instruction_u32);
                if dec_insn.funct7 as u8 == RV64M_FUNCT7 {
                    return None; // handled by Rv64MTranspilerExtension
                }
                process_instruction(&mut transpiler, instruction_u32)
            }

            // --- OP-IMM: intercept 6-bit shifts, delegate rest to rrs_lib ---
            OPCODE_OP_IMM => match funct3 {
                FUNCT3_SLL => {
                    // SLLI with 6-bit shamt
                    let dec_insn = ITypeShamt::new(instruction_u32);
                    let shamt6 = (instruction_u32 >> 20) & 0x3f;
                    Some(from_i_type_shamt_rv64(
                        Rv64ShiftOpcode::SLL.global_opcode().as_usize(),
                        &dec_insn,
                        shamt6,
                    ))
                }
                FUNCT3_SRL_SRA => {
                    // SRLI or SRAI with 6-bit shamt
                    let dec_insn = ITypeShamt::new(instruction_u32);
                    let shamt6 = (instruction_u32 >> 20) & 0x3f;
                    let funct6 = (instruction_u32 >> 26) & 0x3f;
                    let shift_opcode = if funct6 == 0b010000 {
                        Rv64ShiftOpcode::SRA
                    } else {
                        Rv64ShiftOpcode::SRL
                    };
                    Some(from_i_type_shamt_rv64(
                        shift_opcode.global_opcode().as_usize(),
                        &dec_insn,
                        shamt6,
                    ))
                }
                _ => process_instruction(&mut transpiler, instruction_u32),
            },

            // --- LOAD: intercept LD and LWU, delegate rest to rrs_lib ---
            OPCODE_LOAD => match funct3 {
                FUNCT3_LD => {
                    let dec_insn = IType::new(instruction_u32);
                    Some(from_load_rv64(
                        Rv64LoadStoreOpcode::LOADD.global_opcode().as_usize(),
                        &dec_insn,
                    ))
                }
                FUNCT3_LWU => {
                    let dec_insn = IType::new(instruction_u32);
                    Some(from_load_rv64(
                        Rv64LoadStoreOpcode::LOADWU.global_opcode().as_usize(),
                        &dec_insn,
                    ))
                }
                _ => process_instruction(&mut transpiler, instruction_u32),
            },

            // --- STORE: intercept SD, delegate rest to rrs_lib ---
            OPCODE_STORE => match funct3 {
                FUNCT3_SD => {
                    let dec_insn = SType::new(instruction_u32);
                    Some(from_s_type_rv64(
                        Rv64LoadStoreOpcode::STORED.global_opcode().as_usize(),
                        &dec_insn,
                    ))
                }
                _ => process_instruction(&mut transpiler, instruction_u32),
            },

            // --- OP-32: RV64-only word-width ALU operations (exclude M-extension) ---
            OPCODE_OP_32 => {
                let dec_insn = RType::new(instruction_u32);
                let funct7 = dec_insn.funct7 as u8;
                if funct7 == RV64M_FUNCT7 {
                    return None; // handled by Rv64MTranspilerExtension
                }
                match funct3 {
                    FUNCT3_ADDW_SUBW => {
                        let op = if funct7 == FUNCT7_SUB_SRA {
                            Rv64BaseAluWOpcode::SUBW
                        } else {
                            Rv64BaseAluWOpcode::ADDW
                        };
                        Some(from_r_type_rv64(
                            op.global_opcode().as_usize(),
                            1,
                            &dec_insn,
                            false,
                        ))
                    }
                    FUNCT3_SLLW => Some(from_r_type_rv64(
                        Rv64ShiftWOpcode::SLLW.global_opcode().as_usize(),
                        1,
                        &dec_insn,
                        false,
                    )),
                    FUNCT3_SRLW_SRAW => {
                        let op = if funct7 == FUNCT7_SUB_SRA {
                            Rv64ShiftWOpcode::SRAW
                        } else {
                            Rv64ShiftWOpcode::SRLW
                        };
                        Some(from_r_type_rv64(
                            op.global_opcode().as_usize(),
                            1,
                            &dec_insn,
                            false,
                        ))
                    }
                    _ => None,
                }
            }

            // --- OP-IMM-32: ADDIW and shift-W immediates ---
            OPCODE_OP_IMM_32 => match funct3 {
                FUNCT3_ADDW_SUBW => {
                    // ADDIW
                    let dec_insn = IType::new(instruction_u32);
                    Some(from_i_type_rv64(
                        Rv64BaseAluWOpcode::ADDW.global_opcode().as_usize(),
                        &dec_insn,
                    ))
                }
                FUNCT3_SLLW => {
                    // SLLIW — 5-bit shamt (bits [24:20])
                    let dec_insn = ITypeShamt::new(instruction_u32);
                    let shamt5 = (instruction_u32 >> 20) & 0x1f;
                    Some(from_i_type_shamt_rv64(
                        Rv64ShiftWOpcode::SLLW.global_opcode().as_usize(),
                        &dec_insn,
                        shamt5,
                    ))
                }
                FUNCT3_SRLW_SRAW => {
                    // SRLIW or SRAIW — 5-bit shamt
                    let dec_insn = ITypeShamt::new(instruction_u32);
                    let shamt5 = (instruction_u32 >> 20) & 0x1f;
                    let funct7 = ((instruction_u32 >> 25) & 0x7f) as u8;
                    let op = if funct7 == FUNCT7_SUB_SRA {
                        Rv64ShiftWOpcode::SRAW
                    } else {
                        Rv64ShiftWOpcode::SRLW
                    };
                    Some(from_i_type_shamt_rv64(
                        op.global_opcode().as_usize(),
                        &dec_insn,
                        shamt5,
                    ))
                }
                _ => None,
            },

            // --- Everything else: delegate to rrs_lib ---
            _ => process_instruction(&mut transpiler, instruction_u32),
        };

        instruction.map(TranspilerOutput::one_to_one)
    }
}

impl<F: PrimeField32> TranspilerExtension<F> for Rv64MTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];

        let opcode = (instruction_u32 & 0x7f) as u8;
        let dec_insn = RType::new(instruction_u32);
        let funct7 = dec_insn.funct7 as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        match opcode {
            // Standard OP with M-extension funct7
            RV64_ALU_OPCODE if funct7 == RV64M_FUNCT7 => {
                let instruction = process_instruction(
                    &mut InstructionTranspiler::<F>(PhantomData),
                    instruction_u32,
                );
                instruction.map(TranspilerOutput::one_to_one)
            }
            // OP-32 with M-extension funct7 (MULW, DIVW, DIVUW, REMW, REMUW)
            OPCODE_OP_32 if funct7 == RV64M_FUNCT7 => {
                let instruction = match funct3 {
                    FUNCT3_MULW => Some(from_r_type_rv64(
                        Rv64MulWOpcode::MULW.global_opcode().as_usize(),
                        0,
                        &dec_insn,
                        false,
                    )),
                    FUNCT3_DIVW => Some(from_r_type_rv64(
                        Rv64DivRemWOpcode::DIVW.global_opcode().as_usize(),
                        0,
                        &dec_insn,
                        false,
                    )),
                    FUNCT3_DIVUW => Some(from_r_type_rv64(
                        Rv64DivRemWOpcode::DIVUW.global_opcode().as_usize(),
                        0,
                        &dec_insn,
                        false,
                    )),
                    FUNCT3_REMW => Some(from_r_type_rv64(
                        Rv64DivRemWOpcode::REMW.global_opcode().as_usize(),
                        0,
                        &dec_insn,
                        false,
                    )),
                    FUNCT3_REMUW => Some(from_r_type_rv64(
                        Rv64DivRemWOpcode::REMUW.global_opcode().as_usize(),
                        0,
                        &dec_insn,
                        false,
                    )),
                    _ => None,
                };
                instruction.map(TranspilerOutput::one_to_one)
            }
            _ => None,
        }
    }
}

impl<F: PrimeField32> TranspilerExtension<F> for Rv64IoTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];

        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        if opcode != SYSTEM_OPCODE {
            return None;
        }

        let instruction = match funct3 {
            HINT_FUNCT3 => {
                let dec_insn = IType::new(instruction_u32);
                let imm_u16 = (dec_insn.imm as u32) & 0xffff;
                match imm_u16 {
                    HINT_STORED_IMM => Some(Instruction::from_isize(
                        Rv64HintStoreOpcode::HINT_STORED.global_opcode(),
                        0,
                        (RV64_REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                        0,
                        1,
                        2,
                    )),
                    HINT_BUFFER_IMM => Some(Instruction::from_isize(
                        Rv64HintStoreOpcode::HINT_BUFFER.global_opcode(),
                        (RV64_REGISTER_NUM_LIMBS * dec_insn.rs1) as isize,
                        (RV64_REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                        0,
                        1,
                        2,
                    )),
                    _ => None,
                }
            }
            REVEAL_FUNCT3 => {
                let dec_insn = IType::new(instruction_u32);
                let imm_u16 = (dec_insn.imm as u32) & 0xffff;
                // REVEAL is a pseudo-instruction for STORED a,b,c,1,3
                Some(Instruction::large_from_isize(
                    Rv64LoadStoreOpcode::STORED.global_opcode(),
                    (RV64_REGISTER_NUM_LIMBS * dec_insn.rs1) as isize,
                    (RV64_REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                    imm_u16 as isize,
                    1,
                    3,
                    1,
                    (dec_insn.imm < 0) as isize,
                ))
            }
            NATIVE_STORED_FUNCT3 => {
                // NATIVE_STORED is a pseudo-instruction for STORED a,b,0,1,4
                let dec_insn = RType::new(instruction_u32);
                if dec_insn.funct7 != NATIVE_STORED_FUNCT7 {
                    return None;
                }
                Some(Instruction::large_from_isize(
                    Rv64LoadStoreOpcode::STORED.global_opcode(),
                    (RV64_REGISTER_NUM_LIMBS * dec_insn.rs1) as isize,
                    (RV64_REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                    0,
                    1,
                    4,
                    1,
                    0,
                ))
            }
            _ => return None,
        };

        instruction.map(TranspilerOutput::one_to_one)
    }
}
