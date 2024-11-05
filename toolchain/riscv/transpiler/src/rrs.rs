use std::marker::PhantomData;

use axvm_instructions::{
    instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, utils::isize_to_field, BaseAluOpcode,
    BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode, EccOpcode, LessThanOpcode, MulHOpcode,
    MulOpcode, PhantomInstruction, Rv32AuipcOpcode, Rv32BaseAlu256Opcode, Rv32BranchEqual256Opcode,
    Rv32HintStoreOpcode, Rv32JalLuiOpcode, Rv32JalrOpcode, Rv32KeccakOpcode, Rv32LessThan256Opcode,
    Rv32LoadStoreOpcode, Rv32ModularArithmeticOpcode, Rv32Mul256Opcode, Rv32Shift256Opcode,
    ShiftOpcode, UsizeOpcode,
};
use axvm_platform::constants::{
    Custom0Funct3::{self, *},
    Custom1Funct3::{self, *},
    Int256Funct7, ModArithBaseFunct7, PhantomImm, SwBaseFunct7, CUSTOM_0, CUSTOM_1,
    MODULAR_ARITHMETIC_MAX_KINDS, SHORT_WEIERSTRASS_MAX_KINDS,
};
use p3_field::PrimeField32;
use rrs_lib::{
    instruction_formats::{BType, IType, ITypeShamt, JType, RType, SType, UType},
    process_instruction, InstructionProcessor,
};

use crate::util::*;

/// A transpiler that converts the 32-bit encoded instructions into instructions.
pub(crate) struct InstructionTranspiler<F>(PhantomData<F>);

impl<F: PrimeField32> InstructionProcessor for InstructionTranspiler<F> {
    type InstructionResult = Instruction<F>;

    fn process_add(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(BaseAluOpcode::ADD.with_default_offset(), 1, &dec_insn)
    }

    fn process_addi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(BaseAluOpcode::ADD.with_default_offset(), &dec_insn)
    }

    fn process_sub(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(BaseAluOpcode::SUB.with_default_offset(), 1, &dec_insn)
    }

    fn process_xor(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(BaseAluOpcode::XOR.with_default_offset(), 1, &dec_insn)
    }

    fn process_xori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(BaseAluOpcode::XOR.with_default_offset(), &dec_insn)
    }

    fn process_or(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(BaseAluOpcode::OR.with_default_offset(), 1, &dec_insn)
    }

    fn process_ori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(BaseAluOpcode::OR.with_default_offset(), &dec_insn)
    }

    fn process_and(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(BaseAluOpcode::AND.with_default_offset(), 1, &dec_insn)
    }

    fn process_andi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(BaseAluOpcode::AND.with_default_offset(), &dec_insn)
    }

    fn process_sll(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(ShiftOpcode::SLL.with_default_offset(), 1, &dec_insn)
    }

    fn process_slli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftOpcode::SLL.with_default_offset(), &dec_insn)
    }

    fn process_srl(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(ShiftOpcode::SRL.with_default_offset(), 1, &dec_insn)
    }

    fn process_srli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftOpcode::SRL.with_default_offset(), &dec_insn)
    }

    fn process_sra(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(ShiftOpcode::SRA.with_default_offset(), 1, &dec_insn)
    }

    fn process_srai(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftOpcode::SRA.with_default_offset(), &dec_insn)
    }

    fn process_slt(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(LessThanOpcode::SLT.with_default_offset(), 1, &dec_insn)
    }

    fn process_slti(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(LessThanOpcode::SLT.with_default_offset(), &dec_insn)
    }

    fn process_sltu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(LessThanOpcode::SLTU.with_default_offset(), 1, &dec_insn)
    }

    fn process_sltui(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(LessThanOpcode::SLTU.with_default_offset(), &dec_insn)
    }

    fn process_lb(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(Rv32LoadStoreOpcode::LOADB.with_default_offset(), &dec_insn)
    }

    fn process_lh(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(Rv32LoadStoreOpcode::LOADH.with_default_offset(), &dec_insn)
    }

    fn process_lw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(Rv32LoadStoreOpcode::LOADW.with_default_offset(), &dec_insn)
    }

    fn process_lbu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(Rv32LoadStoreOpcode::LOADBU.with_default_offset(), &dec_insn)
    }

    fn process_lhu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(Rv32LoadStoreOpcode::LOADHU.with_default_offset(), &dec_insn)
    }

    fn process_sb(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(Rv32LoadStoreOpcode::STOREB.with_default_offset(), &dec_insn)
    }

    fn process_sh(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(Rv32LoadStoreOpcode::STOREH.with_default_offset(), &dec_insn)
    }

    fn process_sw(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(Rv32LoadStoreOpcode::STOREW.with_default_offset(), &dec_insn)
    }

    fn process_beq(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchEqualOpcode::BEQ.with_default_offset(), &dec_insn)
    }

    fn process_bne(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchEqualOpcode::BNE.with_default_offset(), &dec_insn)
    }

    fn process_blt(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchLessThanOpcode::BLT.with_default_offset(), &dec_insn)
    }

    fn process_bge(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchLessThanOpcode::BGE.with_default_offset(), &dec_insn)
    }

    fn process_bltu(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchLessThanOpcode::BLTU.with_default_offset(), &dec_insn)
    }

    fn process_bgeu(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchLessThanOpcode::BGEU.with_default_offset(), &dec_insn)
    }

    fn process_jal(&mut self, dec_insn: JType) -> Self::InstructionResult {
        from_j_type(Rv32JalLuiOpcode::JAL.with_default_offset(), &dec_insn)
    }

    fn process_jalr(&mut self, dec_insn: IType) -> Self::InstructionResult {
        Instruction::new(
            Rv32JalrOpcode::JALR.with_default_offset(),
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
            F::from_canonical_u32((dec_insn.imm as u32) & 0xffff),
            F::one(),
            F::zero(),
            F::from_bool(dec_insn.rd != 0),
            F::zero(),
        )
    }

    fn process_lui(&mut self, dec_insn: UType) -> Self::InstructionResult {
        if dec_insn.rd == 0 {
            return nop();
        }
        // we need to set f to 1 because this is handled by the same chip as jal
        let mut result = from_u_type(Rv32JalLuiOpcode::LUI.with_default_offset(), &dec_insn);
        result.f = F::one();
        result
    }

    fn process_auipc(&mut self, dec_insn: UType) -> Self::InstructionResult {
        if dec_insn.rd == 0 {
            return nop();
        }
        Instruction::new(
            Rv32AuipcOpcode::AUIPC.with_default_offset(),
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
            F::zero(),
            F::from_canonical_u32(((dec_insn.imm as u32) & 0xfffff000) >> 8),
            F::one(), // rd is a register
            F::zero(),
            F::zero(),
            F::zero(),
        )
    }

    fn process_mul(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(MulOpcode::MUL.with_default_offset(), 0, &dec_insn)
    }

    fn process_mulh(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(MulHOpcode::MULH.with_default_offset(), 0, &dec_insn)
    }

    fn process_mulhu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(MulHOpcode::MULHU.with_default_offset(), 0, &dec_insn)
    }

    fn process_mulhsu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(MulHOpcode::MULHSU.with_default_offset(), 0, &dec_insn)
    }

    fn process_div(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(DivRemOpcode::DIV.with_default_offset(), 0, &dec_insn)
    }

    fn process_divu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(DivRemOpcode::DIVU.with_default_offset(), 0, &dec_insn)
    }

    fn process_rem(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(DivRemOpcode::REM.with_default_offset(), 0, &dec_insn)
    }

    fn process_remu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(DivRemOpcode::REMU.with_default_offset(), 0, &dec_insn)
    }

    fn process_fence(&mut self, dec_insn: IType) -> Self::InstructionResult {
        eprintln!("Transpiling fence ({:?}) to nop", dec_insn);
        nop()
    }
}

fn process_custom_instruction<F: PrimeField32>(instruction_u32: u32) -> Instruction<F> {
    let opcode = (instruction_u32 & 0x7f) as u8;
    let funct3 = ((instruction_u32 >> 12) & 0b111) as u8; // All our instructions are R-, I- or B-type

    match opcode {
        CUSTOM_0 => match Custom0Funct3::from_repr(funct3) {
            Some(Terminate) => {
                let dec_insn = IType::new(instruction_u32);
                Some(terminate(
                    dec_insn.imm.try_into().expect("exit code must be byte"),
                ))
            }
            Some(HintStoreW) => {
                let dec_insn = IType::new(instruction_u32);
                let imm_u16 = (dec_insn.imm as u32) & 0xffff;
                Some(Instruction::from_isize(
                    Rv32HintStoreOpcode::HINT_STOREW.with_default_offset(),
                    0,
                    (RV32_REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                    imm_u16 as isize,
                    1,
                    2,
                ))
            }
            Some(Reveal) => {
                let dec_insn = IType::new(instruction_u32);
                let imm_u16 = (dec_insn.imm as u32) & 0xffff;
                // REVEAL_RV32 is a pseudo-instruction for STOREW_RV32 a,b,c,1,3
                Some(Instruction::from_isize(
                    Rv32LoadStoreOpcode::STOREW.with_default_offset(),
                    (RV32_REGISTER_NUM_LIMBS * dec_insn.rs1) as isize,
                    (RV32_REGISTER_NUM_LIMBS * dec_insn.rd) as isize,
                    imm_u16 as isize,
                    1,
                    3,
                ))
            }
            Some(Phantom) => {
                process_phantom(instruction_u32)
            },
            Some(Keccak256) => {
                let dec_insn = RType::new(instruction_u32);
                Some(from_r_type(Rv32KeccakOpcode::KECCAK256.with_default_offset(), 2, &dec_insn))
            }
            Some(Int256) => {
                let dec_insn = RType::new(instruction_u32);
                let global_opcode = match Int256Funct7::from_repr(dec_insn.funct7 as u8) {
                    Some(Int256Funct7::Add) => {
                        BaseAluOpcode::ADD as usize + Rv32BaseAlu256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Sub) => {
                        BaseAluOpcode::SUB as usize + Rv32BaseAlu256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Xor) => {
                        BaseAluOpcode::XOR as usize + Rv32BaseAlu256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Or) => {
                        BaseAluOpcode::OR as usize + Rv32BaseAlu256Opcode::default_offset()
                    }
                    Some(Int256Funct7::And) => {
                        BaseAluOpcode::AND as usize + Rv32BaseAlu256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Sll) => {
                        ShiftOpcode::SLL as usize + Rv32Shift256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Srl) => {
                        ShiftOpcode::SRL as usize + Rv32Shift256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Sra) => {
                        ShiftOpcode::SRA as usize + Rv32Shift256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Slt) => {
                        LessThanOpcode::SLT as usize + Rv32LessThan256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Sltu) => {
                        LessThanOpcode::SLTU as usize + Rv32LessThan256Opcode::default_offset()
                    }
                    Some(Int256Funct7::Mul) => {
                        MulOpcode::MUL as usize + Rv32Mul256Opcode::default_offset()
                    }
                    _ => unimplemented!(),
                };
                Some(from_r_type(global_opcode, 2, &dec_insn))
            }
            Some(Beq256) => {
                let dec_insn = BType::new(instruction_u32);
                Some(Instruction::new(
                    BranchEqualOpcode::BEQ as usize + Rv32BranchEqual256Opcode::default_offset(),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                    F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
                    isize_to_field(dec_insn.imm as isize),
                    F::one(),
                    F::two(),
                    F::zero(),
                    F::zero(),
                ))
            }
            _ => unimplemented!(),
        },
        CUSTOM_1 => {
            match Custom1Funct3::from_repr(funct3) {
                Some(ModularArithmetic) => {
                    // mod operations
                    let dec_insn = RType::new(instruction_u32);
                    let base_funct7 = (dec_insn.funct7 as u8) % MODULAR_ARITHMETIC_MAX_KINDS;
                    let global_opcode = match ModArithBaseFunct7::from_repr(base_funct7) {
                        Some(ModArithBaseFunct7::AddMod) => {
                            Rv32ModularArithmeticOpcode::ADD as usize
                                + Rv32ModularArithmeticOpcode::default_offset()
                        }
                        Some(ModArithBaseFunct7::SubMod) => {
                            Rv32ModularArithmeticOpcode::SUB as usize
                                + Rv32ModularArithmeticOpcode::default_offset()
                        }
                        Some(ModArithBaseFunct7::MulMod) => {
                            Rv32ModularArithmeticOpcode::MUL as usize
                                + Rv32ModularArithmeticOpcode::default_offset()
                        }
                        Some(ModArithBaseFunct7::DivMod) => {
                            Rv32ModularArithmeticOpcode::DIV as usize
                                + Rv32ModularArithmeticOpcode::default_offset()
                        }
                        Some(ModArithBaseFunct7::IsEqMod) => {
                            Rv32ModularArithmeticOpcode::IS_EQ as usize
                                + Rv32ModularArithmeticOpcode::default_offset()
                        }
                        _ => unimplemented!(),
                    };
                    let mod_idx_shift = ((dec_insn.funct7 as u8) / MODULAR_ARITHMETIC_MAX_KINDS)
                        * MODULAR_ARITHMETIC_MAX_KINDS;
                    let global_opcode = global_opcode + mod_idx_shift as usize;
                    Some(from_r_type(global_opcode, 2, &dec_insn))
                }
                Some(ShortWeierstrass) => {
                    // short weierstrass ec
                    let dec_insn = RType::new(instruction_u32);
                    let base_funct7 = (dec_insn.funct7 as u8) % SHORT_WEIERSTRASS_MAX_KINDS;
                    let global_opcode = match SwBaseFunct7::from_repr(base_funct7) {
                        Some(SwBaseFunct7::SwAddNe) => {
                            EccOpcode::EC_ADD_NE as usize + EccOpcode::default_offset()
                        }
                        Some(SwBaseFunct7::SwDouble) => {
                            EccOpcode::EC_DOUBLE as usize + EccOpcode::default_offset()
                        }
                        _ => unimplemented!(),
                    };
                    let curve_idx_shift = ((dec_insn.funct7 as u8) / SHORT_WEIERSTRASS_MAX_KINDS)
                        * SHORT_WEIERSTRASS_MAX_KINDS;
                    let global_opcode = global_opcode + curve_idx_shift as usize;
                    Some(from_r_type(global_opcode, 2, &dec_insn))
                }
                _ => unimplemented!(),
            }
        }
        _ => None,
    }
    .unwrap_or_else(|| {
        if opcode == 0b1110011 {
            let dec_insn = IType::new(instruction_u32);
            if dec_insn.funct3 == 0b001 {
                // CSRRW
                if dec_insn.rs1 == 0 && dec_insn.rd == 0 {
                    // This resets the CSR counter to zero. Since we don't have any CSR registers, this is a nop.
                    return nop();
                }
            }
            eprintln!(
                "Transpiling system / CSR instruction: {:b} (opcode = {:07b}, funct3 = {:03b}) to unimp",
                instruction_u32, opcode, funct3
            );
            return unimp();
        }
        panic!(
            "Failed to transpile custom instruction: {:b} (opcode = {:07b}, funct3 = {:03b})",
            instruction_u32, opcode, funct3
        )
    })
}

fn process_phantom<F: PrimeField32>(instruction_u32: u32) -> Option<Instruction<F>> {
    let dec_insn = IType::new(instruction_u32);
    PhantomImm::from_repr(dec_insn.imm as u16).map(|phantom| match phantom {
        PhantomImm::HintInput => {
            Instruction::phantom(PhantomInstruction::HintInputRv32, F::zero(), F::zero(), 0)
        }
        PhantomImm::PrintStr => Instruction::phantom(
            PhantomInstruction::PrintStrRv32,
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
            0,
        ),
    })
}

/// Transpile the [`Instruction`]s from the 32-bit encoded instructions.
///
/// # Panics
///
/// This function will return an error if the [`Instruction`] cannot be processed.
pub(crate) fn transpile<F: PrimeField32>(instructions_u32: &[u32]) -> Vec<Instruction<F>> {
    let mut instructions = Vec::new();
    let mut transpiler = InstructionTranspiler::<F>(PhantomData);
    for instruction_u32 in instructions_u32 {
        assert!(*instruction_u32 != 115, "ecall is not supported");
        let instruction = process_instruction(&mut transpiler, *instruction_u32)
            .unwrap_or_else(|| process_custom_instruction(*instruction_u32));
        instructions.push(instruction);
    }
    instructions
}
