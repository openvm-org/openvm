use openvm_circuit::arch::{
    aot::common::{convert_x86_reg, Width, RISCV_TO_X86_OVERRIDE_MAP},
    AotError, AotExecutor, AotMeteredExecutor,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
    LocalOpcode, NATIVE_AS,
};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[allow(unused_imports)]
use crate::{adapters::imm_to_bytes, common::*, BaseAluExecutor};
use crate::{
    common::{address_space_start_to_gpr, gpr_to_rv32_register},
    LoadStoreExecutor,
};

impl<F, A, const NUM_CELLS: usize> AotExecutor<F> for LoadStoreExecutor<A, NUM_CELLS>
where
    F: PrimeField32,
{
    fn is_aot_supported(&self, inst: &openvm_instructions::instruction::Instruction<F>) -> bool {
        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            inst.opcode
                .local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );
        let e_u32 = inst.e.as_canonical_u32();
        let is_native_store = e_u32 == NATIVE_AS;
        // Writing into native address space is not supported in AOT.
        match local_opcode {
            Rv32LoadStoreOpcode::STOREW
            | Rv32LoadStoreOpcode::STOREH
            | Rv32LoadStoreOpcode::STOREB => !is_native_store,
            _ => true,
        }
    }

    fn generate_x86_asm(&self, inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        use crate::common::rv32_register_to_gpr;

        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            ..
        } = inst;
        let enabled = !f.is_zero();

        let e_u32 = e.as_canonical_u32();

        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 == RV32_IMM_AS {
            return Err(AotError::InvalidInstruction);
        }

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = (imm + imm_sign * 0xffff0000) as i32;
        assert_ne!(
            e_u32, NATIVE_AS,
            "Storing into native address space should be handled by the fallback operation"
        );

        let a = a.as_canonical_u32() as u8;
        let b = b.as_canonical_u32() as u8;
        let a_reg = a / 4;
        let b_reg = b / 4;

        let mut asm_str = String::new();
        let (gpr_reg, delta_str) = xmm_to_gpr(b_reg, REG_B_W, true);
        // gpr_reg = [b:4]_1
        asm_str += &delta_str;
        // REG_B_W = ptr = [b:4]_1 + imm_extended
        asm_str += &format!("   add {gpr_reg}, {imm_extended}\n");

        let gpr_reg_w64 =
            convert_x86_reg(&gpr_reg, Width::W64).ok_or(AotError::InvalidInstruction)?;
        // REG_A = <start of destination address space>
        match e_u32 {
            2 => {
                // REG_AS2_PTR = <start of destination address space>
                asm_str += &address_space_start_to_gpr(e_u32, REG_AS2_PTR);
                // REG_B = REG_B + REG_AS2_PTR = <memory address in host memory>
                asm_str += &format!("   lea {gpr_reg_w64}, [{gpr_reg_w64} + {REG_AS2_PTR}]\n");
            }
            _ => {
                asm_str += &address_space_start_to_gpr(e_u32, REG_A);
                // REG_B = REG_B + REG_A = <memory address in host memory>
                asm_str += &format!("   lea {gpr_reg_w64}, [{gpr_reg_w64} + {REG_A}]\n");
            }
        }

        match local_opcode {
            Rv32LoadStoreOpcode::LOADW => {
                let str_reg_a = if RISCV_TO_X86_OVERRIDE_MAP[a_reg as usize].is_some() {
                    RISCV_TO_X86_OVERRIDE_MAP[a_reg as usize].unwrap()
                } else {
                    REG_B_W
                };

                asm_str += &format!("   mov {str_reg_a}, [{gpr_reg_w64}]\n");
                asm_str += &gpr_to_xmm(str_reg_a, a_reg);
            }
            Rv32LoadStoreOpcode::LOADHU => {
                let str_reg_a = if RISCV_TO_X86_OVERRIDE_MAP[a_reg as usize].is_some() {
                    RISCV_TO_X86_OVERRIDE_MAP[a_reg as usize].unwrap()
                } else {
                    REG_B_W
                };

                asm_str += &format!("   movzx {str_reg_a}, word ptr [{gpr_reg_w64}]\n");
                asm_str += &gpr_to_xmm(str_reg_a, a_reg);
            }
            Rv32LoadStoreOpcode::LOADBU => {
                let str_reg_a = if RISCV_TO_X86_OVERRIDE_MAP[a_reg as usize].is_some() {
                    RISCV_TO_X86_OVERRIDE_MAP[a_reg as usize].unwrap()
                } else {
                    REG_B_W
                };

                asm_str += &format!("   movzx {str_reg_a}, byte ptr [{gpr_reg_w64}]\n");
                asm_str += &gpr_to_xmm(str_reg_a, a_reg);
            }
            Rv32LoadStoreOpcode::STOREW => {
                if !enabled {
                    return Err(AotError::InvalidInstruction);
                }
                let (gpr_reg_source, delta_str) = xmm_to_gpr(a_reg, REG_C_W, false);
                asm_str += &delta_str;
                asm_str += &format!("   mov [{gpr_reg_w64}], {gpr_reg_source}\n");
            }
            Rv32LoadStoreOpcode::STOREH => {
                if !enabled {
                    return Err(AotError::InvalidInstruction);
                }
                let (gpr_reg_source, delta_str) = xmm_to_gpr(a_reg, REG_C_W, false);
                let gpr_reg_source_w16 = convert_x86_reg(&gpr_reg_source, Width::W16)
                    .ok_or(AotError::InvalidInstruction)?;
                asm_str += &delta_str;
                asm_str += &format!("   mov word ptr [{gpr_reg_w64}], {gpr_reg_source_w16}\n");
            }
            Rv32LoadStoreOpcode::STOREB => {
                if !enabled {
                    return Err(AotError::InvalidInstruction);
                }
                let (gpr_reg_source, delta_str) = xmm_to_gpr(a_reg, REG_C_W, false);
                let gpr_reg_source_w8l = convert_x86_reg(&gpr_reg_source, Width::W8L)
                    .ok_or(AotError::InvalidInstruction)?;
                asm_str += &delta_str;
                asm_str += &format!("   mov byte ptr [{gpr_reg_w64}], {gpr_reg_source_w8l}\n");
            }
            _ => unreachable!("LoadStoreExecutor should not handle LOADB/LOADH opcodes"),
        }

        Ok(asm_str)
    }
}
impl<F, A, const NUM_CELLS: usize> AotMeteredExecutor<F> for LoadStoreExecutor<A, NUM_CELLS> where
    F: PrimeField32
{
}
