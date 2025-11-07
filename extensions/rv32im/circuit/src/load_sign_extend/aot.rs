use openvm_circuit::arch::{AotError, AotExecutor};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    common::{address_space_start_to_gpr, gpr_to_rv32_register, rv32_register_to_gpr},
    LoadSignExtendExecutor,
};

impl<F, A, const LIMB_BITS: usize> AotExecutor<F>
    for LoadSignExtendExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn is_aot_supported(&self, inst: &Instruction<F>) -> bool {
        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            inst.opcode
                .local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );
        match local_opcode {
            Rv32LoadStoreOpcode::LOADB | Rv32LoadStoreOpcode::LOADH => true,
            _ => unreachable!(),
        }
    }
    fn generate_x86_asm(&self, inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
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

        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 == RV32_IMM_AS {
            return Err(AotError::InvalidInstruction);
        }

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );
        match local_opcode {
            Rv32LoadStoreOpcode::LOADB | Rv32LoadStoreOpcode::LOADH => {}
            _ => unreachable!("LoadSignExtendExecutor should only handle LOADB/LOADH opcodes"),
        }

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;
        let enabled = !f.is_zero();

        let a = a.as_canonical_u32() as u8;
        let b = b.as_canonical_u32() as u8;
        let a_reg = a / 4;
        let b_reg = b / 4;

        let mut asm_str = String::new();
        // eax = [b:4]_1
        asm_str += &rv32_register_to_gpr(b_reg, "eax");
        // eax = ptr = [b:4]_1 + imm_extended
        asm_str += &format!("   add eax, {imm_extended}\n");
        // rcx = <start of destination address space>
        asm_str += &address_space_start_to_gpr(e_u32, "rcx");
        // rax = rax + rcx = <memory address in host memory>
        asm_str += "   lea rax, [rax + rcx]\n";

        if enabled {
            match local_opcode {
                Rv32LoadStoreOpcode::LOADH => {
                    asm_str += "   movsx eax, word ptr [rax]\n";
                    asm_str += &gpr_to_rv32_register("eax", a_reg);
                }
                Rv32LoadStoreOpcode::LOADB => {
                    asm_str += "   movsx eax, byte ptr [rax]\n";
                    asm_str += &gpr_to_rv32_register("eax", a_reg);
                }
                _ => unreachable!("LoadSignExtendExecutor should only handle LOADB/LOADH opcodes"),
            }
        }

        Ok(asm_str)
    }
}
