#[cfg(feature = "aot")]
use openvm_circuit::arch::{AotError, AotExecutor, AotMeteredExecutor, SystemConfig};
use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_AS};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    common::{update_adapter_heights_asm, update_height_change_asm, xmm_to_gpr, REG_A_W, REG_B_W},
    BranchLessThanExecutor,
};

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> AotExecutor<F>
    for BranchLessThanExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> Result<String, AotError> {
        use openvm_instructions::{riscv::RV32_REGISTER_AS, LocalOpcode};
        use openvm_rv32im_transpiler::BranchLessThanOpcode;

        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        let local_opcode = BranchLessThanOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let c = c.as_canonical_u32();
        let imm = if F::ORDER_U32 - c < c {
            -((F::ORDER_U32 - c) as isize)
        } else {
            c as isize
        };
        let next_pc = (pc as isize + imm) as u32;
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            use openvm_circuit::arch::AotError;

            return Err(AotError::InvalidInstruction);
        }
        let a = a.as_canonical_u32() as u8;
        let b = b.as_canonical_u32() as u8;

        let mut asm_str = String::new();
        let a_reg = a / 4;
        let b_reg = b / 4;

        // Calculate the result. Inputs: eax, ecx. Outputs: edx.
        let (reg_a, delta_str_a) = &xmm_to_gpr(a_reg, REG_A_W, false);
        asm_str += delta_str_a;
        let (reg_b, delta_str_b) = &xmm_to_gpr(b_reg, REG_B_W, false);
        asm_str += delta_str_b;

        asm_str += &format!("   cmp {reg_a}, {reg_b}\n");
        let not_jump_label = format!(".asm_execute_pc_{pc}_not_jump");
        match local_opcode {
            BranchLessThanOpcode::BGE => {
                // less (signed) -> not jump
                asm_str += &format!("   jl {not_jump_label}\n");
            }
            BranchLessThanOpcode::BGEU => {
                // below (unsigned) -> not jump
                asm_str += &format!("   jb {not_jump_label}\n");
            }
            BranchLessThanOpcode::BLT => {
                // greater or equal (signed) -> not jump
                asm_str += &format!("   jge {not_jump_label}\n");
            }
            BranchLessThanOpcode::BLTU => {
                // above or equal (unsigned) -> not jump
                asm_str += &format!("   jae {not_jump_label}\n");
            }
        }
        // Jump branch
        asm_str += &format!("   jmp asm_execute_pc_{next_pc}\n");
        asm_str += &format!("{not_jump_label}:\n");

        Ok(asm_str)
    }

    fn is_aot_supported(&self, _inst: &Instruction<F>) -> bool {
        true
    }
}
impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> AotMeteredExecutor<F>
    for BranchLessThanExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn is_aot_metered_supported(&self, _inst: &Instruction<F>) -> bool {
        true
    }
    fn generate_x86_metered_asm(
        &self,
        inst: &Instruction<F>,
        pc: u32,
        chip_idx: usize,
        config: &SystemConfig,
    ) -> Result<String, AotError> {
        let mut asm_str = String::from("");

        asm_str += &update_height_change_asm(chip_idx, 1)?;
        // read [b:4]_1
        asm_str += &update_adapter_heights_asm(config, RV32_REGISTER_AS)?;
        // read [c:4]_1
        asm_str += &update_adapter_heights_asm(config, RV32_REGISTER_AS)?;

        asm_str += &self.generate_x86_asm(inst, pc)?;
        Ok(asm_str)
    }
}
