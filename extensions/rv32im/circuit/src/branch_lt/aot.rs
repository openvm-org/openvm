use openvm_circuit::arch::{AotError, AotExecutor};
#[cfg(feature = "aot")]
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::BranchLessThanExecutor;

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> AotExecutor<F>
    for BranchLessThanExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> Result<String, AotError> {
        use openvm_instructions::{program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode};
        use openvm_rv32im_transpiler::BranchLessThanOpcode;

        use crate::common::rv32_register_to_gpr;

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
        asm_str += &rv32_register_to_gpr(a_reg as u8, "eax");
        asm_str += &rv32_register_to_gpr(b_reg as u8, "ecx");
        // instret += 1
        asm_str += "   add r14, 1\n";
        asm_str += "   cmp eax, ecx\n";
        let not_jump_label = format!(".asm_execute_pc_{}_not_jump", pc);
        match local_opcode {
            BranchLessThanOpcode::BGE => {
                // less (signed) -> not jump
                asm_str += &format!("   jl {}\n", not_jump_label);
            }
            BranchLessThanOpcode::BGEU => {
                // below (unsigned) -> not jump
                asm_str += &format!("   jb {}\n", not_jump_label);
            }
            BranchLessThanOpcode::BLT => {
                // greater or equal (signed) -> not jump
                asm_str += &format!("   jge {}\n", not_jump_label);
            }
            BranchLessThanOpcode::BLTU => {
                // above or equal (unsigned) -> not jump
                asm_str += &format!("   jae {}\n", not_jump_label);
            }
        }
        // Jump branch
        asm_str += &format!("   mov r13, {}\n", next_pc);
        asm_str += &format!("   jmp asm_execute_pc_{}\n", next_pc);

        asm_str += &format!("{}:\n", not_jump_label);
        asm_str += &format!("   add r13, {}\n", DEFAULT_PC_STEP);

        Ok(asm_str)
    }

    fn is_aot_supported(&self, _inst: &Instruction<F>) -> bool {
        true
    }
}
