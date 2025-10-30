use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
#[cfg(feature = "aot")]
use openvm_instructions::VmOpcode;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
};
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::BranchEqualExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct BranchEqualPreCompute {
    imm: isize,
    a: u8,
    b: u8,
}

impl<A, const NUM_LIMBS: usize> BranchEqualExecutor<A, NUM_LIMBS> {
    /// Return `is_bne`, true if the local opcode is BNE.
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BranchEqualPreCompute,
    ) -> Result<bool, StaticProgramError> {
        let data: &mut BranchEqualPreCompute = data.borrow_mut();
        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        let local_opcode = BranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let c = c.as_canonical_u32();
        let imm = if F::ORDER_U32 - c < c {
            -((F::ORDER_U32 - c) as isize)
        } else {
            c as isize
        };
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = BranchEqualPreCompute {
            imm,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok(local_opcode == BranchEqualOpcode::BNE)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_bne:ident) => {
        if $is_bne {
            Ok($execute_impl::<_, _, true>)
        } else {
            Ok($execute_impl::<_, _, false>)
        }
    };
}

impl<F, A, const NUM_LIMBS: usize> InterpreterExecutor<F> for BranchEqualExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BranchEqualPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut BranchEqualPreCompute = data.borrow_mut();
        let is_bne = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, is_bne)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut BranchEqualPreCompute = data.borrow_mut();
        let is_bne = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, is_bne)
    }

    #[cfg(feature = "aot")]
    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> String {
        use crate::common::rv32_register_to_gpr;

        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        let local_opcode = BranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let c = c.as_canonical_u32();
        let imm = if F::ORDER_U32 - c < c {
            -((F::ORDER_U32 - c) as isize)
        } else {
            c as isize
        };
        let next_pc = (pc as isize + imm) as u32;
        // TODO: this should return an error instead.
        assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
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
            BranchEqualOpcode::BEQ => {
                asm_str += &format!("   jne {}\n", not_jump_label);
                asm_str += &format!("   mov r13, {}\n", next_pc);
                asm_str += &format!("   jmp asm_execute_pc_{}\n", next_pc);
            }
            BranchEqualOpcode::BNE => {
                asm_str += &format!("   je {}\n", not_jump_label);
                asm_str += &format!("   mov r13, {}\n", next_pc);
                asm_str += &format!("   jmp asm_execute_pc_{}\n", next_pc);
            }
        }
        asm_str += &format!("{}:\n", not_jump_label);
        asm_str += &format!("   add r13, {}\n", DEFAULT_PC_STEP);

        asm_str
    }

    #[cfg(feature = "aot")]
    fn supports_aot_for_opcode(&self, _opcode: VmOpcode) -> bool {
        true
    }
}

#[cfg(feature = "aot")]
impl<F, A, const NUM_LIMBS: usize> AotExecutor<F> for BranchEqualExecutor<A, NUM_LIMBS> where
    F: PrimeField32
{
}

impl<F, A, const NUM_LIMBS: usize> MeteredExecutor<F> for BranchEqualExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BranchEqualPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<BranchEqualPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_bne = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_bne)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<BranchEqualPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_bne = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_bne)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: &BranchEqualPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs2 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    if (rs1 == rs2) ^ IS_NE {
        *pc = (*pc as isize + pre_compute.imm) as u32;
    } else {
        *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    }
    *instret += 1;
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &BranchEqualPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, IS_NE>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BranchEqualPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_NE>(&pre_compute.data, instret, pc, exec_state);
}
