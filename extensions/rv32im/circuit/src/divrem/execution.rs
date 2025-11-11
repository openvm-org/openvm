use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::DivRemOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::DivRemExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct DivRemPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<A, const LIMB_BITS: usize> DivRemExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut DivRemPreCompute,
    ) -> Result<DivRemOpcode, StaticProgramError> {
        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        let local_opcode = DivRemOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let pre_compute: &mut DivRemPreCompute = data.borrow_mut();
        *pre_compute = DivRemPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        Ok(local_opcode)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        match $local_opcode {
            DivRemOpcode::DIV => Ok($execute_impl::<_, _, DivOp>),
            DivRemOpcode::DIVU => Ok($execute_impl::<_, _, DivuOp>),
            DivRemOpcode::REM => Ok($execute_impl::<_, _, RemOp>),
            DivRemOpcode::REMU => Ok($execute_impl::<_, _, RemuOp>),
        }
    };
}

impl<F, A, const LIMB_BITS: usize> InterpreterExecutor<F>
    for DivRemExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<DivRemPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut DivRemPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
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
        let data: &mut DivRemPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

#[cfg(feature = "aot")]
impl<F, A, const LIMB_BITS: usize> AotExecutor<F>
    for DivRemExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> Result<String, AotError> {
        use crate::common::{gpr_to_rv32_register, rv32_register_to_gpr};

        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        let local_opcode = DivRemOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(AotError::InvalidInstruction);
        }

        let mut asm_str = String::new();
        let a_reg = a.as_canonical_u32() / 4;
        let b_reg = b.as_canonical_u32() / 4;
        let c_reg = c.as_canonical_u32() / 4;

        // Calculate the result. Inputs: eax, ecx. Outputs: edx.

        // JHEEEZZZ need to rewrite all this..
        asm_str += &rv32_register_to_gpr(b_reg as u8, "eax");
        asm_str += &rv32_register_to_gpr(c_reg as u8, "ecx");
        asm_str += "   mov edx, 0\n";

        let label_prefix = format!(
            ".asm_divrem_{}_{}",
            pc,
            match local_opcode {
                DivRemOpcode::DIV => "div",
                DivRemOpcode::DIVU => "divu",
                DivRemOpcode::REM => "rem",
                DivRemOpcode::REMU => "remu",
            }
        );
        let done_label = format!("{label_prefix}__done");

        let zero_label = format!("{label_prefix}__divisor_zero");
        let overflow_label = format!("{label_prefix}__overflow");
        let normal_label = format!("{label_prefix}__normal");
        match local_opcode {
            DivRemOpcode::DIV => {
                asm_str += "   test ecx, ecx\n";
                asm_str += &format!("   je {zero_label}\n");
                asm_str += "   cmp eax, 0x80000000\n";
                asm_str += &format!("   jne {normal_label}\n");
                asm_str += "   cmp ecx, -1\n";
                asm_str += &format!("   jne {normal_label}\n");
                asm_str += &format!("   jmp {overflow_label}\n");

                asm_str += &format!("{normal_label}:\n");
                // sign-extend EAX into EDX:EAX
                asm_str += "   cdq\n";
                // eax = eax / ecx, edx = eax % ecx
                asm_str += "   idiv ecx\n";
                asm_str += "   mov edx, eax\n";
                asm_str += &format!("   jmp {done_label}\n");

                asm_str += &format!("{zero_label}:\n");
                asm_str += "   mov edx, -1\n";
                asm_str += &format!("   jmp {done_label}\n");

                asm_str += &format!("{overflow_label}:\n");
                asm_str += "   mov edx, eax\n";
            }
            DivRemOpcode::DIVU => {
                asm_str += "   test ecx, ecx\n";
                asm_str += &format!("   je {zero_label}\n");
                // eax = eax / ecx, edx = eax % ecx
                asm_str += "   div ecx\n";
                asm_str += "   mov edx, eax\n";
                asm_str += &format!("   jmp {done_label}\n");

                asm_str += &format!("{zero_label}:\n");
                asm_str += "   mov edx, -1\n";
            }
            DivRemOpcode::REM => {
                asm_str += "   test ecx, ecx\n";
                asm_str += &format!("   je {zero_label}\n");
                asm_str += "   cmp eax, 0x80000000\n";
                asm_str += &format!("   jne {normal_label}\n");
                asm_str += "   cmp ecx, -1\n";
                asm_str += &format!("   jne {normal_label}\n");
                asm_str += "   mov edx, 0\n";
                asm_str += &format!("   jmp {done_label}\n");

                asm_str += &format!("{normal_label}:\n");
                // sign-extend EAX into EDX:EAX
                asm_str += "   cdq\n";
                // eax = eax / ecx, edx = eax % ecx
                asm_str += "   idiv ecx\n";
                asm_str += &format!("   jmp {done_label}\n");

                asm_str += &format!("{zero_label}:\n");
                asm_str += "   mov edx, eax\n";
            }
            DivRemOpcode::REMU => {
                asm_str += "   test ecx, ecx\n";
                asm_str += &format!("   je {zero_label}\n");
                // eax = eax / ecx, edx = eax % ecx
                asm_str += "   div ecx\n";
                asm_str += &format!("   jmp {done_label}\n");

                asm_str += &format!("{zero_label}:\n");
                asm_str += "   mov edx, eax\n";
            }
        }
        asm_str += &format!("{done_label}:\n");
        asm_str += &gpr_to_rv32_register("edx", a_reg as u8);

        Ok(asm_str)
    }

    fn is_aot_supported(&self, _inst: &Instruction<F>) -> bool {
        // true
        false
    }
}

impl<F, A, const LIMB_BITS: usize> MeteredExecutor<F>
    for DivRemExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<DivRemPreCompute>>()
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
        let data: &mut E2PreCompute<DivRemPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
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
        let data: &mut E2PreCompute<DivRemPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: DivRemOp>(
    pre_compute: &DivRemPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c as u32);
    let result = <OP as DivRemOp>::compute(rs1, rs2);
    exec_state.vm_write::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32, &result);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: DivRemOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &DivRemPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<DivRemPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: DivRemOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<DivRemPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<DivRemPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, exec_state);
}

trait DivRemOp {
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4];
}
struct DivOp;
struct DivuOp;
struct RemOp;
struct RemuOp;

impl DivRemOp for DivOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1_i32 = i32::from_le_bytes(rs1);
        let rs2_i32 = i32::from_le_bytes(rs2);
        match (rs1_i32, rs2_i32) {
            (_, 0) => [u8::MAX; 4],
            (i32::MIN, -1) => rs1,
            _ => (rs1_i32 / rs2_i32).to_le_bytes(),
        }
    }
}

impl DivRemOp for DivuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        if rs2 == [0; 4] {
            [u8::MAX; 4]
        } else {
            let rs1 = u32::from_le_bytes(rs1);
            let rs2 = u32::from_le_bytes(rs2);
            (rs1 / rs2).to_le_bytes()
        }
    }
}

impl DivRemOp for RemOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1_i32 = i32::from_le_bytes(rs1);
        let rs2_i32 = i32::from_le_bytes(rs2);
        match (rs1_i32, rs2_i32) {
            (_, 0) => rs1,
            (i32::MIN, -1) => [0; 4],
            _ => (rs1_i32 % rs2_i32).to_le_bytes(),
        }
    }
}

impl DivRemOp for RemuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        if rs2 == [0; 4] {
            rs1
        } else {
            let rs1 = u32::from_le_bytes(rs1);
            let rs2 = u32::from_le_bytes(rs2);
            (rs1 % rs2).to_le_bytes()
        }
    }
}
