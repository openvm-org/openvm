use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

#[cfg(feature = "aot")]
use openvm_circuit::arch::aot::common::REG_A_W;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::DivRemWOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::DivRemWExecutor;
#[cfg(feature = "aot")]
use crate::common::{gpr_to_xmm, xmm_to_gpr};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct DivRemWPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<A> DivRemWExecutor<A> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut DivRemWPreCompute,
    ) -> Result<DivRemWOpcode, StaticProgramError> {
        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        let local_opcode = DivRemWOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        if d.as_canonical_u32() != RV64_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let pre_compute: &mut DivRemWPreCompute = data.borrow_mut();
        *pre_compute = DivRemWPreCompute {
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
            DivRemWOpcode::DIVW => Ok($execute_impl::<_, _, DivwOp>),
            DivRemWOpcode::DIVUW => Ok($execute_impl::<_, _, DivuwOp>),
            DivRemWOpcode::REMW => Ok($execute_impl::<_, _, RemwOp>),
            DivRemWOpcode::REMUW => Ok($execute_impl::<_, _, RemuwOp>),
        }
    };
}

impl<F, A> InterpreterExecutor<F> for DivRemWExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<DivRemWPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut DivRemWPreCompute = data.borrow_mut();
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
        let data: &mut DivRemWPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

#[cfg(feature = "aot")]
impl<F, A> AotExecutor<F> for DivRemWExecutor<A>
where
    F: PrimeField32,
{
    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> Result<String, AotError> {
        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        let local_opcode = DivRemWOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        if d.as_canonical_u32() != RV64_REGISTER_AS {
            return Err(AotError::InvalidInstruction);
        }

        let mut asm_str = String::new();
        let a_reg = a.as_canonical_u32() / 4;
        let b_reg = b.as_canonical_u32() / 4;
        let c_reg = c.as_canonical_u32() / 4;

        // Calculate the result. Inputs: eax, ecx. Outputs: edx.
        // Note that for div/rem we are tied to eax/edx because of idiv requirements

        let (_, delta_str_b) = &xmm_to_gpr(b_reg as u8, "eax", true);
        asm_str += delta_str_b;
        let (reg_c, delta_str_c) = &xmm_to_gpr(c_reg as u8, REG_A_W, false);
        asm_str += delta_str_c;
        asm_str += "   mov edx, 0\n";

        let label_prefix = format!(
            ".asm_divrem_{}_{}",
            pc,
            match local_opcode {
                DivRemWOpcode::DIVW => "divw",
                DivRemWOpcode::DIVUW => "divuw",
                DivRemWOpcode::REMW => "remw",
                DivRemWOpcode::REMUW => "remuw",
            }
        );
        let done_label = format!("{label_prefix}__done");

        let zero_label = format!("{label_prefix}__divisor_zero");
        let overflow_label = format!("{label_prefix}__overflow");
        let normal_label = format!("{label_prefix}__normal");
        match local_opcode {
            DivRemWOpcode::DIVW => {
                asm_str += &format!("   test {reg_c}, {reg_c}\n");
                asm_str += &format!("   je {zero_label}\n");
                asm_str += "   cmp eax, 0x80000000\n";
                asm_str += &format!("   jne {normal_label}\n");
                asm_str += &format!("   cmp {reg_c}, -1\n");
                asm_str += &format!("   jne {normal_label}\n");
                asm_str += &format!("   jmp {overflow_label}\n");

                asm_str += &format!("{normal_label}:\n");
                // sign-extend EAX into EDX:EAX
                asm_str += "   cdq\n";
                // eax = eax / ecx, edx = eax % ecx
                asm_str += &format!("   idiv {reg_c}\n");
                asm_str += "   mov edx, eax\n";
                asm_str += &format!("   jmp {done_label}\n");

                asm_str += &format!("{zero_label}:\n");
                asm_str += "   mov edx, -1\n";
                asm_str += &format!("   jmp {done_label}\n");

                asm_str += &format!("{overflow_label}:\n");
                asm_str += "   mov edx, eax\n";
            }
            DivRemWOpcode::DIVUW => {
                asm_str += &format!("   test {reg_c}, {reg_c}\n");
                asm_str += &format!("   je {zero_label}\n");
                // eax = eax / ecx, edx = eax % ecx
                asm_str += &format!("   div {reg_c}\n");
                asm_str += "   mov edx, eax\n";
                asm_str += &format!("   jmp {done_label}\n");

                asm_str += &format!("{zero_label}:\n");
                asm_str += "   mov edx, -1\n";
            }
            DivRemWOpcode::REMW => {
                asm_str += &format!("   test {reg_c}, {reg_c}\n");
                asm_str += &format!("   je {zero_label}\n");
                asm_str += "   cmp eax, 0x80000000\n";
                asm_str += &format!("   jne {normal_label}\n");
                asm_str += &format!("   cmp {reg_c}, -1\n");
                asm_str += &format!("   jne {normal_label}\n");
                asm_str += "   mov edx, 0\n";
                asm_str += &format!("   jmp {done_label}\n");

                asm_str += &format!("{normal_label}:\n");
                // sign-extend EAX into EDX:EAX
                asm_str += "   cdq\n";
                // eax = eax / ecx, edx = eax % ecx
                asm_str += &format!("   idiv {reg_c}\n");
                asm_str += &format!("   jmp {done_label}\n");

                asm_str += &format!("{zero_label}:\n");
                asm_str += "   mov edx, eax\n";
            }
            DivRemWOpcode::REMUW => {
                asm_str += &format!("   test {reg_c}, {reg_c}\n");
                asm_str += &format!("   je {zero_label}\n");
                // eax = eax / ecx, edx = eax % ecx
                asm_str += &format!("   div {reg_c}\n");
                asm_str += &format!("   jmp {done_label}\n");

                asm_str += &format!("{zero_label}:\n");
                asm_str += "   mov edx, eax\n";
            }
        }
        asm_str += &format!("{done_label}:\n");
        asm_str += &gpr_to_xmm("edx", a_reg as u8);

        Ok(asm_str)
    }

    fn is_aot_supported(&self, _inst: &Instruction<F>) -> bool {
        true
    }
}

impl<F, A> InterpreterMeteredExecutor<F> for DivRemWExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<DivRemWPreCompute>>()
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
        let data: &mut E2PreCompute<DivRemWPreCompute> = data.borrow_mut();
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
        let data: &mut E2PreCompute<DivRemWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

#[cfg(feature = "aot")]
impl<F, A> AotMeteredExecutor<F> for DivRemWExecutor<A>
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
        use crate::common::{update_adapter_heights_asm, update_height_change_asm};

        let mut asm_str = self.generate_x86_asm(inst, pc)?;

        asm_str += &update_height_change_asm(chip_idx, 1)?;
        // read [b:4]_1
        asm_str += &update_adapter_heights_asm(config, RV64_REGISTER_AS)?;
        // read [c:4]_1
        asm_str += &update_adapter_heights_asm(config, RV64_REGISTER_AS)?;
        // write [a:4]_1
        asm_str += &update_adapter_heights_asm(config, RV64_REGISTER_AS)?;

        Ok(asm_str)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: DivRemWOp>(
    pre_compute: &DivRemWPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1: [u8; RV64_WORD_NUM_LIMBS] =
        exec_state.vm_read(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs2: [u8; RV64_WORD_NUM_LIMBS] =
        exec_state.vm_read(RV64_REGISTER_AS, pre_compute.c as u32);
    let result_word = <OP as DivRemWOp>::compute(rs1, rs2);
    let rd = (u32::from_le_bytes(result_word) as i32 as i64 as u64).to_le_bytes();
    exec_state.vm_write::<u8, RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.a as u32, &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: DivRemWOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &DivRemWPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<DivRemWPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: DivRemWOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<DivRemWPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<DivRemWPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, exec_state);
}

trait DivRemWOp {
    fn compute(rs1: [u8; RV64_WORD_NUM_LIMBS], rs2: [u8; RV64_WORD_NUM_LIMBS]) -> [u8; RV64_WORD_NUM_LIMBS];
}
struct DivwOp;
struct DivuwOp;
struct RemwOp;
struct RemuwOp;

impl DivRemWOp for DivwOp {
    #[inline(always)]
    fn compute(rs1: [u8; RV64_WORD_NUM_LIMBS], rs2: [u8; RV64_WORD_NUM_LIMBS]) -> [u8; RV64_WORD_NUM_LIMBS] {
        let rs1_i32 = i32::from_le_bytes(rs1);
        let rs2_i32 = i32::from_le_bytes(rs2);
        match (rs1_i32, rs2_i32) {
            (_, 0) => [u8::MAX; RV64_WORD_NUM_LIMBS],
            (i32::MIN, -1) => rs1,
            _ => (rs1_i32 / rs2_i32).to_le_bytes(),
        }
    }
}

impl DivRemWOp for DivuwOp {
    #[inline(always)]
    fn compute(rs1: [u8; RV64_WORD_NUM_LIMBS], rs2: [u8; RV64_WORD_NUM_LIMBS]) -> [u8; RV64_WORD_NUM_LIMBS] {
        if rs2 == [0; RV64_WORD_NUM_LIMBS] {
            [u8::MAX; RV64_WORD_NUM_LIMBS]
        } else {
            let rs1 = u32::from_le_bytes(rs1);
            let rs2 = u32::from_le_bytes(rs2);
            (rs1 / rs2).to_le_bytes()
        }
    }
}

impl DivRemWOp for RemwOp {
    #[inline(always)]
    fn compute(rs1: [u8; RV64_WORD_NUM_LIMBS], rs2: [u8; RV64_WORD_NUM_LIMBS]) -> [u8; RV64_WORD_NUM_LIMBS] {
        let rs1_i32 = i32::from_le_bytes(rs1);
        let rs2_i32 = i32::from_le_bytes(rs2);
        match (rs1_i32, rs2_i32) {
            (_, 0) => rs1,
            (i32::MIN, -1) => [0; RV64_WORD_NUM_LIMBS],
            _ => (rs1_i32 % rs2_i32).to_le_bytes(),
        }
    }
}

impl DivRemWOp for RemuwOp {
    #[inline(always)]
    fn compute(rs1: [u8; RV64_WORD_NUM_LIMBS], rs2: [u8; RV64_WORD_NUM_LIMBS]) -> [u8; RV64_WORD_NUM_LIMBS] {
        if rs2 == [0; RV64_WORD_NUM_LIMBS] {
            rs1
        } else {
            let rs1 = u32::from_le_bytes(rs1);
            let rs2 = u32::from_le_bytes(rs2);
            (rs1 % rs2).to_le_bytes()
        }
    }
}
