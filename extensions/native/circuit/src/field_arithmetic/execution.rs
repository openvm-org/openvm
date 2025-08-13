use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::online::GuestMemory,
    utils::{transmute_field_to_u32, transmute_u32_to_field},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_IMM_AS, LocalOpcode,
};
use openvm_native_compiler::{conversion::AS, FieldArithmeticOpcode};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::FieldArithmeticCoreExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FieldArithmeticPreCompute {
    a: u32,
    b_or_imm: u32,
    c_or_imm: u32,
    e: u32,
    f: u32,
}

impl<A> FieldArithmeticCoreExecutor<A> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut FieldArithmeticPreCompute,
    ) -> Result<(bool, bool, FieldArithmeticOpcode), StaticProgramError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            e,
            f,
            ..
        } = inst;

        let local_opcode = FieldArithmeticOpcode::from_usize(
            opcode.local_opcode_idx(FieldArithmeticOpcode::CLASS_OFFSET),
        );

        let a = a.as_canonical_u32();
        let e = e.as_canonical_u32();
        let f = f.as_canonical_u32();

        let a_is_imm = e == RV32_IMM_AS;
        let b_is_imm = f == RV32_IMM_AS;

        let b_or_imm = if a_is_imm {
            transmute_field_to_u32(&b)
        } else {
            b.as_canonical_u32()
        };
        let c_or_imm = if b_is_imm {
            transmute_field_to_u32(&c)
        } else {
            c.as_canonical_u32()
        };

        *data = FieldArithmeticPreCompute {
            a,
            b_or_imm,
            c_or_imm,
            e,
            f,
        };

        Ok((a_is_imm, b_is_imm, local_opcode))
    }
}

impl<F, A> Executor<F> for FieldArithmeticCoreExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<FieldArithmeticPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut FieldArithmeticPreCompute = data.borrow_mut();

        let (a_is_imm, b_is_imm, local_opcode) = self.pre_compute_impl(pc, inst, pre_compute)?;

        let fn_ptr = match (local_opcode, a_is_imm, b_is_imm) {
            (FieldArithmeticOpcode::ADD, true, true) => {
                execute_e1_impl::<_, _, true, true, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::ADD, true, false) => {
                execute_e1_impl::<_, _, true, false, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::ADD, false, true) => {
                execute_e1_impl::<_, _, false, true, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::ADD, false, false) => {
                execute_e1_impl::<_, _, false, false, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::SUB, true, true) => {
                execute_e1_impl::<_, _, true, true, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::SUB, true, false) => {
                execute_e1_impl::<_, _, true, false, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::SUB, false, true) => {
                execute_e1_impl::<_, _, false, true, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::SUB, false, false) => {
                execute_e1_impl::<_, _, false, false, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::MUL, true, true) => {
                execute_e1_impl::<_, _, true, true, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::MUL, true, false) => {
                execute_e1_impl::<_, _, true, false, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::MUL, false, true) => {
                execute_e1_impl::<_, _, false, true, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::MUL, false, false) => {
                execute_e1_impl::<_, _, false, false, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::DIV, true, true) => {
                execute_e1_impl::<_, _, true, true, { FieldArithmeticOpcode::DIV as u8 }>
            }
            (FieldArithmeticOpcode::DIV, true, false) => {
                execute_e1_impl::<_, _, true, false, { FieldArithmeticOpcode::DIV as u8 }>
            }
            (FieldArithmeticOpcode::DIV, false, true) => {
                execute_e1_impl::<_, _, false, true, { FieldArithmeticOpcode::DIV as u8 }>
            }
            (FieldArithmeticOpcode::DIV, false, false) => {
                execute_e1_impl::<_, _, false, false, { FieldArithmeticOpcode::DIV as u8 }>
            }
        };

        Ok(fn_ptr)
    }
}

impl<F, A> MeteredExecutor<F> for FieldArithmeticCoreExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<FieldArithmeticPreCompute>>()
    }

    #[inline(always)]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<FieldArithmeticPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let (a_is_imm, b_is_imm, local_opcode) =
            self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        let fn_ptr = match (local_opcode, a_is_imm, b_is_imm) {
            (FieldArithmeticOpcode::ADD, true, true) => {
                execute_e2_impl::<_, _, true, true, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::ADD, true, false) => {
                execute_e2_impl::<_, _, true, false, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::ADD, false, true) => {
                execute_e2_impl::<_, _, false, true, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::ADD, false, false) => {
                execute_e2_impl::<_, _, false, false, { FieldArithmeticOpcode::ADD as u8 }>
            }
            (FieldArithmeticOpcode::SUB, true, true) => {
                execute_e2_impl::<_, _, true, true, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::SUB, true, false) => {
                execute_e2_impl::<_, _, true, false, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::SUB, false, true) => {
                execute_e2_impl::<_, _, false, true, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::SUB, false, false) => {
                execute_e2_impl::<_, _, false, false, { FieldArithmeticOpcode::SUB as u8 }>
            }
            (FieldArithmeticOpcode::MUL, true, true) => {
                execute_e2_impl::<_, _, true, true, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::MUL, true, false) => {
                execute_e2_impl::<_, _, true, false, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::MUL, false, true) => {
                execute_e2_impl::<_, _, false, true, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::MUL, false, false) => {
                execute_e2_impl::<_, _, false, false, { FieldArithmeticOpcode::MUL as u8 }>
            }
            (FieldArithmeticOpcode::DIV, true, true) => {
                execute_e2_impl::<_, _, true, true, { FieldArithmeticOpcode::DIV as u8 }>
            }
            (FieldArithmeticOpcode::DIV, true, false) => {
                execute_e2_impl::<_, _, true, false, { FieldArithmeticOpcode::DIV as u8 }>
            }
            (FieldArithmeticOpcode::DIV, false, true) => {
                execute_e2_impl::<_, _, false, true, { FieldArithmeticOpcode::DIV as u8 }>
            }
            (FieldArithmeticOpcode::DIV, false, false) => {
                execute_e2_impl::<_, _, false, false, { FieldArithmeticOpcode::DIV as u8 }>
            }
        };

        Ok(fn_ptr)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const A_IS_IMM: bool,
    const B_IS_IMM: bool,
    const OPCODE: u8,
>(
    pre_compute: &FieldArithmeticPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    // Read values based on the adapter logic
    let b_val = if A_IS_IMM {
        transmute_u32_to_field(&pre_compute.b_or_imm)
    } else {
        vm_state.vm_read::<F, 1>(pre_compute.e, pre_compute.b_or_imm)[0]
    };
    let c_val = if B_IS_IMM {
        transmute_u32_to_field(&pre_compute.c_or_imm)
    } else {
        vm_state.vm_read::<F, 1>(pre_compute.f, pre_compute.c_or_imm)[0]
    };

    let a_val = match OPCODE {
        0 => b_val + c_val, // ADD
        1 => b_val - c_val, // SUB
        2 => b_val * c_val, // MUL
        3 => {
            // DIV
            if c_val.is_zero() {
                vm_state.exit_code = Err(ExecutionError::Fail {
                    pc: vm_state.pc,
                    msg: "DivF divide by zero",
                });
                return;
            }
            b_val * c_val.inverse()
        }
        _ => panic!("Invalid field arithmetic opcode: {OPCODE}"),
    };

    vm_state.vm_write::<F, 1>(AS::Native as u32, pre_compute.a, &[a_val]);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const A_IS_IMM: bool,
    const B_IS_IMM: bool,
    const OPCODE: u8,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FieldArithmeticPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, A_IS_IMM, B_IS_IMM, OPCODE>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const A_IS_IMM: bool,
    const B_IS_IMM: bool,
    const OPCODE: u8,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FieldArithmeticPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, A_IS_IMM, B_IS_IMM, OPCODE>(&pre_compute.data, vm_state);
}
