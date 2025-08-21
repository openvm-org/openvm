use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::online::GuestMemory,
    utils::{transmute_field_to_u32, transmute_u32_to_field},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_IMM_AS, LocalOpcode, NATIVE_AS,
};
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::NativeBranchEqualExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct NativeBranchEqualPreCompute {
    imm: isize,
    a_or_imm: u32,
    b_or_imm: u32,
}

impl<A> NativeBranchEqualExecutor<A> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut NativeBranchEqualPreCompute,
    ) -> Result<(bool, bool, bool), StaticProgramError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let local_opcode = BranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let c = c.as_canonical_u32();
        let imm = if F::ORDER_U32 - c < c {
            -((F::ORDER_U32 - c) as isize)
        } else {
            c as isize
        };
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();

        let a_is_imm = d == RV32_IMM_AS;
        let b_is_imm = e == RV32_IMM_AS;

        let a_or_imm = if a_is_imm {
            transmute_field_to_u32(&a)
        } else {
            a.as_canonical_u32()
        };
        let b_or_imm = if b_is_imm {
            transmute_field_to_u32(&b)
        } else {
            b.as_canonical_u32()
        };

        *data = NativeBranchEqualPreCompute {
            imm,
            a_or_imm,
            b_or_imm,
        };

        let is_bne = local_opcode == BranchEqualOpcode::BNE;

        Ok((a_is_imm, b_is_imm, is_bne))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $a_is_imm:ident, $b_is_imm:ident, $is_bne:ident) => {
        match ($a_is_imm, $b_is_imm, $is_bne) {
            (true, true, true) => Ok($execute_impl::<_, _, true, true, true>),
            (true, true, false) => Ok($execute_impl::<_, _, true, true, false>),
            (true, false, true) => Ok($execute_impl::<_, _, true, false, true>),
            (true, false, false) => Ok($execute_impl::<_, _, true, false, false>),
            (false, true, true) => Ok($execute_impl::<_, _, false, true, true>),
            (false, true, false) => Ok($execute_impl::<_, _, false, true, false>),
            (false, false, true) => Ok($execute_impl::<_, _, false, false, true>),
            (false, false, false) => Ok($execute_impl::<_, _, false, false, false>),
        }
    };
}

impl<F, A> Executor<F> for NativeBranchEqualExecutor<A>
where
    F: PrimeField32,
{
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
        let pre_compute: &mut NativeBranchEqualPreCompute = data.borrow_mut();

        let (a_is_imm, b_is_imm, is_bne) = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(execute_e1_tco_handler, a_is_imm, b_is_imm, is_bne)
    }

    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<NativeBranchEqualPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut NativeBranchEqualPreCompute = data.borrow_mut();

        let (a_is_imm, b_is_imm, is_bne) = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(execute_e1_impl, a_is_imm, b_is_imm, is_bne)
    }
}

impl<F, A> MeteredExecutor<F> for NativeBranchEqualExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<NativeBranchEqualPreCompute>>()
    }

    #[inline(always)]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<NativeBranchEqualPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let (a_is_imm, b_is_imm, is_bne) =
            self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        dispatch!(execute_e2_impl, a_is_imm, b_is_imm, is_bne)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<NativeBranchEqualPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let (a_is_imm, b_is_imm, is_bne) =
            self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        dispatch!(execute_e2_tco_handler, a_is_imm, b_is_imm, is_bne)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const A_IS_IMM: bool,
    const B_IS_IMM: bool,
    const IS_NE: bool,
>(
    pre_compute: &NativeBranchEqualPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = if A_IS_IMM {
        transmute_u32_to_field(&pre_compute.a_or_imm)
    } else {
        vm_state.vm_read::<F, 1>(NATIVE_AS, pre_compute.a_or_imm)[0]
    };
    let rs2 = if B_IS_IMM {
        transmute_u32_to_field(&pre_compute.b_or_imm)
    } else {
        vm_state.vm_read::<F, 1>(NATIVE_AS, pre_compute.b_or_imm)[0]
    };
    if (rs1 == rs2) ^ IS_NE {
        vm_state.pc = (vm_state.pc as isize + pre_compute.imm) as u32;
    } else {
        vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    }
    vm_state.instret += 1;
}

#[create_tco_handler]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const A_IS_IMM: bool,
    const B_IS_IMM: bool,
    const IS_NE: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &NativeBranchEqualPreCompute = pre_compute.borrow();
    execute_e12_impl::<_, _, A_IS_IMM, B_IS_IMM, IS_NE>(pre_compute, vm_state);
}

#[create_tco_handler]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const A_IS_IMM: bool,
    const B_IS_IMM: bool,
    const IS_NE: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<NativeBranchEqualPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<_, _, A_IS_IMM, B_IS_IMM, IS_NE>(&pre_compute.data, vm_state);
}
