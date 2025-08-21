use std::{
    array,
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::GuestMemory, POINTER_MAX_BITS},
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::LoadSignExtendExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct LoadSignExtendPreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
    e: u8,
}

impl<A, const LIMB_BITS: usize> LoadSignExtendExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    /// Return (is_loadb, enabled)
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut LoadSignExtendPreCompute,
    ) -> Result<(bool, bool), StaticProgramError> {
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
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );
        match local_opcode {
            LOADB | LOADH => {}
            _ => unreachable!("LoadSignExtendExecutor should only handle LOADB/LOADH opcodes"),
        }

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        *data = LoadSignExtendPreCompute {
            imm_extended,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            e: e_u32 as u8,
        };
        let enabled = !f.is_zero();
        Ok((local_opcode == LOADB, enabled))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_loadb:ident, $enabled:ident) => {
        match ($is_loadb, $enabled) {
            (true, true) => Ok($execute_impl::<_, _, true, true>),
            (true, false) => Ok($execute_impl::<_, _, true, false>),
            (false, true) => Ok($execute_impl::<_, _, false, true>),
            (false, false) => Ok($execute_impl::<_, _, false, false>),
        }
    };
}

impl<F, A, const LIMB_BITS: usize> Executor<F>
    for LoadSignExtendExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<LoadSignExtendPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut LoadSignExtendPreCompute = data.borrow_mut();
        let (is_loadb, enabled) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_impl, is_loadb, enabled)
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
        let pre_compute: &mut LoadSignExtendPreCompute = data.borrow_mut();
        let (is_loadb, enabled) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_tco_handler, is_loadb, enabled)
    }
}

impl<F, A, const LIMB_BITS: usize> MeteredExecutor<F>
    for LoadSignExtendExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<LoadSignExtendPreCompute>>()
    }

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
        let pre_compute: &mut E2PreCompute<LoadSignExtendPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (is_loadb, enabled) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_impl, is_loadb, enabled)
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
        let pre_compute: &mut E2PreCompute<LoadSignExtendPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (is_loadb, enabled) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_tco_handler, is_loadb, enabled)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_LOADB: bool,
    const ENABLED: bool,
>(
    pre_compute: &LoadSignExtendPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
        vm_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs1_val = u32::from_le_bytes(rs1_bytes);
    let ptr_val = rs1_val.wrapping_add(pre_compute.imm_extended);
    // sign_extend([r32{c,g}(b):2]_e)`
    debug_assert!(ptr_val < (1 << POINTER_MAX_BITS));
    let shift_amount = ptr_val % 4;
    let ptr_val = ptr_val - shift_amount; // aligned ptr

    let read_data: [u8; RV32_REGISTER_NUM_LIMBS] = vm_state.vm_read(pre_compute.e as u32, ptr_val);

    let write_data = if IS_LOADB {
        let byte = read_data[shift_amount as usize];
        let sign_extended = (byte as i8) as i32;
        sign_extended.to_le_bytes()
    } else {
        if shift_amount != 0 && shift_amount != 2 {
            vm_state.exit_code = Err(ExecutionError::Fail {
                pc: vm_state.pc,
                msg: "LoadSignExtend invalid shift amount",
            });
            return;
        }
        let half: [u8; 2] = array::from_fn(|i| read_data[shift_amount as usize + i]);
        (i16::from_le_bytes(half) as i32).to_le_bytes()
    };

    if ENABLED {
        vm_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &write_data);
    }

    vm_state.pc += DEFAULT_PC_STEP;
    vm_state.instret += 1;
}

#[create_tco_handler]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_LOADB: bool,
    const ENABLED: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &LoadSignExtendPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, IS_LOADB, ENABLED>(pre_compute, vm_state);
}

#[create_tco_handler]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_LOADB: bool,
    const ENABLED: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<LoadSignExtendPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_LOADB, ENABLED>(&pre_compute.data, vm_state);
}
