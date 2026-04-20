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
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
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

impl<A, const LIMB_BITS: usize> LoadSignExtendExecutor<A, { RV64_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    /// Return (local_opcode, enabled)
    fn pre_compute_impl_rv64<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut LoadSignExtendPreCompute,
    ) -> Result<(Rv64LoadStoreOpcode, bool), StaticProgramError> {
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
        if d.as_canonical_u32() != RV64_REGISTER_AS || e_u32 == RV64_IMM_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = Rv64LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv64LoadStoreOpcode::CLASS_OFFSET),
        );
        debug_assert!(
            matches!(local_opcode, LOADB | LOADH | LOADW),
            "LoadSignExtendExecutor should only handle LOADB/LOADH/LOADW opcodes"
        );

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
        Ok((local_opcode, enabled))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident, $enabled:ident) => {
        match ($local_opcode, $enabled) {
            (LOADB, true) => Ok($execute_impl::<_, _, LoadBOp, true>),
            (LOADB, false) => Ok($execute_impl::<_, _, LoadBOp, false>),
            (LOADH, true) => Ok($execute_impl::<_, _, LoadHOp, true>),
            (LOADH, false) => Ok($execute_impl::<_, _, LoadHOp, false>),
            (LOADW, true) => Ok($execute_impl::<_, _, LoadWOp, true>),
            (LOADW, false) => Ok($execute_impl::<_, _, LoadWOp, false>),
            _ => unreachable!(),
        }
    };
}

impl<F, A, const LIMB_BITS: usize> InterpreterExecutor<F>
    for LoadSignExtendExecutor<A, { RV64_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<LoadSignExtendPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut LoadSignExtendPreCompute = data.borrow_mut();
        let (local_opcode, enabled) = self.pre_compute_impl_rv64(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode, enabled)
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
        let (local_opcode, enabled) = self.pre_compute_impl_rv64(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode, enabled)
    }
}

impl<F, A, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for LoadSignExtendExecutor<A, { RV64_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<LoadSignExtendPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<LoadSignExtendPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (local_opcode, enabled) =
            self.pre_compute_impl_rv64(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode, enabled)
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
        let (local_opcode, enabled) =
            self.pre_compute_impl_rv64(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode, enabled)
    }
}

trait LoadSignExtOp {
    fn compute_write_data(
        read_data: &[u8; RV64_REGISTER_NUM_LIMBS],
        shift_amount: u32,
    ) -> Option<[u8; RV64_REGISTER_NUM_LIMBS]>;
}

struct LoadBOp;
struct LoadHOp;
struct LoadWOp;

impl LoadSignExtOp for LoadBOp {
    #[inline(always)]
    fn compute_write_data(
        read_data: &[u8; RV64_REGISTER_NUM_LIMBS],
        shift_amount: u32,
    ) -> Option<[u8; RV64_REGISTER_NUM_LIMBS]> {
        let byte = read_data[shift_amount as usize];
        Some(((byte as i8) as i64).to_le_bytes())
    }
}

impl LoadSignExtOp for LoadHOp {
    #[inline(always)]
    fn compute_write_data(
        read_data: &[u8; RV64_REGISTER_NUM_LIMBS],
        shift_amount: u32,
    ) -> Option<[u8; RV64_REGISTER_NUM_LIMBS]> {
        if shift_amount % 2 != 0 {
            return None;
        }
        let half: [u8; 2] = array::from_fn(|i| read_data[shift_amount as usize + i]);
        Some((i16::from_le_bytes(half) as i64).to_le_bytes())
    }
}

impl LoadSignExtOp for LoadWOp {
    #[inline(always)]
    fn compute_write_data(
        read_data: &[u8; RV64_REGISTER_NUM_LIMBS],
        shift_amount: u32,
    ) -> Option<[u8; RV64_REGISTER_NUM_LIMBS]> {
        if shift_amount != 0 && shift_amount != 4 {
            return None;
        }
        let word: [u8; 4] = array::from_fn(|i| read_data[shift_amount as usize + i]);
        Some((i32::from_le_bytes(word) as i64).to_le_bytes())
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    OP: LoadSignExtOp,
    const ENABLED: bool,
>(
    pre_compute: &LoadSignExtendPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();
    let rs1_bytes: [u8; RV64_REGISTER_NUM_LIMBS] =
        exec_state.vm_read(RV64_REGISTER_AS, pre_compute.b as u32);
    debug_assert!(
        rs1_bytes[4..].iter().all(|&x| x == 0),
        "load_sign_extend pointers are expected to live in the low 32 bits of rs1"
    );
    let rs1_val = u32::from_le_bytes(rs1_bytes[..4].try_into().unwrap());
    let ptr_val = rs1_val.wrapping_add(pre_compute.imm_extended);
    // sign_extend([r64{c,g}(b):N]_e)
    debug_assert!(ptr_val < (1 << POINTER_MAX_BITS));

    let shift_amount = ptr_val % RV64_REGISTER_NUM_LIMBS as u32;
    let ptr_val = ptr_val - shift_amount; // aligned ptr

    let read_data: [u8; RV64_REGISTER_NUM_LIMBS] =
        exec_state.vm_read(pre_compute.e as u32, ptr_val);

    let write_data =
        OP::compute_write_data(&read_data, shift_amount).ok_or(ExecutionError::Fail {
            pc,
            msg: "Invalid LoadSignExtendOp",
        })?;

    if ENABLED {
        exec_state.vm_write(RV64_REGISTER_AS, pre_compute.a as u32, &write_data);
    }

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    OP: LoadSignExtOp,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &LoadSignExtendPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<LoadSignExtendPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, OP, ENABLED>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    OP: LoadSignExtOp,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<LoadSignExtendPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<LoadSignExtendPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP, ENABLED>(&pre_compute.data, exec_state)
}
