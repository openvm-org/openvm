use std::{
    borrow::{Borrow, BorrowMut},
    fmt::Debug,
    mem::size_of,
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::GuestMemory, POINTER_MAX_BITS},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode, NATIVE_AS,
};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::LoadStoreExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct LoadStorePreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
    e: u8,
}

impl<A, const NUM_CELLS: usize> LoadStoreExecutor<A, NUM_CELLS> {
    /// Return (local_opcode, enabled, is_native_store)
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut LoadStorePreCompute,
    ) -> Result<(Rv32LoadStoreOpcode, bool, bool), StaticProgramError> {
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
        let enabled = !f.is_zero();

        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 == RV32_IMM_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );
        match local_opcode {
            LOADW | LOADBU | LOADHU => {}
            STOREW | STOREH | STOREB => {
                if !enabled {
                    return Err(StaticProgramError::InvalidInstruction(pc));
                }
            }
            _ => unreachable!("LoadStoreExecutor should not handle LOADB/LOADH opcodes"),
        }

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;
        let is_native_store = e_u32 == NATIVE_AS;

        *data = LoadStorePreCompute {
            imm_extended,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            e: e_u32 as u8,
        };
        Ok((local_opcode, enabled, is_native_store))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident, $enabled:ident, $is_native_store:ident) => {
        match ($local_opcode, $enabled, $is_native_store) {
            (LOADW, true, _) => Ok($execute_impl::<_, _, U8, LoadWOp, true>),
            (LOADW, false, _) => Ok($execute_impl::<_, _, U8, LoadWOp, false>),
            (LOADHU, true, _) => Ok($execute_impl::<_, _, U8, LoadHUOp, true>),
            (LOADHU, false, _) => Ok($execute_impl::<_, _, U8, LoadHUOp, false>),
            (LOADBU, true, _) => Ok($execute_impl::<_, _, U8, LoadBUOp, true>),
            (LOADBU, false, _) => Ok($execute_impl::<_, _, U8, LoadBUOp, false>),
            (STOREW, true, false) => Ok($execute_impl::<_, _, U8, StoreWOp, true>),
            (STOREW, false, false) => Ok($execute_impl::<_, _, U8, StoreWOp, false>),
            (STOREW, true, true) => Ok($execute_impl::<_, _, F, StoreWOp, true>),
            (STOREW, false, true) => Ok($execute_impl::<_, _, F, StoreWOp, false>),
            (STOREH, true, false) => Ok($execute_impl::<_, _, U8, StoreHOp, true>),
            (STOREH, false, false) => Ok($execute_impl::<_, _, U8, StoreHOp, false>),
            (STOREH, true, true) => Ok($execute_impl::<_, _, F, StoreHOp, true>),
            (STOREH, false, true) => Ok($execute_impl::<_, _, F, StoreHOp, false>),
            (STOREB, true, false) => Ok($execute_impl::<_, _, U8, StoreBOp, true>),
            (STOREB, false, false) => Ok($execute_impl::<_, _, U8, StoreBOp, false>),
            (STOREB, true, true) => Ok($execute_impl::<_, _, F, StoreBOp, true>),
            (STOREB, false, true) => Ok($execute_impl::<_, _, F, StoreBOp, false>),
            (_, _, _) => unreachable!(),
        }
    };
}

impl<F, A, const NUM_CELLS: usize> Executor<F> for LoadStoreExecutor<A, NUM_CELLS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<LoadStorePreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut LoadStorePreCompute = data.borrow_mut();
        let (local_opcode, enabled, is_native_store) =
            self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode, enabled, is_native_store)
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
        let pre_compute: &mut LoadStorePreCompute = data.borrow_mut();
        let (local_opcode, enabled, is_native_store) =
            self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode, enabled, is_native_store)
    }
}

impl<F, A, const NUM_CELLS: usize> MeteredExecutor<F> for LoadStoreExecutor<A, NUM_CELLS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<LoadStorePreCompute>>()
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
        let pre_compute: &mut E2PreCompute<LoadStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (local_opcode, enabled, is_native_store) =
            self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode, enabled, is_native_store)
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
        let pre_compute: &mut E2PreCompute<LoadStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (local_opcode, enabled, is_native_store) =
            self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode, enabled, is_native_store)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    T: Copy + Debug + Default,
    OP: LoadStoreOp<T>,
    const ENABLED: bool,
>(
    pre_compute: &LoadStorePreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
        exec_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs1_val = u32::from_le_bytes(rs1_bytes);
    let ptr_val = rs1_val.wrapping_add(pre_compute.imm_extended);
    // sign_extend([r32{c,g}(b):2]_e)`
    debug_assert!(ptr_val < (1 << POINTER_MAX_BITS));
    let shift_amount = ptr_val % 4;
    let ptr_val = ptr_val - shift_amount; // aligned ptr

    let read_data: [u8; RV32_REGISTER_NUM_LIMBS] = if OP::IS_LOAD {
        exec_state.vm_read(pre_compute.e as u32, ptr_val)
    } else {
        exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32)
    };

    // We need to write 4 u32s for STORE.
    let mut write_data: [T; RV32_REGISTER_NUM_LIMBS] = if OP::HOST_READ {
        exec_state.host_read(pre_compute.e as u32, ptr_val)
    } else {
        [T::default(); RV32_REGISTER_NUM_LIMBS]
    };

    if !OP::compute_write_data(&mut write_data, read_data, shift_amount as usize) {
        let err = ExecutionError::Fail {
            pc: *pc,
            msg: "Invalid LoadStoreOp",
        };
        return Err(err);
    }

    if ENABLED {
        if OP::IS_LOAD {
            exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &write_data);
        } else {
            exec_state.vm_write(pre_compute.e as u32, ptr_val, &write_data);
        }
    }

    *pc += DEFAULT_PC_STEP;
    *instret += 1;

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    T: Copy + Debug + Default,
    OP: LoadStoreOp<T>,
    const ENABLED: bool,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &LoadStorePreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, T, OP, ENABLED>(pre_compute, instret, pc, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    T: Copy + Debug + Default,
    OP: LoadStoreOp<T>,
    const ENABLED: bool,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<LoadStorePreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, T, OP, ENABLED>(&pre_compute.data, instret, pc, exec_state)
}

trait LoadStoreOp<T> {
    const IS_LOAD: bool;
    const HOST_READ: bool;

    /// Return if the operation is valid.
    fn compute_write_data(
        write_data: &mut [T; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool;
}
/// Wrapper type for u8 so we can implement `LoadStoreOp<F>` for `F: PrimeField32`.
/// For memory read/write, this type behaves as same as `u8`.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, Default)]
struct U8(u8);
struct LoadWOp;
struct LoadHUOp;
struct LoadBUOp;
struct StoreWOp;
struct StoreHOp;
struct StoreBOp;
impl LoadStoreOp<U8> for LoadWOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        _shift_amount: usize,
    ) -> bool {
        *write_data = read_data.map(U8);
        true
    }
}

impl LoadStoreOp<U8> for LoadHUOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;
    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 2 {
            return false;
        }
        write_data[0] = U8(read_data[shift_amount]);
        write_data[1] = U8(read_data[shift_amount + 1]);
        true
    }
}
impl LoadStoreOp<U8> for LoadBUOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;
    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        write_data[0] = U8(read_data[shift_amount]);
        true
    }
}

impl LoadStoreOp<U8> for StoreWOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = false;
    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        _shift_amount: usize,
    ) -> bool {
        *write_data = read_data.map(U8);
        true
    }
}
impl LoadStoreOp<U8> for StoreHOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 2 {
            return false;
        }
        write_data[shift_amount] = U8(read_data[0]);
        write_data[shift_amount + 1] = U8(read_data[1]);
        true
    }
}
impl LoadStoreOp<U8> for StoreBOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;
    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        write_data[shift_amount] = U8(read_data[0]);
        true
    }
}

impl<F: PrimeField32> LoadStoreOp<F> for StoreWOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = false;
    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [F; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        _shift_amount: usize,
    ) -> bool {
        *write_data = read_data.map(F::from_canonical_u8);
        true
    }
}
impl<F: PrimeField32> LoadStoreOp<F> for StoreHOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [F; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 2 {
            return false;
        }
        write_data[shift_amount] = F::from_canonical_u8(read_data[0]);
        write_data[shift_amount + 1] = F::from_canonical_u8(read_data[1]);
        true
    }
}
impl<F: PrimeField32> LoadStoreOp<F> for StoreBOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;
    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [F; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        write_data[shift_amount] = F::from_canonical_u8(read_data[0]);
        true
    }
}
