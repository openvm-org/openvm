use std::{
    borrow::{Borrow, BorrowMut},
    fmt::Debug,
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::LoadStoreExecutor;
use crate::adapters::{rv64_address_add_imm, rv64_bytes_to_u32, sign_extend_imm16};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct LoadStorePreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
    e: u8,
}

impl<A, const NUM_CELLS: usize> LoadStoreExecutor<A, NUM_CELLS> {
    /// Return (local_opcode, enabled)
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut LoadStorePreCompute,
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
        let enabled = !f.is_zero();

        let local_opcode = Rv64LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv64LoadStoreOpcode::CLASS_OFFSET),
        );
        let e_u32 = e.as_canonical_u32();
        let valid_address_space = match local_opcode {
            LOADD | LOADWU | LOADHU | LOADBU => e_u32 == RV64_MEMORY_AS,
            STORED | STOREW | STOREH | STOREB => {
                e_u32 == RV64_MEMORY_AS || e_u32 == PUBLIC_VALUES_AS
            }
            _ => false,
        };
        if d.as_canonical_u32() != RV64_REGISTER_AS || !valid_address_space {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        match local_opcode {
            LOADD | LOADWU | LOADHU | LOADBU => {}
            STORED | STOREW | STOREH | STOREB => {
                if !enabled {
                    return Err(StaticProgramError::InvalidInstruction(pc));
                }
            }
            _ => unreachable!("LoadStoreExecutor should not handle sign-extension load opcodes"),
        }

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = sign_extend_imm16(imm, imm_sign);

        *data = LoadStorePreCompute {
            imm_extended,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            e: e_u32 as u8,
        };
        Ok((local_opcode, enabled))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident, $enabled:ident) => {
        match ($local_opcode, $enabled) {
            (LOADD, true) => Ok($execute_impl::<_, _, U8, LoadDOp, true>),
            (LOADD, false) => Ok($execute_impl::<_, _, U8, LoadDOp, false>),
            (LOADWU, true) => Ok($execute_impl::<_, _, U8, LoadWUOp, true>),
            (LOADWU, false) => Ok($execute_impl::<_, _, U8, LoadWUOp, false>),
            (LOADHU, true) => Ok($execute_impl::<_, _, U8, LoadHUOp, true>),
            (LOADHU, false) => Ok($execute_impl::<_, _, U8, LoadHUOp, false>),
            (LOADBU, true) => Ok($execute_impl::<_, _, U8, LoadBUOp, true>),
            (LOADBU, false) => Ok($execute_impl::<_, _, U8, LoadBUOp, false>),
            (STORED, true) => Ok($execute_impl::<_, _, U8, StoreDOp, true>),
            (STORED, false) => Ok($execute_impl::<_, _, U8, StoreDOp, false>),
            (STOREW, true) => Ok($execute_impl::<_, _, U8, StoreWOp, true>),
            (STOREW, false) => Ok($execute_impl::<_, _, U8, StoreWOp, false>),
            (STOREH, true) => Ok($execute_impl::<_, _, U8, StoreHOp, true>),
            (STOREH, false) => Ok($execute_impl::<_, _, U8, StoreHOp, false>),
            (STOREB, true) => Ok($execute_impl::<_, _, U8, StoreBOp, true>),
            (STOREB, false) => Ok($execute_impl::<_, _, U8, StoreBOp, false>),
            (_, _) => unreachable!(),
        }
    };
}

impl<F, A, const NUM_CELLS: usize> InterpreterExecutor<F> for LoadStoreExecutor<A, NUM_CELLS>
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
        let (local_opcode, enabled) = self.pre_compute_impl(pc, inst, pre_compute)?;
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
        let pre_compute: &mut LoadStorePreCompute = data.borrow_mut();
        let (local_opcode, enabled) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode, enabled)
    }
}

impl<F, A, const NUM_CELLS: usize> InterpreterMeteredExecutor<F> for LoadStoreExecutor<A, NUM_CELLS>
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
        let (local_opcode, enabled) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
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
        let pre_compute: &mut E2PreCompute<LoadStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (local_opcode, enabled) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode, enabled)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    T: Copy + Debug + Default + 'static,
    OP: LoadStoreOp<T>,
    const ENABLED: bool,
>(
    pre_compute: &LoadStorePreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();
    let rs1_bytes: [u8; RV64_REGISTER_NUM_LIMBS] =
        exec_state.vm_read_bytes(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs1_val = rv64_bytes_to_u32(rs1_bytes);
    let addr = rv64_address_add_imm(rs1_val, pre_compute.imm_extended);
    debug_assert!((addr as usize) < RV64_MEMORY_BYTES);
    let ptr_val = addr as u32;

    let shift_amount = ptr_val % RV64_REGISTER_NUM_LIMBS as u32;
    let ptr_val = ptr_val - shift_amount; // aligned ptr

    let read_data: [u8; RV64_REGISTER_NUM_LIMBS] = if OP::IS_LOAD {
        exec_state.vm_read_bytes(pre_compute.e as u32, ptr_val)
    } else {
        exec_state.vm_read_bytes(RV64_REGISTER_AS, pre_compute.a as u32)
    };

    let mut write_data: [T; RV64_REGISTER_NUM_LIMBS] = if OP::HOST_READ {
        exec_state.host_read(pre_compute.e as u32, ptr_val)
    } else {
        [T::default(); RV64_REGISTER_NUM_LIMBS]
    };

    if !OP::compute_write_data(&mut write_data, read_data, shift_amount as usize) {
        let err = ExecutionError::Fail {
            pc,
            msg: "Invalid LoadStoreOp",
        };
        return Err(err);
    }

    if ENABLED {
        if OP::IS_LOAD {
            exec_state.vm_write(RV64_REGISTER_AS, pre_compute.a as u32, &write_data);
        } else {
            exec_state.vm_write(pre_compute.e as u32, ptr_val, &write_data);
        }
    }

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    T: Copy + Debug + Default + 'static,
    OP: LoadStoreOp<T>,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &LoadStorePreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<LoadStorePreCompute>()).borrow();
    execute_e12_impl::<F, CTX, T, OP, ENABLED>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    T: Copy + Debug + Default + 'static,
    OP: LoadStoreOp<T>,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<LoadStorePreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<LoadStorePreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, T, OP, ENABLED>(&pre_compute.data, exec_state)
}

trait LoadStoreOp<T> {
    const IS_LOAD: bool;
    const HOST_READ: bool;

    /// Return if the operation is valid.
    fn compute_write_data(
        write_data: &mut [T; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool;
}

/// Wrapper type for u8 so we can implement `LoadStoreOp<F>` for `F: PrimeField32`.
/// For memory read/write, this type behaves as same as `u8`.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, Default)]
struct U8(u8);
struct LoadDOp;
struct LoadWUOp;
struct LoadHUOp;
struct LoadBUOp;
struct StoreDOp;
struct StoreWOp;
struct StoreHOp;
struct StoreBOp;

impl LoadStoreOp<U8> for LoadDOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; RV64_REGISTER_NUM_LIMBS],
        _shift_amount: usize,
    ) -> bool {
        *write_data = read_data.map(U8);
        true
    }
}

impl LoadStoreOp<U8> for LoadWUOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 4 {
            return false;
        }
        write_data[0] = U8(read_data[shift_amount]);
        write_data[1] = U8(read_data[shift_amount + 1]);
        write_data[2] = U8(read_data[shift_amount + 2]);
        write_data[3] = U8(read_data[shift_amount + 3]);
        true
    }
}

impl LoadStoreOp<U8> for LoadHUOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 2 && shift_amount != 4 && shift_amount != 6 {
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
        write_data: &mut [U8; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        write_data[0] = U8(read_data[shift_amount]);
        true
    }
}

impl LoadStoreOp<U8> for StoreDOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = false;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; RV64_REGISTER_NUM_LIMBS],
        _shift_amount: usize,
    ) -> bool {
        *write_data = read_data.map(U8);
        true
    }
}

impl LoadStoreOp<U8> for StoreWOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 4 {
            return false;
        }
        write_data[shift_amount] = U8(read_data[0]);
        write_data[shift_amount + 1] = U8(read_data[1]);
        write_data[shift_amount + 2] = U8(read_data[2]);
        write_data[shift_amount + 3] = U8(read_data[3]);
        true
    }
}

impl LoadStoreOp<U8> for StoreHOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 2 && shift_amount != 4 && shift_amount != 6 {
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
        write_data: &mut [U8; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        write_data[shift_amount] = U8(read_data[0]);
        true
    }
}

#[cfg(test)]
mod tests {
    use openvm_circuit::arch::StaticProgramError;
    use openvm_instructions::{
        instruction::Instruction, riscv::RV64_REGISTER_AS, LocalOpcode, DEFERRAL_AS,
    };
    use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{LOADWU, STORED};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::{LoadStorePreCompute, Rv64LoadStoreOpcode};
    use crate::{adapters::Rv64LoadStoreAdapterExecutor, Rv64LoadStoreExecutor};

    #[test]
    fn precompute_enforces_address_space_domain() {
        const PC: u32 = 0x100;

        let executor = Rv64LoadStoreExecutor::new(
            Rv64LoadStoreAdapterExecutor::new(29),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        for opcode in [LOADWU, STORED] {
            let inst = Instruction::<BabyBear>::from_usize(
                opcode.global_opcode(),
                [
                    8,
                    16,
                    0,
                    RV64_REGISTER_AS as usize,
                    DEFERRAL_AS as usize,
                    1,
                    0,
                ],
            );
            let mut data = LoadStorePreCompute {
                imm_extended: 0,
                a: 0,
                b: 0,
                e: 0,
            };

            let err = executor
                .pre_compute_impl(PC, &inst, &mut data)
                .expect_err("load/store address-space domain should be enforced");

            assert!(matches!(err, StaticProgramError::InvalidInstruction(PC)));
        }
    }
}
