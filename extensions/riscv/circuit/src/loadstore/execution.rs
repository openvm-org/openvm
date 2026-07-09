use std::{
    borrow::{Borrow, BorrowMut},
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
            (LOADD, true) => Ok($execute_impl::<_, _, LoadDOp, true>),
            (LOADD, false) => Ok($execute_impl::<_, _, LoadDOp, false>),
            (LOADWU, true) => Ok($execute_impl::<_, _, LoadWUOp, true>),
            (LOADWU, false) => Ok($execute_impl::<_, _, LoadWUOp, false>),
            (LOADHU, true) => Ok($execute_impl::<_, _, LoadHUOp, true>),
            (LOADHU, false) => Ok($execute_impl::<_, _, LoadHUOp, false>),
            (LOADBU, true) => Ok($execute_impl::<_, _, LoadBUOp, true>),
            (LOADBU, false) => Ok($execute_impl::<_, _, LoadBUOp, false>),
            (STORED, true) => Ok($execute_impl::<_, _, StoreDOp, true>),
            (STORED, false) => Ok($execute_impl::<_, _, StoreDOp, false>),
            (STOREW, true) => Ok($execute_impl::<_, _, StoreWOp, true>),
            (STOREW, false) => Ok($execute_impl::<_, _, StoreWOp, false>),
            (STOREH, true) => Ok($execute_impl::<_, _, StoreHOp, true>),
            (STOREH, false) => Ok($execute_impl::<_, _, StoreHOp, false>),
            (STOREB, true) => Ok($execute_impl::<_, _, StoreBOp, true>),
            (STOREB, false) => Ok($execute_impl::<_, _, StoreBOp, false>),
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
    OP: LoadStoreOp,
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

    let shift_amount = (ptr_val as usize) % RV64_REGISTER_NUM_LIMBS;
    let aligned_ptr = ptr_val - shift_amount as u32;
    // Whether the access spans two adjacent 8-byte blocks.
    let crosses = shift_amount + OP::WIDTH > RV64_REGISTER_NUM_LIMBS;
    debug_assert!(
        !crosses || (aligned_ptr as usize) + 2 * RV64_REGISTER_NUM_LIMBS <= RV64_MEMORY_BYTES
    );

    if OP::IS_LOAD {
        let mut src = [0u8; 2 * RV64_REGISTER_NUM_LIMBS];
        let block0: [u8; RV64_REGISTER_NUM_LIMBS] =
            exec_state.vm_read_bytes(pre_compute.e as u32, aligned_ptr);
        src[..RV64_REGISTER_NUM_LIMBS].copy_from_slice(&block0);
        if crosses {
            let block1: [u8; RV64_REGISTER_NUM_LIMBS] = exec_state.vm_read_bytes(
                pre_compute.e as u32,
                aligned_ptr + RV64_REGISTER_NUM_LIMBS as u32,
            );
            src[RV64_REGISTER_NUM_LIMBS..].copy_from_slice(&block1);
        }
        if ENABLED {
            let mut write_data = [0u8; RV64_REGISTER_NUM_LIMBS];
            write_data[..OP::WIDTH].copy_from_slice(&src[shift_amount..shift_amount + OP::WIDTH]);
            exec_state.vm_write(RV64_REGISTER_AS, pre_compute.a as u32, &write_data);
        }
    } else {
        let read_data: [u8; RV64_REGISTER_NUM_LIMBS] =
            exec_state.vm_read_bytes(RV64_REGISTER_AS, pre_compute.a as u32);
        let mut block0: [u8; RV64_REGISTER_NUM_LIMBS] =
            exec_state.host_read(pre_compute.e as u32, aligned_ptr);
        let end0 = (shift_amount + OP::WIDTH).min(RV64_REGISTER_NUM_LIMBS);
        block0[shift_amount..end0].copy_from_slice(&read_data[..end0 - shift_amount]);
        if ENABLED {
            exec_state.vm_write(pre_compute.e as u32, aligned_ptr, &block0);
        }
        if crosses {
            let block1_ptr = aligned_ptr + RV64_REGISTER_NUM_LIMBS as u32;
            let mut block1: [u8; RV64_REGISTER_NUM_LIMBS] =
                exec_state.host_read(pre_compute.e as u32, block1_ptr);
            let spill = shift_amount + OP::WIDTH - RV64_REGISTER_NUM_LIMBS;
            block1[..spill].copy_from_slice(&read_data[end0 - shift_amount..OP::WIDTH]);
            if ENABLED {
                exec_state.vm_write(pre_compute.e as u32, block1_ptr, &block1);
            }
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
    OP: LoadStoreOp,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &LoadStorePreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<LoadStorePreCompute>()).borrow();
    execute_e12_impl::<F, CTX, OP, ENABLED>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    OP: LoadStoreOp,
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
    execute_e12_impl::<F, CTX, OP, ENABLED>(&pre_compute.data, exec_state)
}

/// Loads select `WIDTH` bytes starting at the byte shift (possibly spanning two blocks) and
/// zero-extend into rd. Stores merge the low `WIDTH` source register bytes into the touched
/// block(s). Any byte shift is supported.
trait LoadStoreOp {
    const IS_LOAD: bool;
    const WIDTH: usize;
}

struct LoadDOp;
struct LoadWUOp;
struct LoadHUOp;
struct LoadBUOp;
struct StoreDOp;
struct StoreWOp;
struct StoreHOp;
struct StoreBOp;

impl LoadStoreOp for LoadDOp {
    const IS_LOAD: bool = true;
    const WIDTH: usize = 8;
}
impl LoadStoreOp for LoadWUOp {
    const IS_LOAD: bool = true;
    const WIDTH: usize = 4;
}
impl LoadStoreOp for LoadHUOp {
    const IS_LOAD: bool = true;
    const WIDTH: usize = 2;
}
impl LoadStoreOp for LoadBUOp {
    const IS_LOAD: bool = true;
    const WIDTH: usize = 1;
}
impl LoadStoreOp for StoreDOp {
    const IS_LOAD: bool = false;
    const WIDTH: usize = 8;
}
impl LoadStoreOp for StoreWOp {
    const IS_LOAD: bool = false;
    const WIDTH: usize = 4;
}
impl LoadStoreOp for StoreHOp {
    const IS_LOAD: bool = false;
    const WIDTH: usize = 2;
}
impl LoadStoreOp for StoreBOp {
    const IS_LOAD: bool = false;
    const WIDTH: usize = 1;
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
