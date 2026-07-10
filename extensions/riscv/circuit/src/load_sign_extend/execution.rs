use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

#[cfg(feature = "tco")]
use openvm_circuit::arch::execution::Handler;
use openvm_circuit::{
    arch::{
        create_handler, E2PreCompute, ExecuteFunc, ExecutionCtxTrait, ExecutionError,
        InterpreterExecutor, InterpreterMeteredExecutor, MeteredExecutionCtxTrait,
        StaticProgramError, VmExecState, RV64_MEMORY_BYTES,
    },
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, LOADB, LOADH, LOADW};
use openvm_stark_backend::p3_field::PrimeField32;

use super::common::{load_sign_extend_width_for_opcode, LoadSignExtendExecutor};
use crate::adapters::{rv64_address_add_imm, rv64_bytes_to_u32, sign_extend_imm16};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct LoadSignExtendPreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
    e: u8,
}

impl<A, const LOAD_WIDTH: usize> LoadSignExtendExecutor<A, LOAD_WIDTH> {
    fn pre_compute_impl<F: PrimeField32>(
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
        let enabled = !f.is_zero();

        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV64_REGISTER_AS || e_u32 != RV64_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = Rv64LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv64LoadStoreOpcode::CLASS_OFFSET),
        );
        match local_opcode {
            LOADW | LOADH | LOADB
                if load_sign_extend_width_for_opcode(local_opcode) == LOAD_WIDTH => {}
            _ => return Err(StaticProgramError::InvalidInstruction(pc)),
        }

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        *data = LoadSignExtendPreCompute {
            imm_extended: sign_extend_imm16(imm, imm_sign),
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
            (LOADW, true) => Ok($execute_impl::<_, _, LoadWOp, true>),
            (LOADW, false) => Ok($execute_impl::<_, _, LoadWOp, false>),
            (LOADH, true) => Ok($execute_impl::<_, _, LoadHOp, true>),
            (LOADH, false) => Ok($execute_impl::<_, _, LoadHOp, false>),
            (LOADB, true) => Ok($execute_impl::<_, _, LoadBOp, true>),
            (LOADB, false) => Ok($execute_impl::<_, _, LoadBOp, false>),
            _ => Err(StaticProgramError::InvalidInstruction(0)),
        }
    };
}

impl<F, A, const LOAD_WIDTH: usize> InterpreterExecutor<F> for LoadSignExtendExecutor<A, LOAD_WIDTH>
where
    F: PrimeField32,
{
    #[inline(always)]
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
        let pre_compute: &mut LoadSignExtendPreCompute = data.borrow_mut();
        let (local_opcode, enabled) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode, enabled)
    }
}

impl<F, A, const LOAD_WIDTH: usize> InterpreterMeteredExecutor<F>
    for LoadSignExtendExecutor<A, LOAD_WIDTH>
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
        let pre_compute: &mut E2PreCompute<LoadSignExtendPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (local_opcode, enabled) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode, enabled)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    OP: LoadSignExtendOp,
    const ENABLED: bool,
>(
    pre_compute: &LoadSignExtendPreCompute,
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
    let ptr_val = ptr_val - shift_amount;
    // read_data is the containing block followed by the next block, which is only read
    // when the access spans both.
    let mut read_data = [0u8; 2 * RV64_REGISTER_NUM_LIMBS];
    let block0: [u8; RV64_REGISTER_NUM_LIMBS] =
        exec_state.vm_read_bytes(pre_compute.e as u32, ptr_val);
    read_data[..RV64_REGISTER_NUM_LIMBS].copy_from_slice(&block0);
    if shift_amount as usize + OP::WIDTH > RV64_REGISTER_NUM_LIMBS {
        debug_assert!((ptr_val as usize) + 2 * RV64_REGISTER_NUM_LIMBS <= RV64_MEMORY_BYTES);
        let block1: [u8; RV64_REGISTER_NUM_LIMBS] = exec_state.vm_read_bytes(
            pre_compute.e as u32,
            ptr_val + RV64_REGISTER_NUM_LIMBS as u32,
        );
        read_data[RV64_REGISTER_NUM_LIMBS..].copy_from_slice(&block1);
    }
    let mut write_data = [0u8; RV64_REGISTER_NUM_LIMBS];

    if !OP::compute_write_data(&mut write_data, read_data, shift_amount as usize) {
        return Err(ExecutionError::Fail {
            pc,
            msg: "Invalid signed load",
        });
    }

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
    OP: LoadSignExtendOp,
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
    OP: LoadSignExtendOp,
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

trait LoadSignExtendOp {
    /// Access width in bytes.
    const WIDTH: usize;

    /// Return if the operation is valid.
    fn compute_write_data(
        write_data: &mut [u8; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; 2 * RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool;
}

struct LoadWOp;
struct LoadHOp;
struct LoadBOp;

impl LoadSignExtendOp for LoadWOp {
    const WIDTH: usize = 4;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [u8; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; 2 * RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        let word = i32::from_le_bytes([
            read_data[shift_amount],
            read_data[shift_amount + 1],
            read_data[shift_amount + 2],
            read_data[shift_amount + 3],
        ]);
        *write_data = (word as i64).to_le_bytes();
        true
    }
}

impl LoadSignExtendOp for LoadHOp {
    const WIDTH: usize = 2;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [u8; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; 2 * RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        let half = i16::from_le_bytes([read_data[shift_amount], read_data[shift_amount + 1]]);
        *write_data = (half as i64).to_le_bytes();
        true
    }
}

impl LoadSignExtendOp for LoadBOp {
    const WIDTH: usize = 1;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [u8; RV64_REGISTER_NUM_LIMBS],
        read_data: [u8; 2 * RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        let byte = read_data[shift_amount] as i8;
        *write_data = (byte as i64).to_le_bytes();
        true
    }
}
