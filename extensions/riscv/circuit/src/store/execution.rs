use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

#[cfg(feature = "tco")]
use openvm_circuit::arch::execution::Handler;
#[cfg(not(feature = "tco"))]
use openvm_circuit::arch::ExecuteFunc;
use openvm_circuit::{
    arch::{
        create_handler, E2PreCompute, ExecutionCtxTrait, ExecutionError, InterpreterExecutor,
        InterpreterMeteredExecutor, MeteredExecutionCtxTrait, StaticProgramError, VmExecState,
    },
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_platform::memory::MEM_SIZE;
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, STOREB, STORED, STOREH, STOREW};
use openvm_stark_backend::p3_field::PrimeField32;

use super::common::{store_width_for_opcode, StoreExecutor};
use crate::adapters::{
    rv64_address_add_imm, rv64_bytes_to_u32, sign_extend_imm16, BYTE_ACCESS_WIDTH,
    DOUBLEWORD_ACCESS_WIDTH, HALFWORD_ACCESS_WIDTH, WORD_ACCESS_WIDTH,
};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct StorePreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
    e: u8,
}

impl<A, const STORE_WIDTH: usize, const NUM_BLOCKS: usize>
    StoreExecutor<A, STORE_WIDTH, NUM_BLOCKS>
{
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut StorePreCompute,
    ) -> Result<Rv64LoadStoreOpcode, StaticProgramError> {
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
        if !enabled {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV64_REGISTER_AS
            || (e_u32 != RV64_MEMORY_AS && e_u32 != PUBLIC_VALUES_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = Rv64LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv64LoadStoreOpcode::CLASS_OFFSET),
        );
        match local_opcode {
            STORED | STOREW | STOREH | STOREB
                if store_width_for_opcode(local_opcode) == STORE_WIDTH => {}
            _ => return Err(StaticProgramError::InvalidInstruction(pc)),
        }

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        *data = StorePreCompute {
            imm_extended: sign_extend_imm16(imm, imm_sign),
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            e: e_u32 as u8,
        };
        Ok(local_opcode)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        match $local_opcode {
            STORED => Ok($execute_impl::<_, StoreDOp>),
            STOREW => Ok($execute_impl::<_, StoreWOp>),
            STOREH => Ok($execute_impl::<_, StoreHOp>),
            STOREB => Ok($execute_impl::<_, StoreBOp>),
            _ => Err(StaticProgramError::InvalidInstruction(0)),
        }
    };
}

impl<F, A, const STORE_WIDTH: usize, const NUM_BLOCKS: usize> InterpreterExecutor<F>
    for StoreExecutor<A, STORE_WIDTH, NUM_BLOCKS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<StorePreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let pre_compute: &mut StorePreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut StorePreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F, A, const STORE_WIDTH: usize, const NUM_BLOCKS: usize> InterpreterMeteredExecutor<F>
    for StoreExecutor<A, STORE_WIDTH, NUM_BLOCKS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<StorePreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<StorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<StorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, OP: StoreOp>(
    pre_compute: &StorePreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();
    let rs1_bytes: [u8; RV64_REGISTER_NUM_LIMBS] =
        exec_state.vm_read_bytes(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs1_val = rv64_bytes_to_u32(rs1_bytes);
    let addr = rv64_address_add_imm(rs1_val, pre_compute.imm_extended);
    debug_assert!(addr <= (MEM_SIZE - OP::WIDTH) as u64);
    let ptr_val = addr as u32;
    OP::write(
        exec_state,
        pre_compute.e as u32,
        ptr_val,
        pre_compute.a as u32,
    );
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, OP: StoreOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &StorePreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<StorePreCompute>()).borrow();
    execute_e12_impl::<CTX, OP>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, OP: StoreOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<StorePreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<StorePreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, OP>(&pre_compute.data, exec_state)
}

trait StoreOp {
    /// Access width in bytes.
    const WIDTH: usize;

    fn write<CTX: ExecutionCtxTrait>(
        exec_state: &mut VmExecState<GuestMemory, CTX>,
        address_space: u32,
        ptr: u32,
        rs2_ptr: u32,
    );
}

struct StoreDOp;
struct StoreWOp;
struct StoreHOp;
struct StoreBOp;

impl StoreOp for StoreDOp {
    const WIDTH: usize = DOUBLEWORD_ACCESS_WIDTH;

    #[inline(always)]
    fn write<CTX: ExecutionCtxTrait>(
        exec_state: &mut VmExecState<GuestMemory, CTX>,
        address_space: u32,
        ptr: u32,
        rs2_ptr: u32,
    ) {
        let value: [u8; Self::WIDTH] = exec_state.vm_read_bytes(RV64_REGISTER_AS, rs2_ptr);
        exec_state.vm_write_bytes(address_space, ptr, &value);
    }
}

impl StoreOp for StoreWOp {
    const WIDTH: usize = WORD_ACCESS_WIDTH;

    #[inline(always)]
    fn write<CTX: ExecutionCtxTrait>(
        exec_state: &mut VmExecState<GuestMemory, CTX>,
        address_space: u32,
        ptr: u32,
        rs2_ptr: u32,
    ) {
        let value: [u8; Self::WIDTH] = exec_state.vm_read_bytes(RV64_REGISTER_AS, rs2_ptr);
        exec_state.vm_write_bytes(address_space, ptr, &value);
    }
}

impl StoreOp for StoreHOp {
    const WIDTH: usize = HALFWORD_ACCESS_WIDTH;

    #[inline(always)]
    fn write<CTX: ExecutionCtxTrait>(
        exec_state: &mut VmExecState<GuestMemory, CTX>,
        address_space: u32,
        ptr: u32,
        rs2_ptr: u32,
    ) {
        let value: [u8; Self::WIDTH] = exec_state.vm_read_bytes(RV64_REGISTER_AS, rs2_ptr);
        exec_state.vm_write_bytes(address_space, ptr, &value);
    }
}

impl StoreOp for StoreBOp {
    const WIDTH: usize = BYTE_ACCESS_WIDTH;

    #[inline(always)]
    fn write<CTX: ExecutionCtxTrait>(
        exec_state: &mut VmExecState<GuestMemory, CTX>,
        address_space: u32,
        ptr: u32,
        rs2_ptr: u32,
    ) {
        let value: [u8; Self::WIDTH] = exec_state.vm_read_bytes(RV64_REGISTER_AS, rs2_ptr);
        exec_state.vm_write_bytes(address_space, ptr, &value);
    }
}
