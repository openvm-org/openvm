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
    LocalOpcode,
};
use openvm_platform::memory::MEM_SIZE;
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, LOADB, LOADH, LOADW};
use openvm_stark_backend::p3_field::PrimeField32;

use super::common::{load_sign_extend_width_for_opcode, LoadSignExtendExecutor};
use crate::adapters::{
    rv64_address_add_imm, rv64_bytes_to_u32, sign_extend_imm16, BYTE_ACCESS_WIDTH,
    HALFWORD_ACCESS_WIDTH, WORD_ACCESS_WIDTH,
};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct LoadSignExtendPreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
}

impl<A, const LOAD_WIDTH: usize, const NUM_BLOCKS: usize>
    LoadSignExtendExecutor<A, LOAD_WIDTH, NUM_BLOCKS>
{
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
        };
        Ok((local_opcode, enabled))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident, $enabled:ident) => {
        match ($local_opcode, $enabled) {
            (LOADW, true) => Ok($execute_impl::<_, LoadWOp, true>),
            (LOADW, false) => Ok($execute_impl::<_, LoadWOp, false>),
            (LOADH, true) => Ok($execute_impl::<_, LoadHOp, true>),
            (LOADH, false) => Ok($execute_impl::<_, LoadHOp, false>),
            (LOADB, true) => Ok($execute_impl::<_, LoadBOp, true>),
            (LOADB, false) => Ok($execute_impl::<_, LoadBOp, false>),
            _ => Err(StaticProgramError::InvalidInstruction(0)),
        }
    };
}

impl<F, A, const LOAD_WIDTH: usize, const NUM_BLOCKS: usize> InterpreterExecutor<F>
    for LoadSignExtendExecutor<A, LOAD_WIDTH, NUM_BLOCKS>
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
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
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
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut LoadSignExtendPreCompute = data.borrow_mut();
        let (local_opcode, enabled) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode, enabled)
    }
}

impl<F, A, const LOAD_WIDTH: usize, const NUM_BLOCKS: usize> InterpreterMeteredExecutor<F>
    for LoadSignExtendExecutor<A, LOAD_WIDTH, NUM_BLOCKS>
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
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
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
    ) -> Result<Handler<Ctx>, StaticProgramError>
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
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, OP: LoadSignExtendOp, const ENABLED: bool>(
    pre_compute: &LoadSignExtendPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();
    let rs1_bytes: [u8; RV64_REGISTER_NUM_LIMBS] =
        exec_state.vm_read_bytes(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs1_val = rv64_bytes_to_u32(rs1_bytes);
    let addr = rv64_address_add_imm(rs1_val, pre_compute.imm_extended);
    debug_assert!(addr <= (MEM_SIZE - OP::WIDTH) as u64);
    let ptr_val = addr as u32;
    let write_data = OP::read(exec_state, ptr_val);

    if ENABLED {
        exec_state.vm_write(RV64_REGISTER_AS, pre_compute.a as u32, &write_data);
    }
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, OP: LoadSignExtendOp, const ENABLED: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &LoadSignExtendPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<LoadSignExtendPreCompute>()).borrow();
    execute_e12_impl::<CTX, OP, ENABLED>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    CTX: MeteredExecutionCtxTrait,
    OP: LoadSignExtendOp,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<LoadSignExtendPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<LoadSignExtendPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, OP, ENABLED>(&pre_compute.data, exec_state)
}

trait LoadSignExtendOp {
    /// Access width in bytes.
    const WIDTH: usize;

    fn read<CTX: ExecutionCtxTrait>(
        exec_state: &mut VmExecState<GuestMemory, CTX>,
        ptr: u32,
    ) -> [u8; RV64_REGISTER_NUM_LIMBS];
}

struct LoadWOp;
struct LoadHOp;
struct LoadBOp;

impl LoadSignExtendOp for LoadWOp {
    const WIDTH: usize = WORD_ACCESS_WIDTH;

    #[inline(always)]
    fn read<CTX: ExecutionCtxTrait>(
        exec_state: &mut VmExecState<GuestMemory, CTX>,
        ptr: u32,
    ) -> [u8; RV64_REGISTER_NUM_LIMBS] {
        let word = i32::from_le_bytes(exec_state.vm_read_bytes(RV64_MEMORY_AS, ptr));
        (word as i64).to_le_bytes()
    }
}

impl LoadSignExtendOp for LoadHOp {
    const WIDTH: usize = HALFWORD_ACCESS_WIDTH;

    #[inline(always)]
    fn read<CTX: ExecutionCtxTrait>(
        exec_state: &mut VmExecState<GuestMemory, CTX>,
        ptr: u32,
    ) -> [u8; RV64_REGISTER_NUM_LIMBS] {
        let half = i16::from_le_bytes(exec_state.vm_read_bytes(RV64_MEMORY_AS, ptr));
        (half as i64).to_le_bytes()
    }
}

impl LoadSignExtendOp for LoadBOp {
    const WIDTH: usize = BYTE_ACCESS_WIDTH;

    #[inline(always)]
    fn read<CTX: ExecutionCtxTrait>(
        exec_state: &mut VmExecState<GuestMemory, CTX>,
        ptr: u32,
    ) -> [u8; RV64_REGISTER_NUM_LIMBS] {
        let [byte] = exec_state.vm_read_bytes(RV64_MEMORY_AS, ptr);
        (byte as i8 as i64).to_le_bytes()
    }
}
