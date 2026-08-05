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
    riscv::{MEMORY_AS, REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::LoadStoreOpcode::{self, STOREB, STORED, STOREH, STOREW};
use openvm_stark_backend::p3_field::PrimeField32;

use super::common::{store_width_for_opcode, StoreExecutor};
use crate::adapters::{
    bytes_to_u32, checked_memory_address, sign_extend_imm16, BYTE_ACCESS_WIDTH,
    DOUBLEWORD_ACCESS_WIDTH, HALFWORD_ACCESS_WIDTH, WORD_ACCESS_WIDTH,
};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct StorePreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
}

impl<const STORE_WIDTH: usize> StoreExecutor<STORE_WIDTH> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut StorePreCompute,
    ) -> Result<LoadStoreOpcode, StaticProgramError> {
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

        if d.as_canonical_u32() != REGISTER_AS || e.as_canonical_u32() != MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode =
            LoadStoreOpcode::from_usize(opcode.local_opcode_idx(LoadStoreOpcode::CLASS_OFFSET));
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

impl<F, const STORE_WIDTH: usize> InterpreterExecutor<F> for StoreExecutor<STORE_WIDTH>
where
    F: PrimeField32,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", LoadStoreOpcode::from_usize(opcode - self.offset))
    }

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

impl<F, const STORE_WIDTH: usize> InterpreterMeteredExecutor<F> for StoreExecutor<STORE_WIDTH>
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
    let rs1_bytes: [u8; REGISTER_NUM_LIMBS] =
        exec_state.vm_read_bytes(REGISTER_AS, pre_compute.b as u32);
    let rs1_val = bytes_to_u32(rs1_bytes);
    let ptr_val = checked_memory_address(pc, rs1_val, pre_compute.imm_extended, OP::WIDTH)?;
    OP::write(exec_state, ptr_val, pre_compute.a as u32);
    if OP::WIDTH != BYTE_ACCESS_WIDTH
        && ptr_val as usize % DOUBLEWORD_ACCESS_WIDTH + OP::WIDTH <= DOUBLEWORD_ACCESS_WIDTH
    {
        exec_state.ctx.advance_timestamp(1);
    }
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
        ptr: u32,
        rs2_ptr: u32,
    ) {
        let value: [u8; Self::WIDTH] = exec_state.vm_read_bytes(REGISTER_AS, rs2_ptr);
        exec_state.vm_write_bytes(MEMORY_AS, ptr, &value);
    }
}

impl StoreOp for StoreWOp {
    const WIDTH: usize = WORD_ACCESS_WIDTH;

    #[inline(always)]
    fn write<CTX: ExecutionCtxTrait>(
        exec_state: &mut VmExecState<GuestMemory, CTX>,
        ptr: u32,
        rs2_ptr: u32,
    ) {
        let value: [u8; Self::WIDTH] = exec_state.vm_read_bytes(REGISTER_AS, rs2_ptr);
        exec_state.vm_write_bytes(MEMORY_AS, ptr, &value);
    }
}

impl StoreOp for StoreHOp {
    const WIDTH: usize = HALFWORD_ACCESS_WIDTH;

    #[inline(always)]
    fn write<CTX: ExecutionCtxTrait>(
        exec_state: &mut VmExecState<GuestMemory, CTX>,
        ptr: u32,
        rs2_ptr: u32,
    ) {
        let value: [u8; Self::WIDTH] = exec_state.vm_read_bytes(REGISTER_AS, rs2_ptr);
        exec_state.vm_write_bytes(MEMORY_AS, ptr, &value);
    }
}

impl StoreOp for StoreBOp {
    const WIDTH: usize = BYTE_ACCESS_WIDTH;

    #[inline(always)]
    fn write<CTX: ExecutionCtxTrait>(
        exec_state: &mut VmExecState<GuestMemory, CTX>,
        ptr: u32,
        rs2_ptr: u32,
    ) {
        let value: [u8; Self::WIDTH] = exec_state.vm_read_bytes(REGISTER_AS, rs2_ptr);
        exec_state.vm_write_bytes(MEMORY_AS, ptr, &value);
    }
}

#[cfg(test)]
mod tests {
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{MEMORY_AS, REGISTER_AS},
        LocalOpcode,
    };
    use openvm_riscv_transpiler::LoadStoreOpcode::{self, STOREB, STORED, STOREH, STOREW};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::{StoreExecutor, StorePreCompute};
    use crate::adapters::{
        BYTE_ACCESS_WIDTH, DOUBLEWORD_ACCESS_WIDTH, HALFWORD_ACCESS_WIDTH, WORD_ACCESS_WIDTH,
    };

    fn instruction(opcode: LoadStoreOpcode, address_space: u32) -> Instruction<BabyBear> {
        Instruction::from_usize(
            opcode.global_opcode(),
            [8, 16, 0, REGISTER_AS as usize, address_space as usize, 1, 0],
        )
    }

    fn assert_address_space<const WIDTH: usize>(opcode: LoadStoreOpcode) {
        let executor = StoreExecutor::<WIDTH>::new(LoadStoreOpcode::CLASS_OFFSET);
        let mut data = StorePreCompute {
            imm_extended: 0,
            a: 0,
            b: 0,
        };
        assert!(executor
            .pre_compute_impl(4, &instruction(opcode, MEMORY_AS), &mut data)
            .is_ok());
        assert!(executor
            .pre_compute_impl(4, &instruction(opcode, MEMORY_AS + 1), &mut data)
            .is_err());
    }

    #[test]
    fn ordinary_stores_reject_non_memory_address_spaces() {
        assert_address_space::<DOUBLEWORD_ACCESS_WIDTH>(STORED);
        assert_address_space::<WORD_ACCESS_WIDTH>(STOREW);
        assert_address_space::<HALFWORD_ACCESS_WIDTH>(STOREH);
        assert_address_space::<BYTE_ACCESS_WIDTH>(STOREB);
    }
}
