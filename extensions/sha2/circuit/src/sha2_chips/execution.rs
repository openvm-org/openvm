use std::borrow::{Borrow, BorrowMut};

#[cfg(not(feature = "tco"))]
use openvm_circuit::arch::{ExecutionCtxTrait, MeteredExecutionCtxTrait};
use openvm_circuit::{
    arch::{
        E2PreCompute, ExecuteFunc, InterpreterExecutor, InterpreterMeteredExecutor,
        StaticProgramError, VmExecState,
    },
    system::memory::online::GuestMemory,
};
use openvm_circuit_derive::create_handler;
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{Sha2Config, Sha2VmExecutor, SHA2_READ_SIZE};
use crate::SHA2_WRITE_SIZE;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct Sha2PreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<F: PrimeField32, C: Sha2Config> InterpreterExecutor<F> for Sha2VmExecutor<C> {
    fn pre_compute_size(&self) -> usize {
        size_of::<Sha2PreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut Sha2PreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_impl::<_, _, C>)
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
        let data: &mut Sha2PreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler::<_, _>)
    }
}

impl<F: PrimeField32, C: Sha2Config> InterpreterMeteredExecutor<F> for Sha2VmExecutor<C> {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Sha2PreCompute>>()
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
        let data: &mut E2PreCompute<Sha2PreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_impl::<_, _, C>)
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
        let data: &mut E2PreCompute<Sha2PreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler::<_, _, C>)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    C: Sha2Config,
    CTX: ExecutionCtxTrait,
    const IS_E1: bool,
>(
    pre_compute: &Sha2PreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let dst = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32);
    let state = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let input = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.c as u32);
    let dst_u32 = u32::from_le_bytes(dst);
    let state_u32 = u32::from_le_bytes(state);
    let input_u32 = u32::from_le_bytes(input);

    // state is in 4-byte little-endian words
    let mut state_data = Vec::with_capacity(C::STATE_BYTES);
    for i in 0..C::STATE_READS {
        state_data.extend_from_slice(&exec_state.vm_read::<u8, SHA2_READ_SIZE>(
            RV32_MEMORY_AS,
            state_u32 + (i * SHA2_READ_SIZE) as u32,
        ));
    }
    let mut input_block = Vec::with_capacity(C::BLOCK_BYTES);
    for i in 0..C::BLOCK_READS {
        input_block.extend_from_slice(&exec_state.vm_read::<u8, SHA2_READ_SIZE>(
            RV32_MEMORY_AS,
            input_u32 + (i * SHA2_READ_SIZE) as u32,
        ));
    }

    C::compress(&mut state_data, &input_block);

    for i in 0..C::STATE_WRITES {
        exec_state.vm_write::<u8, SHA2_WRITE_SIZE>(
            RV32_MEMORY_AS,
            dst_u32 + (i * SHA2_WRITE_SIZE) as u32,
            &state_data[i * SHA2_WRITE_SIZE..(i + 1) * SHA2_WRITE_SIZE]
                .try_into()
                .unwrap(),
        );
    }

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    1 // height delta
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, C: Sha2Config>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Sha2PreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Sha2PreCompute>()).borrow();
    execute_e12_impl::<F, C, CTX, true>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, C: Sha2Config>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Sha2PreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<Sha2PreCompute>>()).borrow();
    let height = execute_e12_impl::<F, C, CTX, false>(&pre_compute.data, exec_state);
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

#[cfg(feature = "aot")]
impl<F: PrimeField32, C: Sha2Config> AotExecutor<F> for Sha2VmExecutor<C> {}

#[cfg(feature = "aot")]
impl<F: PrimeField32, C: Sha2Config> AotMeteredExecutor<F> for Sha2VmExecutor<C> {}

impl<C: Sha2Config> Sha2VmExecutor<C> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Sha2PreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = Sha2PreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        assert_eq!(&C::OPCODE.global_opcode(), opcode);
        Ok(())
    }
}
