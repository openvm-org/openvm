use std::{
    borrow::{Borrow, BorrowMut},
    convert::TryInto,
    mem::size_of,
};

use openvm_circuit::{
    arch::{StaticProgramError, *},
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_stark_backend::p3_field::PrimeField32;
use p3_keccak_air::NUM_ROUNDS;

use super::KeccakfVmExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct KeccakfPreCompute {
    a: u8,
}

impl KeccakfVmExecutor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut KeccakfPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            opcode: _,
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

        *data = KeccakfPreCompute {
            a: a.as_canonical_u32() as u8,
        };

        Ok(())
    }
}

impl<F: PrimeField32> InterpreterExecutor<F> for KeccakfVmExecutor {
    fn pre_compute_size(&self) -> usize {
        size_of::<KeccakfPreCompute>()
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
        let data: &mut KeccakfPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_impl::<_, _>)
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
        todo!()
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for KeccakfVmExecutor {}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for KeccakfVmExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<KeccakfPreCompute>()
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
        let data: &mut E2PreCompute<KeccakfPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_impl::<_, _>)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        _chip_idx: usize,
        _pc: u32,
        _inst: &Instruction<F>,
        _data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        todo!()
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &KeccakfPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<KeccakfPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, true>(pre_compute, exec_state);
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_E1: bool>(
    pre_compute: &KeccakfPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    use tiny_keccak::keccakf;

    // the variable naming might be misleading
    // buf_ptr is the pointer to the register which holds the actual pointer to the buffer
    let buf_ptr = pre_compute.a as u32;
    let buffer_limbs: [u8; 4] = exec_state.vm_read(RV32_REGISTER_AS, buf_ptr);
    let buffer = u32::from_le_bytes(buffer_limbs);
    let message: &[u8] = exec_state.vm_read_slice(RV32_MEMORY_AS, buffer, 200);
    assert_eq!(message.len(), 200);

    // todo: check if this is correct representation / conversion between u8 <-> u64
    let mut message_u64 = [0u64; 25];
    for (i, message_chunk) in message.chunks_exact(8).enumerate() {
        let message_chunk_u64 = u64::from_le_bytes(message_chunk.try_into().unwrap());
        message_u64[i] = message_chunk_u64;
    }
    keccakf(&mut message_u64);

    let mut result: [u8; 200] = [0; 200];
    for (i, message) in message_u64.into_iter().enumerate() {
        let message_bytes = message.to_be_bytes();
        result[8 * i..8 * i + 8].copy_from_slice(&message_bytes);
    }

    exec_state.vm_write(RV32_MEMORY_AS, buffer, &result);

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
#[allow(dead_code)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<KeccakfPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<KeccakfPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, NUM_ROUNDS as u32);
    execute_e12_impl::<F, CTX, false>(&pre_compute.data, exec_state);
}
