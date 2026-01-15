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

use super::{KeccakfExecutor, NUM_OP_ROWS_PER_INS};
use crate::{keccakf_op::keccakf_postimage_bytes, KECCAK_WIDTH_BYTES, KECCAK_WORD_SIZE};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct KeccakfPreCompute {
    a: u8,
}

impl KeccakfExecutor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut KeccakfPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            opcode: _,
            a,
            b: _,
            c: _,
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

impl<F: PrimeField32> InterpreterExecutor<F> for KeccakfExecutor {
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
        let data: &mut KeccakfPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for KeccakfExecutor {}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for KeccakfExecutor {
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
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<KeccakfPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for KeccakfExecutor {}

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
    let rd_ptr = pre_compute.a as u32;
    let buffer_ptr_limbs: [u8; 4] = exec_state.vm_read(RV32_REGISTER_AS, rd_ptr);
    let buffer_ptr = u32::from_le_bytes(buffer_ptr_limbs);

    let preimage: &[u8] =
        exec_state.host_read_slice(RV32_MEMORY_AS, buffer_ptr, KECCAK_WIDTH_BYTES);
    let postimage = keccakf_postimage_bytes(preimage.try_into().unwrap());

    if IS_E1 {
        exec_state.vm_write(RV32_MEMORY_AS, buffer_ptr, &postimage);
    } else {
        for (word_idx, word) in postimage.chunks_exact(KECCAK_WORD_SIZE).enumerate() {
            exec_state.vm_write::<u8, KECCAK_WORD_SIZE>(
                RV32_MEMORY_AS,
                buffer_ptr + (word_idx * KECCAK_WORD_SIZE) as u32,
                word.try_into().unwrap(),
            );
        }
    }

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<KeccakfPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<KeccakfPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, NUM_OP_ROWS_PER_INS as u32);
    execute_e12_impl::<F, CTX, false>(&pre_compute.data, exec_state);
}
