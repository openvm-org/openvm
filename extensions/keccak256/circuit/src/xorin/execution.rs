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
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::XorinVmExecutor;
use crate::KECCAK_MEMORY_BLOCK;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct XorinPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl XorinVmExecutor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut XorinPreCompute,
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
        if d.as_canonical_u32() != RV64_REGISTER_AS || e_u32 != RV64_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        *data = XorinPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };

        Ok(())
    }
}

impl<F: PrimeField32> InterpreterExecutor<F> for XorinVmExecutor {
    fn pre_compute_size(&self) -> usize {
        size_of::<XorinPreCompute>()
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
        let data: &mut XorinPreCompute = data.borrow_mut();
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
        let data: &mut XorinPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for XorinVmExecutor {}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for XorinVmExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<XorinPreCompute>>()
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
        let data: &mut E2PreCompute<XorinPreCompute> = data.borrow_mut();
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
        let data: &mut E2PreCompute<XorinPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for XorinVmExecutor {}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &XorinPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<XorinPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, true>(pre_compute, exec_state);
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_E1: bool>(
    pre_compute: &XorinPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let buffer: [u8; 8] = exec_state.vm_read(RV64_REGISTER_AS, pre_compute.a as u32);
    let input: [u8; 8] = exec_state.vm_read(RV64_REGISTER_AS, pre_compute.b as u32);
    let length: [u8; 8] = exec_state.vm_read(RV64_REGISTER_AS, pre_compute.c as u32);
    let buffer_u64 = u64::from_le_bytes(buffer);
    let input_u64 = u64::from_le_bytes(input);
    let length_u64 = u64::from_le_bytes(length);
    debug_assert_eq!(buffer_u64 >> 32, 0, "xorin buffer pointer upper 4 bytes must be zero");
    debug_assert_eq!(input_u64 >> 32, 0, "xorin input pointer upper 4 bytes must be zero");
    debug_assert_eq!(length_u64 >> 32, 0, "xorin length upper 4 bytes must be zero");
    let buffer_u32 = buffer_u64 as u32;
    let input_u32 = input_u64 as u32;
    let length_u32 = length_u64 as u32;

    // SAFETY: RV64_MEMORY_AS is memory address space of type u8
    let num_reads = (length_u32 as usize).div_ceil(KECCAK_MEMORY_BLOCK);
    let buffer_bytes: Vec<_> = (0..num_reads)
        .flat_map(|i| {
            exec_state.vm_read::<u8, KECCAK_MEMORY_BLOCK>(
                RV64_MEMORY_AS,
                buffer_u32 + (i * KECCAK_MEMORY_BLOCK) as u32,
            )
        })
        .collect();

    let input_bytes: Vec<_> = (0..num_reads)
        .flat_map(|i| {
            exec_state.vm_read::<u8, KECCAK_MEMORY_BLOCK>(
                RV64_MEMORY_AS,
                input_u32 + (i * KECCAK_MEMORY_BLOCK) as u32,
            )
        })
        .collect();

    let mut output_bytes = buffer_bytes;
    // Only XOR the active bytes (first length_u32 bytes).
    // Padding bytes in boundary 8-byte blocks remain unchanged.
    for i in 0..(length_u32 as usize) {
        output_bytes[i] ^= input_bytes[i];
    }

    // Write XOR result back to the buffer memory in 8-byte blocks.
    // Note: this means output_bytes length is a multiple of KECCAK_MEMORY_BLOCK
    for (i, chunk) in output_bytes
        .chunks_exact(KECCAK_MEMORY_BLOCK)
        .enumerate()
    {
        let chunk: [u8; KECCAK_MEMORY_BLOCK] = chunk.try_into().unwrap();
        exec_state.vm_write::<u8, KECCAK_MEMORY_BLOCK>(
            RV64_MEMORY_AS,
            buffer_u32 + (i * KECCAK_MEMORY_BLOCK) as u32,
            &chunk,
        );
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
    let pre_compute: &E2PreCompute<XorinPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<XorinPreCompute>>())
            .borrow();
    execute_e12_impl::<F, CTX, false>(&pre_compute.data, exec_state);
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
}
