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

use super::{XorinVmExecutor, KECCAK_WORD_SIZE};

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
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 != RV32_MEMORY_AS {
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
        size_of::<XorinPreCompute>()
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
    let buffer = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32);
    let input = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let length = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.c as u32);
    let buffer_u32 = u32::from_le_bytes(buffer);
    let input_u32 = u32::from_le_bytes(input);
    let length_u32 = u32::from_le_bytes(length);

    // SAFETY: RV32_MEMORY_AS is memory address space of type u8
    let num_reads = (length_u32 as usize).div_ceil(KECCAK_WORD_SIZE);
    let buffer_bytes: Vec<_> = (0..num_reads)
        .flat_map(|i| {
            exec_state.vm_read::<u8, KECCAK_WORD_SIZE>(
                RV32_MEMORY_AS,
                buffer_u32 + (i * KECCAK_WORD_SIZE) as u32,
            )
        })
        .collect();

    let input_bytes: Vec<_> = (0..num_reads)
        .flat_map(|i| {
            exec_state.vm_read::<u8, KECCAK_WORD_SIZE>(
                RV32_MEMORY_AS,
                input_u32 + (i * KECCAK_WORD_SIZE) as u32,
            )
        })
        .collect();

    let mut output_bytes = buffer_bytes;
    for i in 0..output_bytes.len() {
        output_bytes[i] ^= input_bytes[i];
    }

    // Write XOR result back to the buffer memory in KECCAK_WORD_SIZE chunks.
    // Note: this means output_bytes has to be multiple of KECCAK_WORD_SIZE
    // Todo: recheck the above condition is okay
    for (i, chunk) in output_bytes.chunks_exact(KECCAK_WORD_SIZE).enumerate() {
        let chunk: [u8; KECCAK_WORD_SIZE] = chunk.try_into().unwrap();
        exec_state.vm_write::<u8, KECCAK_WORD_SIZE>(
            RV32_MEMORY_AS,
            buffer_u32 + (i * KECCAK_WORD_SIZE) as u32,
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
