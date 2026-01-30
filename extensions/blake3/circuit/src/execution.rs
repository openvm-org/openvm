use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_blake3_transpiler::Rv32Blake3Opcode;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{Blake3VmExecutor, BLAKE3_WORD_SIZE};
use crate::{
    utils::{blake3_hash_p3_full_blocks, num_blake3_compressions},
    BLAKE3_BLOCK_BYTES,
};

/// Pre-computed instruction data extracted during decode phase
#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct Blake3PreCompute {
    a: u8, // dst register pointer
    b: u8, // src register pointer
    c: u8, // len register pointer
}

impl Blake3VmExecutor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Blake3PreCompute,
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

        *data = Blake3PreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };

        assert_eq!(&Rv32Blake3Opcode::BLAKE3.global_opcode(), opcode);
        Ok(())
    }
}

impl<F: PrimeField32> InterpreterExecutor<F> for Blake3VmExecutor {
    fn pre_compute_size(&self) -> usize {
        size_of::<Blake3PreCompute>()
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
        let data: &mut Blake3PreCompute = data.borrow_mut();
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
        let data: &mut Blake3PreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for Blake3VmExecutor {}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Blake3VmExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Blake3PreCompute>>()
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
        let data: &mut E2PreCompute<Blake3PreCompute> = data.borrow_mut();
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
        let data: &mut E2PreCompute<Blake3PreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler::<_, _>)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Blake3VmExecutor {}

/// Core execution logic shared between E1 (non-metered) and E2 (metered) modes
#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_E1: bool>(
    pre_compute: &Blake3PreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    // Read register values: dst pointer, src pointer, length
    let dst = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32);
    let src = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let len = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.c as u32);

    let dst_u32 = u32::from_le_bytes(dst);
    let src_u32 = u32::from_le_bytes(src);
    let len_u32 = u32::from_le_bytes(len);

    let (output, height) = if IS_E1 {
        // E1: Simple execution, read full blocks and compute hash
        let num_blocks = num_blake3_compressions(len_u32 as usize);
        let full_blocks_len = num_blocks * BLAKE3_BLOCK_BYTES;
        let message = exec_state.vm_read_slice(RV32_MEMORY_AS, src_u32, full_blocks_len);
        let output = blake3_hash_p3_full_blocks(message);
        (output, 0)
    } else {
        // E2: Metered execution, track trace height
        // Read full blocks (including padding) to match AIR constraints
        let num_blocks = num_blake3_compressions(len_u32 as usize);
        let num_reads = num_blocks * (BLAKE3_BLOCK_BYTES / BLAKE3_WORD_SIZE);
        let message: Vec<_> = (0..num_reads)
            .flat_map(|i| {
                exec_state.vm_read::<u8, BLAKE3_WORD_SIZE>(
                    RV32_MEMORY_AS,
                    src_u32 + (i * BLAKE3_WORD_SIZE) as u32,
                )
            })
            .collect();
        let output = blake3_hash_p3_full_blocks(&message);
        // One row per compression
        let height = num_blocks as u32;
        (output, height)
    };

    // Write output digest to memory
    exec_state.vm_write(RV32_MEMORY_AS, dst_u32, &output);

    // Advance program counter
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    height
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Blake3PreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Blake3PreCompute>()).borrow();
    execute_e12_impl::<F, CTX, true>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Blake3PreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<Blake3PreCompute>>())
            .borrow();
    let height = execute_e12_impl::<F, CTX, false>(&pre_compute.data, exec_state);
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}
