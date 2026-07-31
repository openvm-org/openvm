use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    slice::from_raw_parts,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_riscv_circuit::adapters::rv64_bytes_to_u32;
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use super::{checked_deferral_index, DeferralOutputExecutor};
use crate::{
    utils::{
        byte_memory_op_chunk, join_byte_memory_ops, split_output, DIGEST_BYTE_MEMORY_OPS,
        OUTPUT_TOTAL_BYTES, OUTPUT_TOTAL_MEMORY_OPS,
    },
    OUTPUT_AIR_REL_IDX, POSEIDON2_AIR_REL_IDX,
};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct DeferralOutputPrecompute {
    rd_ptr: u32,
    rs_ptr: u32,
    deferral_idx: u32,
}

#[inline(always)]
fn checked_output_len(pc: u32, output_len: [u8; 8]) -> Result<u32, ExecutionError> {
    let output_len =
        u32::try_from(u64::from_le_bytes(output_len)).map_err(|_| ExecutionError::Fail {
            pc,
            msg: "deferral output length exceeds u32",
        })?;
    if !output_len.is_multiple_of(DIGEST_SIZE as u32) {
        return Err(ExecutionError::Fail {
            pc,
            msg: "deferral output length must be a whole sponge row",
        });
    }
    Ok(output_len)
}

#[inline(always)]
fn check_block_aligned_ptr(pc: u32, ptr: u32) -> Result<u32, ExecutionError> {
    if !ptr.is_multiple_of(MEMORY_BLOCK_BYTES as u32) {
        return Err(ExecutionError::Fail {
            pc,
            msg: "deferral pointers must be eight-byte aligned",
        });
    }
    Ok(ptr)
}

impl DeferralOutputExecutor {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut DeferralOutputPrecompute,
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

        if opcode.local_opcode_idx(DeferralOpcode::CLASS_OFFSET) != DeferralOpcode::OUTPUT as usize
            || d.as_canonical_u32() != RV64_REGISTER_AS
            || e.as_canonical_u32() != RV64_MEMORY_AS
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        *data = DeferralOutputPrecompute {
            rd_ptr: a.as_canonical_u32(),
            rs_ptr: b.as_canonical_u32(),
            deferral_idx: c.as_canonical_u32(),
        };
        Ok(())
    }
}

impl<F: PrimeField32> InterpreterExecutor<F> for DeferralOutputExecutor {
    fn get_opcode_name(&self, _opcode: usize) -> String {
        format!("{:?}", DeferralOpcode::OUTPUT)
    }

    fn pre_compute_size(&self) -> usize {
        size_of::<DeferralOutputPrecompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut DeferralOutputPrecompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_e1_handler::<_>)
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
        let pre_compute: &mut DeferralOutputPrecompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_e1_handler::<_>)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for DeferralOutputExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<DeferralOutputPrecompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        air_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<DeferralOutputPrecompute> = data.borrow_mut();
        pre_compute.chip_idx = air_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_e2_handler::<_>)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        air_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<DeferralOutputPrecompute> = data.borrow_mut();
        pre_compute.chip_idx = air_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_e2_handler::<_>)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait>(
    pre_compute: &DeferralOutputPrecompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<u32, ExecutionError> {
    let pc = exec_state.pc();
    let deferral_idx = checked_deferral_index(
        pc,
        exec_state.streams.deferrals.len(),
        pre_compute.deferral_idx,
    )?;
    let output_ptr = check_block_aligned_ptr(
        pc,
        rv64_bytes_to_u32(exec_state.vm_read_bytes(RV64_REGISTER_AS, pre_compute.rd_ptr)),
    )?;
    let input_ptr = check_block_aligned_ptr(
        pc,
        rv64_bytes_to_u32(exec_state.vm_read_bytes(RV64_REGISTER_AS, pre_compute.rs_ptr)),
    )?;
    let output_key_chunks: [[u8; MEMORY_BLOCK_BYTES]; OUTPUT_TOTAL_MEMORY_OPS] = from_fn(|i| {
        exec_state.vm_read_bytes(RV64_MEMORY_AS, input_ptr + (i * MEMORY_BLOCK_BYTES) as u32)
    });
    let output_key: [u8; OUTPUT_TOTAL_BYTES] = join_byte_memory_ops(output_key_chunks);
    let (output_commit, output_len) = split_output(output_key);

    let output_len_val = checked_output_len(pc, output_len)? as usize;

    // Bytes are sponge-hashed and constrained against output_commit. The
    // sponge rate is DIGEST_SIZE.
    let num_rows = output_len_val / DIGEST_SIZE + 1;
    let output_raw = exec_state.streams.deferrals[deferral_idx]
        .try_get_output(&output_commit.to_vec())
        .filter(|output| output.len() == output_len_val)
        .cloned()
        .ok_or(ExecutionError::Fail {
            pc,
            msg: "deferral output advice is missing or has the wrong length",
        })?;

    for (row_idx, output_chunk) in output_raw.chunks_exact(DIGEST_SIZE).enumerate() {
        let row_output_ptr = output_ptr + (row_idx * DIGEST_SIZE) as u32;
        for chunk_idx in 0..DIGEST_BYTE_MEMORY_OPS {
            exec_state.vm_write_bytes::<MEMORY_BLOCK_BYTES>(
                RV64_MEMORY_AS,
                row_output_ptr + (chunk_idx * MEMORY_BLOCK_BYTES) as u32,
                &byte_memory_op_chunk(output_chunk, chunk_idx),
            );
        }
    }

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
    Ok(num_rows as u32)
}

#[cfg(test)]
mod tests {
    use super::{checked_output_len, DIGEST_SIZE};

    #[test]
    fn output_length_accepts_u32_boundary_and_rejects_high_word() {
        let max_aligned = u32::MAX - (DIGEST_SIZE as u32 - 1);
        assert_eq!(
            checked_output_len(7, u64::from(max_aligned).to_le_bytes()).unwrap(),
            max_aligned
        );
        let error = checked_output_len(7, (u64::from(u32::MAX) + 1).to_le_bytes()).unwrap_err();
        assert_eq!(
            error.to_string(),
            "execution failed at pc 7, err: deferral output length exceeds u32"
        );
    }

    #[test]
    fn output_length_rejects_partial_sponge_row() {
        let error = checked_output_len(7, 1u64.to_le_bytes()).unwrap_err();
        assert_eq!(
            error.to_string(),
            "execution failed at pc 7, err: deferral output length must be a whole sponge row"
        );
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &DeferralOutputPrecompute =
        from_raw_parts(pre_compute, size_of::<DeferralOutputPrecompute>()).borrow();
    execute_e12_impl(pre_compute, exec_state)?;
    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<DeferralOutputPrecompute> = from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<DeferralOutputPrecompute>>(),
    )
    .borrow();
    let height = execute_e12_impl(&pre_compute.data, exec_state)?;
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);

    // The Poseidon2 peripheral chip's height also increases as a result of
    // this opcode's execution. Computing an output commit from the raw output
    // takes height Poseidon2 compressions.
    exec_state.ctx.on_height_change(
        pre_compute.chip_idx as usize + (OUTPUT_AIR_REL_IDX - POSEIDON2_AIR_REL_IDX),
        height,
    );
    Ok(())
}
