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
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use super::DeferralOutputExecutor;
use crate::{
    utils::{split_output, OUTPUT_TOTAL_BYTES},
    OUTPUT_AIR_IDX, POSEIDON2_AIR_IDX,
};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct DeferralOutputPrecompute {
    rd_ptr: u32,
    rs_ptr: u32,
    deferral_idx: u32,
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
            || d.as_canonical_u32() != RV32_REGISTER_AS
            || e.as_canonical_u32() != RV32_MEMORY_AS
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
    fn pre_compute_size(&self) -> usize {
        size_of::<DeferralOutputPrecompute>()
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
        let pre_compute: &mut DeferralOutputPrecompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
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
        let pre_compute: &mut DeferralOutputPrecompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_e1_handler::<_, _>)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for DeferralOutputExecutor {}

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
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<DeferralOutputPrecompute> = data.borrow_mut();
        pre_compute.chip_idx = air_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_e2_impl::<_, _>)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        air_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<DeferralOutputPrecompute> = data.borrow_mut();
        pre_compute.chip_idx = air_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_e2_handler::<_, _>)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for DeferralOutputExecutor {}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &DeferralOutputPrecompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let output_ptr = u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, pre_compute.rd_ptr));
    let input_ptr = u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, pre_compute.rs_ptr));
    let output_key = exec_state.vm_read::<u8, OUTPUT_TOTAL_BYTES>(RV32_MEMORY_AS, input_ptr);
    let (output_commit, output_len) = split_output(output_key);

    let output_len_val = u32::from_le_bytes(output_len) as usize;

    // Bytes are sponge-hashed and constrained against output_commit. Thhe
    // sponge rate is DIGEST_SIZE.
    let num_rows = output_len_val / DIGEST_SIZE;
    debug_assert_eq!(output_len_val % DIGEST_SIZE, 0);

    let output_raw = exec_state.streams.deferrals[pre_compute.deferral_idx as usize]
        .get_output(&output_commit.to_vec())
        .clone();
    debug_assert_eq!(output_raw.len(), output_len_val);

    for (row_idx, output_chunk) in output_raw.chunks_exact(DIGEST_SIZE).enumerate() {
        let row_output_ptr = output_ptr + (row_idx * DIGEST_SIZE) as u32;
        exec_state.vm_write::<u8, DIGEST_SIZE>(
            RV32_MEMORY_AS,
            row_output_ptr,
            &from_fn(|i| output_chunk[i]),
        );
    }

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
    num_rows as u32
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &DeferralOutputPrecompute =
        from_raw_parts(pre_compute, size_of::<DeferralOutputPrecompute>()).borrow();
    execute_e12_impl(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<DeferralOutputPrecompute> = from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<DeferralOutputPrecompute>>(),
    )
    .borrow();
    let height = execute_e12_impl(&pre_compute.data, exec_state);
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);

    // The Poseidon2 peripheral chip's height also increases as a result of
    // this opcode's execution. Computing an output commit from the raw output
    // takes height Poseidon2 compressions.
    exec_state.ctx.on_height_change(
        pre_compute.chip_idx as usize + (POSEIDON2_AIR_IDX - OUTPUT_AIR_IDX),
        height,
    );
}
