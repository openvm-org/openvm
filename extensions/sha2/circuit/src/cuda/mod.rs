use std::{marker::PhantomData, sync::Arc};

use openvm_circuit::{
    arch::cuda::postflight::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_instructions::{
    riscv::{MEMORY_AS, REGISTER_AS},
    LocalOpcode,
};
use openvm_sha2_air::{Sha256Config, Sha2Variant, Sha512Config};
use openvm_stark_backend::prover::AirProvingContext;

use crate::Sha2Config;

mod cuda_abi;

pub struct Sha2MainChipGpu<C: Sha2Config> {
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    _marker: PhantomData<C>,
}

impl<C: Sha2Config> Sha2MainChipGpu<C> {
    pub fn new(
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Self {
        Self {
            range_checker,
            pointer_max_bits,
            timestamp_max_bits,
            _marker: PhantomData,
        }
    }

    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let step_range = replay_plan.opcode_range(C::OPCODE.global_opcode());
        if step_range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let trace_height = next_power_of_two_or_zero(step_range.len());
        let trace =
            DeviceMatrix::<F>::with_capacity_on(trace_height, C::MAIN_CHIP_WIDTH, device_ctx);
        unsafe {
            match C::VARIANT {
                Sha2Variant::Sha256 => cuda_abi::sha256::sha256_main_replay_tracegen(
                    trace.buffer(),
                    trace_height,
                    program.instructions(),
                    program.pc_base(),
                    transcript.program_log(),
                    transcript.memory_log(),
                    transcript.initial_write_log(),
                    transcript.memory_predecessors(),
                    replay_plan.steps(),
                    step_range.start,
                    step_range.len(),
                    C::OPCODE.global_opcode().as_usize() as u32,
                    REGISTER_AS,
                    MEMORY_AS,
                    self.pointer_max_bits,
                    &self.range_checker.count,
                    self.timestamp_max_bits,
                    transcript.error_ptr(),
                    device_ctx.stream.as_raw(),
                )?,
                Sha2Variant::Sha512 | Sha2Variant::Sha384 => {
                    cuda_abi::sha512::sha512_main_replay_tracegen(
                        trace.buffer(),
                        trace_height,
                        program.instructions(),
                        program.pc_base(),
                        transcript.program_log(),
                        transcript.memory_log(),
                        transcript.initial_write_log(),
                        transcript.memory_predecessors(),
                        replay_plan.steps(),
                        step_range.start,
                        step_range.len(),
                        C::OPCODE.global_opcode().as_usize() as u32,
                        REGISTER_AS,
                        MEMORY_AS,
                        self.pointer_max_bits,
                        &self.range_checker.count,
                        self.timestamp_max_bits,
                        transcript.error_ptr(),
                        device_ctx.stream.as_raw(),
                    )?
                }
            }
        }
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}

/// Generic hybrid GPU wrapper that reuses CPU block-hasher tracegen.
pub struct Sha2BlockHasherChipGpu<C: Sha2Config> {
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<8>>,
    /// Range checker for digest-row `final_hash` limbs.
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pointer_max_bits: u32,
    _marker: PhantomData<C>,
}

impl<C: Sha2Config> Sha2BlockHasherChipGpu<C> {
    pub fn new(
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<8>>,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        pointer_max_bits: u32,
    ) -> Self {
        Self {
            bitwise_lookup,
            range_checker,
            pointer_max_bits,
            _marker: PhantomData,
        }
    }

    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.bitwise_lookup.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let step_range = replay_plan.opcode_range(C::OPCODE.global_opcode());
        if step_range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let rows_used = step_range
            .len()
            .checked_mul(C::ROWS_PER_BLOCK)
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "SHA-2 block-hasher replay row count overflow".to_string(),
                )
            })?;
        let trace_height = next_power_of_two_or_zero(rows_used);
        let trace =
            DeviceMatrix::<F>::with_capacity_on(trace_height, C::BLOCK_HASHER_WIDTH, device_ctx);
        u32::try_from(step_range.len()).map_err(|_| {
            GpuPostflightError::InvalidTranscript(
                "SHA-2 block-hasher replay block count exceeds u32".to_string(),
            )
        })?;
        let prev_hash_words = step_range.len().checked_mul(C::HASH_WORDS).ok_or_else(|| {
            GpuPostflightError::InvalidTranscript(
                "SHA-2 block-hasher replay hash scratch size overflow".to_string(),
            )
        })?;
        let scratch_words = step_range
            .len()
            .checked_mul(C::ROWS_PER_BLOCK)
            .and_then(|words| words.checked_mul(8 + C::BLOCK_WORDS))
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "SHA-2 block-hasher replay scratch size overflow".to_string(),
                )
            })?;
        unsafe {
            match C::VARIANT {
                Sha2Variant::Sha256 => {
                    let prev_hashes =
                        DeviceBuffer::<u32>::with_capacity_on(prev_hash_words, device_ctx);
                    let scratch = DeviceBuffer::<u32>::with_capacity_on(scratch_words, device_ctx);
                    cuda_abi::sha256::sha256_block_replay_tracegen(
                        trace.buffer(),
                        trace_height,
                        program.instructions(),
                        program.pc_base(),
                        transcript.program_log(),
                        transcript.memory_log(),
                        transcript.initial_write_log(),
                        transcript.memory_predecessors(),
                        replay_plan.steps(),
                        step_range.start,
                        step_range.len(),
                        C::OPCODE.global_opcode().as_usize() as u32,
                        REGISTER_AS,
                        MEMORY_AS,
                        self.pointer_max_bits,
                        &prev_hashes,
                        &self.bitwise_lookup.count,
                        &scratch,
                        &self.range_checker.count,
                        transcript.error_ptr(),
                        device_ctx.stream.as_raw(),
                    )?;
                    cuda_abi::sha256::sha256_fill_invalid_rows(
                        trace.buffer(),
                        trace_height,
                        rows_used,
                        &prev_hashes,
                        device_ctx.stream.as_raw(),
                    )?;
                    cuda_abi::sha256::sha256_second_pass_dependencies(
                        trace.buffer(),
                        trace_height,
                        rows_used,
                        device_ctx.stream.as_raw(),
                    )?;
                    drop(scratch);
                    drop(prev_hashes);
                }
                Sha2Variant::Sha512 | Sha2Variant::Sha384 => {
                    let prev_hashes =
                        DeviceBuffer::<u64>::with_capacity_on(prev_hash_words, device_ctx);
                    let scratch = DeviceBuffer::<u64>::with_capacity_on(scratch_words, device_ctx);
                    cuda_abi::sha512::sha512_block_replay_tracegen(
                        trace.buffer(),
                        trace_height,
                        program.instructions(),
                        program.pc_base(),
                        transcript.program_log(),
                        transcript.memory_log(),
                        transcript.initial_write_log(),
                        transcript.memory_predecessors(),
                        replay_plan.steps(),
                        step_range.start,
                        step_range.len(),
                        C::OPCODE.global_opcode().as_usize() as u32,
                        REGISTER_AS,
                        MEMORY_AS,
                        self.pointer_max_bits,
                        &prev_hashes,
                        &self.bitwise_lookup.count,
                        &scratch,
                        &self.range_checker.count,
                        transcript.error_ptr(),
                        device_ctx.stream.as_raw(),
                    )?;
                    cuda_abi::sha512::sha512_fill_invalid_rows(
                        trace.buffer(),
                        trace_height,
                        rows_used,
                        &prev_hashes,
                        device_ctx.stream.as_raw(),
                    )?;
                    cuda_abi::sha512::sha512_second_pass_dependencies(
                        trace.buffer(),
                        trace_height,
                        rows_used,
                        device_ctx.stream.as_raw(),
                    )?;
                    drop(scratch);
                    drop(prev_hashes);
                }
            }
        }
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}

// Convenience aliases for the common SHA-2 variants.
pub type Sha256VmChipGpu = Sha2MainChipGpu<Sha256Config>;
pub type Sha256BlockHasherChipGpu = Sha2BlockHasherChipGpu<Sha256Config>;
pub type Sha512VmChipGpu = Sha2MainChipGpu<Sha512Config>;
pub type Sha512BlockHasherChipGpu = Sha2BlockHasherChipGpu<Sha512Config>;
