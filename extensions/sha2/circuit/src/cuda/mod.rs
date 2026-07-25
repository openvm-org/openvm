use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use openvm_circuit::{
    arch::{DenseRecordArena, RecordSeeker},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_sha2_air::{Sha256Config, Sha2Variant, Sha512Config};
use openvm_stark_backend::prover::AirProvingContext;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::rvr::cuda::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    openvm_instructions::{
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
        LocalOpcode,
    },
};

use crate::{Sha2Config, Sha2RecordLayout, Sha2RecordMut};

mod cuda_abi;

pub struct Sha2SharedRecordsGpu {
    d_records: DeviceBuffer<u8>,
    d_record_offsets: DeviceBuffer<usize>,
    num_records: usize,
}

pub struct Sha2MainChipGpu<C: Sha2Config> {
    records: Arc<Mutex<Option<Sha2SharedRecordsGpu>>>,
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    _marker: PhantomData<C>,
}

impl<C: Sha2Config> Sha2MainChipGpu<C> {
    pub fn new(
        records: Arc<Mutex<Option<Sha2SharedRecordsGpu>>>,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Self {
        Self {
            records,
            range_checker,
            pointer_max_bits,
            timestamp_max_bits,
            _marker: PhantomData,
        }
    }

    #[cfg(feature = "rvr")]
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
                    RV64_REGISTER_AS,
                    RV64_MEMORY_AS,
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
                        RV64_REGISTER_AS,
                        RV64_MEMORY_AS,
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

impl<C> Chip<DenseRecordArena, GpuBackend> for Sha2MainChipGpu<C>
where
    C: Sha2Config,
{
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated_mut();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let mut record_offsets = Vec::<usize>::new();
        let mut offset = 0usize;
        while offset < records.len() {
            record_offsets.push(offset);
            let _record =
                RecordSeeker::<DenseRecordArena, Sha2RecordMut, Sha2RecordLayout>::get_record_at(
                    &mut offset,
                    records,
                );
        }

        let num_records = record_offsets.len();
        let trace_height = next_power_of_two_or_zero(num_records);
        let device_ctx = &self.range_checker.device_ctx;
        let trace =
            DeviceMatrix::<F>::with_capacity_on(trace_height, C::MAIN_CHIP_WIDTH, device_ctx);

        let d_records = records.to_device_on(device_ctx).unwrap();
        let d_record_offsets = record_offsets.to_device_on(device_ctx).unwrap();

        unsafe {
            match C::VARIANT {
                Sha2Variant::Sha256 => {
                    cuda_abi::sha256::sha256_main_tracegen(
                        trace.buffer(),
                        trace_height,
                        &d_records,
                        num_records,
                        &d_record_offsets,
                        self.pointer_max_bits,
                        &self.range_checker.count,
                        self.timestamp_max_bits,
                        device_ctx.stream.as_raw(),
                    )
                    .unwrap();
                }
                Sha2Variant::Sha512 | Sha2Variant::Sha384 => {
                    cuda_abi::sha512::sha512_main_tracegen(
                        trace.buffer(),
                        trace_height,
                        &d_records,
                        num_records,
                        &d_record_offsets,
                        self.pointer_max_bits,
                        &self.range_checker.count,
                        self.timestamp_max_bits,
                        device_ctx.stream.as_raw(),
                    )
                    .unwrap();
                }
            }
        }

        // Pass the records to Sha2BlockHasherChip
        *self.records.lock().unwrap() = Some(Sha2SharedRecordsGpu {
            d_records,
            d_record_offsets,
            num_records,
        });

        AirProvingContext::simple_no_pis(trace)
    }
}

/// Generic hybrid GPU wrapper that reuses CPU block-hasher tracegen.
pub struct Sha2BlockHasherChipGpu<C: Sha2Config> {
    records: Arc<Mutex<Option<Sha2SharedRecordsGpu>>>,
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<8>>,
    /// Range checker for digest-row `final_hash` limbs.
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    #[cfg(feature = "rvr")]
    pointer_max_bits: u32,
    _marker: PhantomData<C>,
}

impl<C, R> Chip<R, GpuBackend> for Sha2BlockHasherChipGpu<C>
where
    C: Sha2Config,
{
    /// We don't use the record arena associated with this chip. Instead, we will use the record
    /// arena provided by the main chip, which will be passed to this chip after the main chip's
    /// tracegen is done.
    fn generate_proving_ctx(&self, _: R) -> AirProvingContext<GpuBackend> {
        let mut records = self.records.lock().unwrap();
        if records.is_none() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let Sha2SharedRecordsGpu {
            d_records,
            d_record_offsets,
            num_records,
        } = records.take().unwrap();

        if num_records == 0 {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let rows_used = num_records * C::ROWS_PER_BLOCK;
        let trace_height = next_power_of_two_or_zero(rows_used);
        let device_ctx = &self.bitwise_lookup.device_ctx;
        let trace =
            DeviceMatrix::<F>::with_capacity_on(trace_height, C::BLOCK_HASHER_WIDTH, device_ctx);

        // one record per block, right now
        let num_blocks: u32 = num_records as u32;

        // prev_hashes
        unsafe {
            match C::VARIANT {
                Sha2Variant::Sha256 => {
                    let d_prev_hashes = DeviceBuffer::<u32>::with_capacity_on(
                        num_blocks as usize * C::HASH_WORDS,
                        device_ctx,
                    );
                    cuda_abi::sha256::sha256_hash_computation(
                        &d_records,
                        num_records,
                        &d_record_offsets,
                        &d_prev_hashes,
                        num_blocks,
                        device_ctx.stream.as_raw(),
                    )
                    .unwrap();

                    // Scratch for three-phase tracegen: state[8] + w_buf[BLOCK_WORDS] u32s per
                    // row per block.
                    // 17 rows * (8 + 16) * 4 bytes = 1632 bytes/block, vs
                    // 17 * 456 * 4 = 31008 bytes/block for the trace matrix (~5.3% overhead).
                    let scratch_words_per_block = C::ROWS_PER_BLOCK * (8 + C::BLOCK_WORDS);
                    let d_scratch = DeviceBuffer::<u32>::with_capacity_on(
                        num_blocks as usize * scratch_words_per_block,
                        device_ctx,
                    );

                    cuda_abi::sha256::sha256_first_pass_tracegen(
                        trace.buffer(),
                        trace_height,
                        &d_records,
                        num_records,
                        &d_record_offsets,
                        num_blocks,
                        &d_prev_hashes,
                        &self.bitwise_lookup.count,
                        &d_scratch,
                        &self.range_checker.count,
                        device_ctx.stream.as_raw(),
                    )
                    .unwrap();

                    cuda_abi::sha256::sha256_fill_invalid_rows(
                        trace.buffer(),
                        trace_height,
                        rows_used,
                        &d_prev_hashes,
                        device_ctx.stream.as_raw(),
                    )
                    .unwrap();
                    cuda_abi::sha256::sha256_second_pass_dependencies(
                        trace.buffer(),
                        trace_height,
                        rows_used,
                        device_ctx.stream.as_raw(),
                    )
                    .unwrap();
                }
                Sha2Variant::Sha512 | Sha2Variant::Sha384 => {
                    let d_prev_hashes = DeviceBuffer::<u64>::with_capacity_on(
                        num_blocks as usize * C::HASH_WORDS,
                        device_ctx,
                    );
                    cuda_abi::sha512::sha512_hash_computation(
                        &d_records,
                        num_records,
                        &d_record_offsets,
                        &d_prev_hashes,
                        num_blocks,
                        device_ctx.stream.as_raw(),
                    )
                    .unwrap();

                    // Scratch for three-phase tracegen: state[8] + w_buf[BLOCK_WORDS] u64s per
                    // row per block.
                    // 21 rows * (8 + 16) * 8 bytes = 4032 bytes/block, vs
                    // 21 * 903 * 4 = 75852 bytes/block for the trace matrix (~5.3% overhead).
                    let scratch_words_per_block = C::ROWS_PER_BLOCK * (8 + C::BLOCK_WORDS);
                    let d_scratch = DeviceBuffer::<u64>::with_capacity_on(
                        num_blocks as usize * scratch_words_per_block,
                        device_ctx,
                    );

                    cuda_abi::sha512::sha512_first_pass_tracegen(
                        trace.buffer(),
                        trace_height,
                        &d_records,
                        num_records,
                        &d_record_offsets,
                        num_blocks,
                        &d_prev_hashes,
                        &self.bitwise_lookup.count,
                        &d_scratch,
                        &self.range_checker.count,
                        device_ctx.stream.as_raw(),
                    )
                    .unwrap();

                    cuda_abi::sha512::sha512_fill_invalid_rows(
                        trace.buffer(),
                        trace_height,
                        rows_used,
                        &d_prev_hashes,
                        device_ctx.stream.as_raw(),
                    )
                    .unwrap();
                    cuda_abi::sha512::sha512_second_pass_dependencies(
                        trace.buffer(),
                        trace_height,
                        rows_used,
                        device_ctx.stream.as_raw(),
                    )
                    .unwrap();
                }
            }
        }

        AirProvingContext::simple_no_pis(trace)
    }
}

impl<C: Sha2Config> Sha2BlockHasherChipGpu<C> {
    pub fn new(
        records: Arc<Mutex<Option<Sha2SharedRecordsGpu>>>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<8>>,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        _pointer_max_bits: u32,
    ) -> Self {
        Self {
            records,
            bitwise_lookup,
            range_checker,
            #[cfg(feature = "rvr")]
            pointer_max_bits: _pointer_max_bits,
            _marker: PhantomData,
        }
    }

    #[cfg(feature = "rvr")]
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
                        RV64_REGISTER_AS,
                        RV64_MEMORY_AS,
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
                        RV64_REGISTER_AS,
                        RV64_MEMORY_AS,
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
