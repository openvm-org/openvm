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
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<8>>,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    _marker: PhantomData<C>,
}

impl<C: Sha2Config> Sha2MainChipGpu<C> {
    pub fn new(
        records: Arc<Mutex<Option<Sha2SharedRecordsGpu>>>,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<8>>,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Self {
        Self {
            records,
            range_checker,
            bitwise_lookup,
            pointer_max_bits,
            timestamp_max_bits,
            _marker: PhantomData,
        }
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
                        &self.bitwise_lookup.count,
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
                        &self.bitwise_lookup.count,
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
    ) -> Self {
        Self {
            records,
            bitwise_lookup,
            _marker: PhantomData,
        }
    }
}

// Convenience aliases for the common SHA-2 variants.
pub type Sha256VmChipGpu = Sha2MainChipGpu<Sha256Config>;
pub type Sha256BlockHasherChipGpu = Sha2BlockHasherChipGpu<Sha256Config>;
pub type Sha512VmChipGpu = Sha2MainChipGpu<Sha512Config>;
pub type Sha512BlockHasherChipGpu = Sha2BlockHasherChipGpu<Sha512Config>;
