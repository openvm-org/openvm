use std::{marker::PhantomData, sync::Arc};

use openvm_circuit::{
    arch::{DenseRecordArena, RecordSeeker},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prelude::F, prover_backend::GpuBackend,
};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_sha2_air::{Sha256Config, Sha2Variant, Sha512Config};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use crate::{Sha2Config, Sha2RecordLayout, Sha2RecordMut};

mod cuda_abi;

pub struct Sha2MainChipGpu<C: Sha2Config> {
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<8>>,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    _marker: PhantomData<C>,
}

impl<C: Sha2Config> Sha2MainChipGpu<C> {
    pub fn new(
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<8>>,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Self {
        Self {
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
            return get_empty_air_proving_ctx::<GpuBackend>();
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
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, C::MAIN_CHIP_WIDTH);

        let d_records = records.to_device().unwrap();
        let d_record_offsets = record_offsets.to_device().unwrap();

        unsafe {
            if C::MESSAGE_LENGTH_BITS == 64 {
                cuda_abi::sha256::sha256_main_tracegen(
                    trace.buffer(),
                    trace_height,
                    &d_records,
                    num_records,
                    &d_record_offsets,
                    self.pointer_max_bits,
                    &self.range_checker.count,
                    &self.bitwise_lookup.count,
                    8,
                    self.timestamp_max_bits,
                )
                .unwrap();
            } else {
                cuda_abi::sha512::sha512_main_tracegen(
                    trace.buffer(),
                    trace_height,
                    &d_records,
                    num_records,
                    &d_record_offsets,
                    self.pointer_max_bits,
                    &self.range_checker.count,
                    &self.bitwise_lookup.count,
                    8,
                    self.timestamp_max_bits,
                )
                .unwrap();
            }
        }

        AirProvingContext::simple_no_pis(trace)
    }
}

/// Generic hybrid GPU wrapper that reuses CPU block-hasher tracegen.
pub struct Sha2BlockHasherChipGpu<C: Sha2Config> {
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<8>>,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
    _marker: PhantomData<C>,
}

impl<C> Chip<DenseRecordArena, GpuBackend> for Sha2BlockHasherChipGpu<C>
where
    C: Sha2Config,
{
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated_mut();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }

        let mut record_offsets = Vec::<usize>::new();
        let mut block_offsets = Vec::<u32>::new();
        let mut block_to_record_idx = Vec::<u32>::new();
        let mut offset = 0usize;
        let mut num_blocks = 0u32;

        while offset < records.len() {
            record_offsets.push(offset);
            block_offsets.push(num_blocks);

            let record =
                RecordSeeker::<DenseRecordArena, Sha2RecordMut, Sha2RecordLayout>::get_record_at(
                    &mut offset,
                    records,
                );
            debug_assert!((record.inner.variant as u8) == (C::VARIANT as u8));

            block_to_record_idx.push(record_offsets.len() as u32 - 1);
            num_blocks += 1;
        }

        let rows_used_blocks = num_blocks as usize * C::ROWS_PER_BLOCK;
        let rows_used_total = rows_used_blocks + 1;
        let trace_height = next_power_of_two_or_zero(rows_used_total);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, C::BLOCK_HASHER_WIDTH);

        let records_num = record_offsets.len();
        let d_records = records.to_device().unwrap();
        let d_record_offsets = record_offsets.to_device().unwrap();
        let d_block_offsets = block_offsets.to_device().unwrap();
        let d_block_to_record_idx = block_to_record_idx.to_device().unwrap();

        // prev_hashes
        unsafe {
            match C::VARIANT {
                Sha2Variant::Sha256 => {
                    let d_prev_hashes =
                        DeviceBuffer::<u32>::with_capacity(num_blocks as usize * C::HASH_WORDS);
                    cuda_abi::sha256::sha256_hash_computation(
                        &d_records,
                        records_num,
                        &d_record_offsets,
                        &d_block_offsets,
                        &d_prev_hashes,
                        num_blocks,
                    )
                    .unwrap();

                    cuda_abi::sha256::sha256_first_pass_tracegen(
                        trace.buffer(),
                        trace_height,
                        &d_records,
                        records_num,
                        &d_record_offsets,
                        &d_block_offsets,
                        &d_block_to_record_idx,
                        num_blocks,
                        &d_prev_hashes,
                        self.pointer_max_bits,
                        &self.range_checker.count,
                        &self.bitwise_lookup.count,
                        8,
                        self.timestamp_max_bits,
                    )
                    .unwrap();

                    cuda_abi::sha256::sha256_second_pass_dependencies(
                        trace.buffer(),
                        trace_height,
                        rows_used_blocks,
                    )
                    .unwrap();
                    cuda_abi::sha256::sha256_fill_invalid_rows(
                        trace.buffer(),
                        trace_height,
                        rows_used_total,
                    )
                    .unwrap();
                }
                Sha2Variant::Sha512 => {
                    let d_prev_hashes =
                        DeviceBuffer::<u64>::with_capacity(num_blocks as usize * C::HASH_WORDS);
                    cuda_abi::sha512::sha512_hash_computation(
                        &d_records,
                        records_num,
                        &d_record_offsets,
                        &d_block_offsets,
                        &d_prev_hashes,
                        num_blocks,
                    )
                    .unwrap();

                    cuda_abi::sha512::sha512_first_pass_tracegen(
                        trace.buffer(),
                        trace_height,
                        &d_records,
                        records_num,
                        &d_record_offsets,
                        &d_block_offsets,
                        &d_block_to_record_idx,
                        num_blocks,
                        &d_prev_hashes,
                        self.pointer_max_bits,
                        &self.range_checker.count,
                        &self.bitwise_lookup.count,
                        8,
                        self.timestamp_max_bits,
                    )
                    .unwrap();

                    cuda_abi::sha512::sha512_second_pass_dependencies(
                        trace.buffer(),
                        trace_height,
                        rows_used_blocks,
                    )
                    .unwrap();
                    cuda_abi::sha512::sha512_fill_invalid_rows(
                        trace.buffer(),
                        trace_height,
                        rows_used_total,
                    )
                    .unwrap();
                }
                Sha2Variant::Sha384 => unreachable!(),
            }
        }

        AirProvingContext::simple_no_pis(trace)
    }
}

impl<C: Sha2Config> Sha2BlockHasherChipGpu<C> {
    pub fn new(
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<8>>,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Self {
        Self {
            range_checker,
            bitwise_lookup,
            pointer_max_bits,
            timestamp_max_bits,
            _marker: PhantomData,
        }
    }
}

// Convenience aliases for the common BabyBear+SHA variants.
pub type Sha256VmChipGpu = Sha2MainChipGpu<Sha256Config>;
pub type Sha256BlockHasherChipGpu = Sha2BlockHasherChipGpu<Sha256Config>;
pub type Sha512VmChipGpu = Sha2MainChipGpu<Sha512Config>;
pub type Sha512BlockHasherChipGpu = Sha2BlockHasherChipGpu<Sha512Config>;
