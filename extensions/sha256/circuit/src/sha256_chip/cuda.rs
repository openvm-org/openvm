// crates/tracegen/src/extensions/sha256/mod.rs

use std::{iter::repeat_n, sync::Arc};

use derive_new::new;
use openvm_circuit::{
    arch::{DenseRecordArena, MultiRowLayout, RecordSeeker},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prelude::F, prover_backend::GpuBackend,
};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_sha256_air::{get_sha256_num_blocks, SHA256_HASH_WORDS, SHA256_ROWS_PER_BLOCK};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use crate::{
    cuda_abi::sha256::{
        sha256_fill_invalid_rows, sha256_first_pass_tracegen, sha256_hash_computation,
        sha256_second_pass_dependencies,
    },
    Sha256VmMetadata, Sha256VmRecordMut, SHA256VM_WIDTH,
};

// ===== SHA256 GPU CHIP IMPLEMENTATION =====
#[derive(new)]
pub struct Sha256VmChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub ptr_max_bits: u32,
    pub timestamp_max_bits: u32,
}

impl Chip<DenseRecordArena, GpuBackend> for Sha256VmChipGpu {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated_mut();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }

        let mut record_offsets = Vec::<usize>::new();
        let mut block_to_record_idx = Vec::<u32>::new();
        let mut block_offsets = Vec::<u32>::new();
        let mut offset_so_far = 0;
        let mut num_blocks_so_far: u32 = 0;

        while offset_so_far < records.len() {
            record_offsets.push(offset_so_far);
            block_offsets.push(num_blocks_so_far);

            let record = RecordSeeker::<
                DenseRecordArena,
                Sha256VmRecordMut,
                MultiRowLayout<Sha256VmMetadata>,
            >::get_record_at(&mut offset_so_far, records);

            let num_blocks = get_sha256_num_blocks(record.inner.len);
            let record_idx = record_offsets.len() - 1;

            block_to_record_idx.extend(repeat_n(record_idx as u32, num_blocks as usize));
            num_blocks_so_far += num_blocks;
        }

        assert_eq!(num_blocks_so_far as usize, block_to_record_idx.len());
        assert_eq!(offset_so_far, records.len());
        assert_eq!(block_offsets.len(), record_offsets.len());

        let d_records = records.to_device().unwrap();
        let d_record_offsets = record_offsets.to_device().unwrap();
        let d_block_offsets = block_offsets.to_device().unwrap();
        let d_block_to_record_idx = block_to_record_idx.to_device().unwrap();

        let d_prev_hashes = DeviceBuffer::<u32>::with_capacity(
            num_blocks_so_far as usize * SHA256_HASH_WORDS, // 8 words per SHA256 hash block
        );

        unsafe {
            sha256_hash_computation(
                &d_records,
                record_offsets.len(),
                &d_record_offsets,
                &d_block_offsets,
                &d_prev_hashes,
                num_blocks_so_far,
            )
            .expect("Hash computation kernel failed");
        }

        let rows_used = num_blocks_so_far as usize * SHA256_ROWS_PER_BLOCK;
        let trace_height = next_power_of_two_or_zero(rows_used);
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, SHA256VM_WIDTH);

        unsafe {
            sha256_first_pass_tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                record_offsets.len(),
                &d_record_offsets,
                &d_block_offsets,
                &d_block_to_record_idx,
                num_blocks_so_far,
                &d_prev_hashes,
                self.ptr_max_bits,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS as u32,
                self.timestamp_max_bits,
            )
            .expect("First pass trace generation failed");
        }

        unsafe {
            sha256_fill_invalid_rows(d_trace.buffer(), trace_height, rows_used)
                .expect("Invalid rows filling failed");
        }

        unsafe {
            sha256_second_pass_dependencies(d_trace.buffer(), trace_height, rows_used)
                .expect("Second pass trace generation failed");
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
