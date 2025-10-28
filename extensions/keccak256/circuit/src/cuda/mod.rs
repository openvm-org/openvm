use std::{iter::repeat_n, sync::Arc};

use derive_new::new;
use openvm_circuit::arch::{DenseRecordArena, MultiRowLayout, RecordSeeker};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, utils::next_power_of_two_or_zero,
    var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prelude::F, prover_backend::GpuBackend,
};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use p3_keccak_air::NUM_ROUNDS;

use crate::{
    columns::NUM_KECCAK_VM_COLS,
    trace::{KeccakVmMetadata, KeccakVmRecordMut},
    utils::num_keccak_f,
};

mod cuda_abi;
use cuda_abi::keccak256::*;

#[derive(new)]
pub struct Keccak256ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub ptr_max_bits: u32,
    pub timestamp_max_bits: u32,
}

impl Chip<DenseRecordArena, GpuBackend> for Keccak256ChipGpu {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated_mut();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }

        let mut record_offsets = Vec::<usize>::new();
        let mut block_to_record_idx = Vec::<u32>::new();
        let mut block_offsets = Vec::<u32>::new();
        let mut offset_so_far = 0;
        let mut num_blocks_so_far = 0;
        while offset_so_far < records.len() {
            record_offsets.push(offset_so_far);
            block_offsets.push(num_blocks_so_far);

            let record = RecordSeeker::<
                DenseRecordArena,
                KeccakVmRecordMut,
                MultiRowLayout<KeccakVmMetadata>,
            >::get_record_at(&mut offset_so_far, records);

            let num_blocks = num_keccak_f(record.inner.len as usize);
            let record_idx = record_offsets.len() - 1;
            block_to_record_idx.extend(repeat_n(record_idx as u32, num_blocks));
            num_blocks_so_far += num_blocks as u32;
        }
        assert_eq!(num_blocks_so_far as usize, block_to_record_idx.len());
        assert_eq!(offset_so_far, records.len());
        assert_eq!(block_offsets.len(), record_offsets.len());

        let records_num = record_offsets.len();
        let d_records = records.to_device().unwrap();
        let d_record_offsets = record_offsets.to_device().unwrap();
        let d_block_offsets = block_offsets.to_device().unwrap();
        let d_block_to_record_idx = block_to_record_idx.to_device().unwrap();

        let rows_used = num_blocks_so_far as usize * NUM_ROUNDS;
        let trace_height = next_power_of_two_or_zero(rows_used);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, NUM_KECCAK_VM_COLS);

        // We store state + keccakf(state) for each block
        let states_num = 2 * num_blocks_so_far as usize;
        let d_states = DeviceBuffer::<u64>::with_capacity(states_num * 25);

        unsafe {
            keccakf(
                &d_records,
                records_num,
                &d_record_offsets,
                &d_block_offsets,
                num_blocks_so_far,
                &d_states,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
            )
            .unwrap();

            p3_tracegen(trace.buffer(), trace_height, num_blocks_so_far, &d_states).unwrap();

            tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                records_num,
                &d_record_offsets,
                &d_block_offsets,
                &d_block_to_record_idx,
                num_blocks_so_far,
                &d_states,
                rows_used,
                self.ptr_max_bits,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.timestamp_max_bits,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(trace)
    }
}
