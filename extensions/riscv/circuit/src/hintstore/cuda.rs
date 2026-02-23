use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{
    arch::{DenseRecordArena, RecordSeeker},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use crate::{
    cuda_abi::hintstore_cuda::tracegen, Rv32HintStoreCols, Rv32HintStoreLayout,
    Rv32HintStoreRecordMut,
};

#[derive(new)]
pub struct Rv32HintStoreChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

// This is the info needed by each row to do parallel tracegen
#[repr(C)]
#[derive(new)]
pub struct OffsetInfo {
    pub record_offset: u32,
    pub local_idx: u32,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32HintStoreChipGpu {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let width = Rv32HintStoreCols::<u8>::width();
        let records = arena.allocated_mut();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }

        let mut offsets = Vec::<OffsetInfo>::new();
        let mut offset = 0;

        while offset < records.len() {
            let prev_offset = offset;
            let record = RecordSeeker::<
                DenseRecordArena,
                Rv32HintStoreRecordMut,
                Rv32HintStoreLayout,
            >::get_record_at(&mut offset, records);
            for idx in 0..record.inner.num_words {
                offsets.push(OffsetInfo::new(prev_offset as u32, idx));
            }
        }

        let d_records = records.to_device().unwrap();
        let d_record_offsets = offsets.to_device().unwrap();

        let trace_height = next_power_of_two_or_zero(offsets.len());
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, width);

        unsafe {
            tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                offsets.len(),
                &d_record_offsets,
                self.pointer_max_bits as u32,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
