use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_instructions::riscv::RV64_CELL_BITS;
use openvm_stark_backend::prover::AirProvingContext;

use super::{
    DeferralCallAdapterCols, DeferralCallAdapterRecord, DeferralCallCoreCols,
    DeferralCallCoreRecord,
};
use crate::{cuda_abi::call, poseidon2::DeferralPoseidon2SharedBuffer};

#[derive(new)]
pub struct DeferralCallChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_CELL_BITS>>,
    pub address_bits: usize,
    pub timestamp_max_bits: usize,
    pub count: Arc<DeviceBuffer<u32>>,
    pub num_deferral_circuits: usize,
    pub poseidon2: DeferralPoseidon2SharedBuffer,
}

impl Chip<DenseRecordArena, GpuBackend> for DeferralCallChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        type Record = (DeferralCallAdapterRecord<F>, DeferralCallCoreRecord<F>);
        const RECORD_SIZE: usize = size_of::<Record>();

        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let num_records = records.len() / RECORD_SIZE;
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace_width =
            DeferralCallAdapterCols::<F>::width() + DeferralCallCoreCols::<F>::width();
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = records.to_device_on(device_ctx).unwrap();
        let trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            call::tracegen(
                trace.buffer(),
                trace_height,
                trace_width,
                &d_records,
                num_records,
                &self.count,
                self.num_deferral_circuits,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                &self.bitwise_lookup.count,
                &self.poseidon2.records,
                &self.poseidon2.counts,
                &self.poseidon2.idx,
                // Length in F elements; the CUDA side converts to record count.
                self.poseidon2.records.len(),
                self.address_bits,
                device_ctx.stream.as_raw(),
            )
            .expect("Failed to generate deferral call trace");
        }

        AirProvingContext::simple_no_pis(trace)
    }
}
