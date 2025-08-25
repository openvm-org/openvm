use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use crate::{
    adapters::{Rv32LoadStoreAdapterCols, Rv32LoadStoreAdapterRecord},
    cuda_abi::loadstore_cuda::tracegen,
    LoadStoreCoreCols, LoadStoreCoreRecord,
};

#[derive(new)]
pub struct Rv32LoadStoreChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32LoadStoreChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32LoadStoreAdapterRecord,
            LoadStoreCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = Rv32LoadStoreAdapterCols::<F>::width()
            + LoadStoreCoreCols::<F, RV32_REGISTER_NUM_LIMBS>::width();
        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(padded_height, trace_width);

        unsafe {
            tracegen(
                d_trace.buffer(),
                padded_height,
                trace_width,
                &d_records,
                self.pointer_max_bits,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
