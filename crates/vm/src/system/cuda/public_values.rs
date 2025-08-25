use std::{mem::size_of, sync::Arc};

use openvm_circuit::{
    arch::DenseRecordArena,
    system::{
        native_adapter::{NativeAdapterCols, NativeAdapterRecord},
        public_values::PublicValuesRecord,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{encoder::Encoder, var_range::VariableRangeCheckerChipGPU};
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prelude::F, prover_backend::GpuBackend,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{
    prover::{hal::MatrixDimensions, types::AirProvingContext},
    Chip,
};

use crate::cuda_abi::public_values;

#[repr(C)]
struct FullPublicValuesRecord {
    #[allow(unused)]
    adapter: NativeAdapterRecord<F, 2, 0>,
    #[allow(unused)]
    core: PublicValuesRecord<F>,
}

pub struct PublicValuesChipGPU {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub public_values: Vec<F>,
    pub num_custom_pvs: usize,
    pub max_degree: u32,
    // needed to compute the width of the trace
    encoder: Encoder,
    pub timestamp_max_bits: u32,
}

impl PublicValuesChipGPU {
    pub fn new(
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        num_custom_pvs: usize,
        max_degree: u32,
        timestamp_max_bits: u32,
    ) -> Self {
        Self {
            range_checker,
            public_values: Vec::new(),
            num_custom_pvs,
            max_degree,
            encoder: Encoder::new(num_custom_pvs, max_degree, true),
            timestamp_max_bits,
        }
    }
}

impl PublicValuesChipGPU {
    pub fn trace_height(arena: &DenseRecordArena) -> usize {
        let record_size = size_of::<FullPublicValuesRecord>();
        let records_len = arena.allocated().len();
        assert_eq!(records_len % record_size, 0);
        records_len / record_size
    }

    pub fn trace_width(&self) -> usize {
        NativeAdapterCols::<u8, 2, 0>::width() + 3 + self.encoder.width()
    }
}

impl Chip<DenseRecordArena, GpuBackend> for PublicValuesChipGPU {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let num_records = Self::trace_height(&arena);
        if num_records == 0 {
            return get_empty_air_proving_ctx();
        }
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            public_values::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &arena.allocated().to_device().unwrap(),
                &self.range_checker.count,
                self.timestamp_max_bits,
                self.num_custom_pvs,
                self.max_degree,
            )
            .expect("Failed to generate trace");
        }
        AirProvingContext::simple(trace, self.public_values.clone())
    }
}
