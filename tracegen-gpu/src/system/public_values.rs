use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{
    arch::DenseRecordArena,
    system::{
        native_adapter::NativeAdapterRecord,
        public_values::{core::PublicValuesRecord, PublicValuesAir},
    },
    utils::next_power_of_two_or_zero,
};
use openvm_stark_backend::{
    prover::hal::MatrixDimensions, rap::get_air_name, AirRef, ChipUsageGetter,
};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::copy::MemCopyH2D,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{primitives::var_range::VariableRangeCheckerChipGPU, system::cuda, DeviceChip};

#[repr(C)]
struct FullPublicValuesRecord {
    #[allow(unused)]
    adapter: NativeAdapterRecord<F, 2, 0>,
    #[allow(unused)]
    core: PublicValuesRecord<F>,
}

#[derive(new)]
pub struct PublicValuesChipGpu {
    pub air: PublicValuesAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub arena: DenseRecordArena,
    pub num_custom_pvs: usize,
    pub max_degree: u32,
}

impl ChipUsageGetter for PublicValuesChipGpu {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        let record_size = size_of::<FullPublicValuesRecord>();
        let records_len = self.arena.allocated().len();
        assert_eq!(records_len % record_size, 0);
        records_len / record_size
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for PublicValuesChipGpu {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let num_records = self.current_trace_height();
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            cuda::public_values::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &self.arena.allocated().to_device().unwrap(),
                num_records,
                &self.range_checker.count,
                self.num_custom_pvs,
                self.max_degree,
            )
            .expect("Failed to generate trace");
        }
        trace
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_public_values_tracegen() {
        // TODO[stephen]: test PublicValuesChipGpu tracegen after feat/new-exec-device
        println!("Skipping test_public_values_tracegen...");
    }
}
