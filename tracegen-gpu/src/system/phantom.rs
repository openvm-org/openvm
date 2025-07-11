use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{
    arch::DenseRecordArena, system::phantom::PhantomAir, utils::next_power_of_two_or_zero,
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

use crate::{system::cuda, DeviceChip};

// TODO[stephen]: remove these definitions, already defined in feat/new-exec-device
const NUM_PHANTOM_OPERANDS: usize = 3;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct PhantomRecord {
    pub pc: u32,
    pub operands: [u32; NUM_PHANTOM_OPERANDS],
    pub timestamp: u32,
}

#[derive(new)]
pub struct PhantomChipGpu {
    pub air: PhantomAir,
    pub arena: DenseRecordArena,
}

impl ChipUsageGetter for PhantomChipGpu {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        let record_size = size_of::<PhantomRecord>();
        let records_len = self.arena.allocated().len();
        assert_eq!(records_len % record_size, 0);
        records_len / record_size
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for PhantomChipGpu {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let num_records = self.current_trace_height();
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            cuda::phantom::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &self.arena.allocated().to_device().unwrap(),
                num_records,
            )
            .expect("Failed to generate trace");
        }
        trace
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_phantom_tracegen() {
        // TODO[stephen]: test PhantomChipGpu tracegen after feat/new-exec-device
        println!("Skipping test_phantom_tracegen...");
    }
}
