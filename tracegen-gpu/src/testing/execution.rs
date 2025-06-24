use std::{slice::from_raw_parts, sync::Arc};

use openvm_circuit::arch::{
    testing::{
        execution::air::{DummyExecutionInteractionCols, ExecutionDummyAir},
        ExecutionTester,
    },
    ExecutionBus, ExecutionState,
};
use openvm_stark_backend::{AirRef, ChipUsageGetter};
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::copy::MemCopyH2D,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{testing::cuda::execution_testing, DeviceChip};

pub struct DeviceExecutionTester(ExecutionTester<F>);

impl DeviceExecutionTester {
    pub fn new(bus: ExecutionBus) -> Self {
        Self(ExecutionTester::new(bus))
    }

    pub fn bus(&self) -> ExecutionBus {
        self.0.bus
    }

    pub fn execute(
        &mut self,
        initial_state: ExecutionState<u32>,
        final_state: ExecutionState<u32>,
    ) {
        self.0.execute(initial_state, final_state);
    }
}

impl ChipUsageGetter for DeviceExecutionTester {
    fn air_name(&self) -> String {
        self.0.air_name()
    }

    fn current_trace_height(&self) -> usize {
        self.0.current_trace_height()
    }

    fn trace_width(&self) -> usize {
        self.0.trace_width()
    }
}

impl DeviceChip<SC, GpuBackend> for DeviceExecutionTester {
    fn air(&self) -> AirRef<SC> {
        Arc::new(ExecutionDummyAir::new(self.0.bus))
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let height = self.0.current_trace_height().next_power_of_two();
        let width = self.0.trace_width();
        let trace = DeviceMatrix::<F>::with_capacity(height, width);

        let records = &self.0.records;
        let num_records = records.len();

        unsafe {
            let bytes_size = num_records * size_of::<DummyExecutionInteractionCols<F>>();
            let records_bytes = from_raw_parts(records.as_ptr() as *const u8, bytes_size);
            let records = records_bytes.to_device().unwrap();
            execution_testing::tracegen(trace.buffer(), height, width, &records, num_records)
                .unwrap();
        }
        trace
    }
}
