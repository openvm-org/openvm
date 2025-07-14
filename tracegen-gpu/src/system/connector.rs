use openvm_circuit::{
    arch::{ExecutionBus, ExecutionState},
    system::{connector::VmConnectorChip, program::ProgramBus},
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_stark_backend::{AirRef, Chip, ChipUsageGetter};
use stark_backend_gpu::{
    base::DeviceMatrix,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::DeviceChip;

// TODO[stephen]: We should consider creating a generic Chip<_, GpuBackend>
// implementation for all impls of Chip<_, CpuBackend>.
pub struct VmConnectorChipGpu(VmConnectorChip<F>);

impl VmConnectorChipGpu {
    // TODO[stephen]: VmConnectorChip constructor changed in OpenVM feat/new-exec-device
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        range_checker: SharedVariableRangeCheckerChip,
        timestamp_max_bits: usize,
    ) -> Self {
        Self(VmConnectorChip::new(
            execution_bus,
            program_bus,
            range_checker,
            timestamp_max_bits,
        ))
    }

    pub fn begin(&mut self, state: ExecutionState<u32>) {
        self.0.begin(state);
    }

    pub fn end(&mut self, state: ExecutionState<u32>, exit_code: Option<u32>) {
        self.0.end(state, exit_code);
    }
}

impl ChipUsageGetter for VmConnectorChipGpu {
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

impl DeviceChip<SC, GpuBackend> for VmConnectorChipGpu {
    fn air(&self) -> AirRef<SC> {
        self.0.air()
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        // TODO[stephen]: Because this trace is only 2 rows, this should just be a
        // wrapper for the CPU version of this chip. Our DeviceChip trait currently
        // uses &self instead of self though, so we'll leave this unimplemented for
        // now and fix this up with the introduction of feat/new-exec-device.
        todo!()
    }
}
