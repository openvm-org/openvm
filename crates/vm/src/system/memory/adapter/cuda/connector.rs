use openvm_circuit::{arch::ExecutionState, system::connector::VmConnectorChip};

use crate::utils::HybridChip;

pub type VmConnectorChipGPU = HybridChip<(), VmConnectorChip<F>>;

impl VmConnectorChipGPU {
    pub fn begin(&mut self, state: ExecutionState<u32>) {
        self.cpu_chip.begin(state);
    }

    pub fn end(&mut self, state: ExecutionState<u32>, exit_code: Option<u32>) {
        self.cpu_chip.end(state, exit_code);
    }
}
