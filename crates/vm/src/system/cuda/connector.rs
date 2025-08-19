use openvm_circuit::system::connector::VmConnectorChip;
use openvm_cuda_backend::{chip::HybridChip, prelude::F};

pub type VmConnectorChipGPU = HybridChip<(), VmConnectorChip<F>>;
