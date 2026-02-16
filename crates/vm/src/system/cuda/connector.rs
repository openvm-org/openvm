use openvm_circuit::{primitives::hybrid_chip::HybridChip, system::connector::VmConnectorChip};
use openvm_cuda_backend::prelude::F;

pub type VmConnectorChipGPU = HybridChip<(), VmConnectorChip<F>>;
