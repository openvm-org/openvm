use openvm_circuit::{primitives::hybrid_chip::HybridChip, system::connector::VmConnectorChip};
pub type VmConnectorChipGPU = HybridChip<(), VmConnectorChip>;
