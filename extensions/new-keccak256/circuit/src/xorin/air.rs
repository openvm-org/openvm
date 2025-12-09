use openvm_circuit::arch::ExecutionBridge;
use openvm_circuit::system::memory::offline_checker::MemoryBridge;
use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupBus;

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct XorinVmAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    /// Bus to send 8-bit XOR requests to.
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    /// Maximum number of bits allowed for an address pointer
    pub ptr_max_bits: usize,
    pub(super) offset: usize,
}