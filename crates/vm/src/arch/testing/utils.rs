use openvm_circuit::{
    arch::testing::{BITWISE_OP_LOOKUP_BUS, RANGE_CHECKER_BUS},
    system::memory::SharedMemoryHelper,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupBus, BitwiseOperationLookupChip, SharedBitwiseOperationLookupChip,
    },
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip, SharedRangeTupleCheckerChip},
    var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerBus, VariableRangeCheckerChip,
    },
};
use openvm_stark_backend::p3_field::{Field, PrimeField32};

use crate::{
    arch::{hasher::poseidon2::vm_poseidon2_hasher, MemoryConfig, VmState},
    system::memory::{dimensions::MemoryDimensions, merkle::MerkleTree, online::TracingMemory},
};

pub fn default_var_range_checker_bus() -> VariableRangeCheckerBus {
    // setting default range_max_bits to 17 because that's the default decomp value in MemoryConfig
    VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, 17)
}

pub fn default_bitwise_lookup_bus() -> BitwiseOperationLookupBus {
    BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS)
}

pub fn dummy_range_checker(bus: VariableRangeCheckerBus) -> SharedVariableRangeCheckerChip {
    SharedVariableRangeCheckerChip::new(VariableRangeCheckerChip::new(bus))
}

pub fn dummy_bitwise_op_lookup(
    bus: BitwiseOperationLookupBus,
) -> SharedBitwiseOperationLookupChip<8> {
    SharedBitwiseOperationLookupChip::new(BitwiseOperationLookupChip::new(bus))
}

pub fn dummy_range_tuple_checker(bus: RangeTupleCheckerBus<2>) -> SharedRangeTupleCheckerChip<2> {
    SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::new(bus))
}

pub fn dummy_memory_helper<F: Field>(
    bus: VariableRangeCheckerBus,
    timestamp_max_bits: usize,
) -> SharedMemoryHelper<F> {
    SharedMemoryHelper::new(dummy_range_checker(bus), timestamp_max_bits)
}

pub fn default_tracing_memory(mem_config: &MemoryConfig) -> TracingMemory {
    TracingMemory::new(mem_config)
}

/// Asserts that two final [`VmState`]s reached via different execution paths (e.g. interpreter
/// vs AOT) are equivalent: same pc and same guest memory (compared via Merkle roots).
pub fn assert_vm_states_equivalent<F: PrimeField32>(
    state1: &VmState<F>,
    state2: &VmState<F>,
    memory_dimensions: &MemoryDimensions,
) {
    assert_eq!(state1.pc(), state2.pc(), "PCs differ");
    let hasher = vm_poseidon2_hasher::<F>();
    let root1 = MerkleTree::from_memory(&state1.memory.memory, memory_dimensions, &hasher).root();
    let root2 = MerkleTree::from_memory(&state2.memory.memory, memory_dimensions, &hasher).root();
    assert_eq!(root1, root2, "Memory states differ");
}
