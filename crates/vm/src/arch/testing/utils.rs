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
use openvm_stark_backend::p3_field::Field;

use crate::{
    arch::{MemoryConfig, VmState},
    system::memory::online::{LinearMemory, TracingMemory},
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
/// vs AOT) have the same pc and guest memory.
pub fn assert_vm_states_equivalent(state1: &VmState, state2: &VmState) {
    assert_eq!(state1.pc(), state2.pc(), "PCs differ");

    let memory1 = &state1.memory.memory;
    let memory2 = &state2.memory.memory;
    assert_eq!(
        memory1.config.len(),
        memory1.mem.len(),
        "First memory state has inconsistent address-space metadata"
    );
    assert_eq!(
        memory2.config.len(),
        memory2.mem.len(),
        "Second memory state has inconsistent address-space metadata"
    );
    assert_eq!(
        memory1.config.len(),
        memory2.config.len(),
        "Memory address-space counts differ"
    );

    for (addr_space, ((config1, data1), (config2, data2))) in memory1
        .config
        .iter()
        .zip(&memory1.mem)
        .zip(memory2.config.iter().zip(&memory2.mem))
        .enumerate()
    {
        assert_eq!(
            config1.num_cells, config2.num_cells,
            "Memory sizes differ in address space {addr_space}"
        );
        assert_eq!(
            config1.layout, config2.layout,
            "Memory layouts differ in address space {addr_space}"
        );

        let bytes1 = data1.as_slice();
        let bytes2 = data2.as_slice();
        assert_eq!(
            bytes1.len(),
            bytes2.len(),
            "Memory byte lengths differ in address space {addr_space}"
        );
        if bytes1 != bytes2 {
            let offset = bytes1.iter().zip(bytes2).position(|(a, b)| a != b).unwrap();
            panic!(
                "Memory states differ at address space {addr_space}, byte offset {offset}: {} != {}",
                bytes1[offset], bytes2[offset]
            );
        }
    }
}
