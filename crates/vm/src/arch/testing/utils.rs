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
use openvm_stark_backend::{p3_field::Field, p3_util::log2_strict_usize};

use crate::{
    arch::MemoryConfig,
    system::memory::{adapter::records::arena_size_bound, online::TracingMemory},
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

pub fn default_tracing_memory(mem_config: &MemoryConfig, init_block_size: usize) -> TracingMemory {
    let max_access_adapter_n = log2_strict_usize(mem_config.max_access_adapter_n);
    let arena_size_bound = arena_size_bound(&vec![1 << 16; max_access_adapter_n]);
    TracingMemory::new(mem_config, init_block_size, arena_size_bound)
}
