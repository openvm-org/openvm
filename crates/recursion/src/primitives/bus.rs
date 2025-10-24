use stark_recursion_circuit_derive::AlignedBorrow;

use crate::define_typed_lookup_bus;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct RangeCheckerBusMessage<T> {
    pub value: T,
    pub max_bits: T,
}

define_typed_lookup_bus!(RangeCheckerBus, RangeCheckerBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct PowerCheckerBusMessage<T> {
    pub log: T,
    pub exp: T,
}

define_typed_lookup_bus!(PowerCheckerBus, PowerCheckerBusMessage);
