use recursion_circuit::define_typed_per_proof_lookup_bus;
use stark_recursion_circuit_derive::AlignedBorrow;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct PvsAirConsistencyMessage<T> {
    pub deferral_flag: T,
    pub has_verifier_pvs: T,
}

define_typed_per_proof_lookup_bus!(PvsAirConsistencyBus, PvsAirConsistencyMessage);
