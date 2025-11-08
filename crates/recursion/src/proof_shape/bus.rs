use stark_recursion_circuit_derive::AlignedBorrow;

use crate::define_typed_per_proof_permutation_bus;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ProofShapePermutationMessage<T> {
    pub idx: T,
}

define_typed_per_proof_permutation_bus!(ProofShapePermutationBus, ProofShapePermutationMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StartingTidxMessage<T> {
    pub air_idx: T,
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(StartingTidxBus, StartingTidxMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NumPublicValuesMessage<T> {
    pub air_idx: T,
    pub tidx: T,
    pub num_pvs: T,
}

define_typed_per_proof_permutation_bus!(NumPublicValuesBus, NumPublicValuesMessage);
