use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use recursion_circuit::define_typed_per_proof_permutation_bus;
use stark_recursion_circuit_derive::AlignedBorrow;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct InputOrMerkleCommitMessage<T> {
    pub has_verifier_pvs: T,
    pub commit: [T; DIGEST_SIZE],
}

define_typed_per_proof_permutation_bus!(InputOrMerkleCommitBus, InputOrMerkleCommitMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct DefPvsConsistencyMessage<T> {
    pub has_verifier_pvs: T,
}

define_typed_per_proof_permutation_bus!(DefPvsConsistencyBus, DefPvsConsistencyMessage);
