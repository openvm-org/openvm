use openvm_recursion_circuit::define_typed_permutation_bus;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MerkleRootMessage<T> {
    pub merkle_root: [T; DIGEST_SIZE],
    pub idx: T,
}

define_typed_permutation_bus!(MerkleRootBus, MerkleRootMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MerkleTreeInternalMessage<T> {
    pub child_value: [T; DIGEST_SIZE],
    pub is_right_child: T,
    pub parent_idx: T,
}

define_typed_permutation_bus!(MerkleTreeInternalBus, MerkleTreeInternalMessage);
