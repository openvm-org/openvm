use openvm_recursion_circuit::define_typed_permutation_bus;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct UserPvsCommitMessage<T> {
    pub user_pvs_commit: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(UserPvsCommitBus, UserPvsCommitMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct UserPvsCommitTreeMessage<T> {
    pub child_value: [T; DIGEST_SIZE],
    pub is_right_child: T,
    pub parent_idx: T,
}

define_typed_permutation_bus!(UserPvsCommitTreeBus, UserPvsCommitTreeMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MemoryMerkleCommitMessage<T> {
    pub merkle_root: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(MemoryMerkleCommitBus, MemoryMerkleCommitMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct DeferralAccPathMessage<T> {
    pub initial_acc_hash: [T; DIGEST_SIZE],
    pub final_acc_hash: [T; DIGEST_SIZE],
    pub depth: T,
    pub is_unset: T,
}

define_typed_permutation_bus!(DeferralAccPathBus, DeferralAccPathMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct DeferralMerkleRootsMessage<T> {
    pub initial_root: [T; DIGEST_SIZE],
    pub final_root: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(DeferralMerkleRootsBus, DeferralMerkleRootsMessage);
