use recursion_circuit::define_typed_permutation_bus;
use stark_backend_v2::DIGEST_SIZE;
use stark_recursion_circuit_derive::AlignedBorrow;

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
