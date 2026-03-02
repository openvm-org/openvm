use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use recursion_circuit::define_typed_permutation_bus;
use stark_recursion_circuit_derive::AlignedBorrow;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct IoCommitMessage<T> {
    pub idx: T,
    pub input_commit: [T; DIGEST_SIZE],
    pub output_commit: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(IoCommitBus, IoCommitMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct OnionResultMessage<T> {
    pub input_onion: [T; DIGEST_SIZE],
    pub output_onion: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(OnionResultBus, OnionResultMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct DefVkCommitMessage<T> {
    pub def_vk_commit: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(DefVkCommitBus, DefVkCommitMessage);
