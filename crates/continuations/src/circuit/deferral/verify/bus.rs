use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use recursion_circuit::define_typed_permutation_bus;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::circuit::deferral::verify::output::VALS_IN_DIGEST;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct OutputValMessage<T> {
    pub values: [T; VALS_IN_DIGEST],
    pub idx: T,
}

define_typed_permutation_bus!(OutputValBus, OutputValMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct OutputCommitMessage<T> {
    pub commit: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(OutputCommitBus, OutputCommitMessage);
