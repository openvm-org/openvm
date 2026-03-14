use openvm_recursion_circuit::prelude::DIGEST_SIZE;
use openvm_recursion_circuit_derive::AlignedBorrow;
pub use openvm_verify_stark_host::deferral::DeferralMerkleProofs;

pub mod hook;
pub mod inner;

pub const DEF_CIRCUIT_PVS_AIR_ID: usize = 0;

pub const DEF_AGG_VERIFIER_AIR_ID: usize = 0;
pub const DEF_AGG_PVS_AIR_ID: usize = 1;

pub const DEF_HOOK_PVS_AIR_ID: usize = 0;

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy)]
pub struct DeferralCircuitPvs<F> {
    /// Commit to the input to the deferral circuit
    pub input_commit: [F; DIGEST_SIZE],
    /// Commit to the output of the deferral circuit given the input
    pub output_commit: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy)]
pub struct DeferralAggregationPvs<F> {
    /// Compression of input_commit and output_commit at the leaf layer, and
    /// the Merkle root of the aggregation subtree this proof is the root of
    /// at internal layers
    pub merkle_commit: [F; DIGEST_SIZE],
}
