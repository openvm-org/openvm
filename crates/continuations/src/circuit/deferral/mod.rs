use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::prelude::DIGEST_SIZE;
use openvm_recursion_circuit_derive::AlignedBorrow;
pub use openvm_verify_stark_host::deferral::DeferralMerkleProofs;

pub mod hook;
pub mod inner;

pub const DEF_CIRCUIT_PVS_AIR_ID: usize = 0;

pub const DEF_AGG_VERIFIER_AIR_ID: usize = 0;
pub const DEF_AGG_PVS_AIR_ID: usize = 1;

pub const DEF_HOOK_PVS_AIR_ID: usize = 0;

// Set to the default max trace log height of the deferral hook circuit
const MAX_DEF_AGG_MERKLE_DEPTH: usize = 20;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy)]
pub struct DeferralCircuitPvs<F> {
    /// Commit to the input to the deferral circuit
    pub input_commit: [F; DIGEST_SIZE],
    /// Commit to the output of the deferral circuit given the input
    pub output_commit: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy)]
pub struct DeferralAggregationPvs<F> {
    /// Compression of input_commit and output_commit at the leaf layer, and
    /// the Merkle root of the aggregation subtree this proof is the root of
    /// at internal layers
    pub merkle_commit: [F; DIGEST_SIZE],
    /// Number of present deferral circuit proofs that we've seen so far
    pub num_def_circuit_proofs: F,
    /// Current Merkle depth of this subtree, there are be 2^merkle_depth
    /// leaves (including padding)
    pub merkle_depth: F,
}
