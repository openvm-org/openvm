use recursion_circuit::prelude::DIGEST_SIZE;
use stark_recursion_circuit_derive::AlignedBorrow;

pub mod aggregation;
pub mod verify;

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

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy)]
pub struct DeferralVerifierPvs<F> {
    /// Ternary flag to indicate which continuations layer this Proof is for. Should be 0 for
    /// the leaf verifier, 1 for the internal-for-leaf verifier, and 2 for the internal-
    /// recursive verifier.
    pub internal_flag: F,
    /// Cached trace commit of the leaf verifier circuit's SymbolicExpressionAir, which is
    /// derived from the def_vk
    pub def_dag_commit: [F; DIGEST_SIZE],
    /// Cached trace commit of the internal-for-leaf verifier circuit's SymbolicExpressionAir,
    /// which is derived from the leaf_vk
    pub leaf_dag_commit: [F; DIGEST_SIZE],
    /// Cached trace commit of the first (i.e. index 0) internal-recursive layer verifier
    /// circuit's SymbolicExpressionAir, which is derived from the internal_for_leaf_vk
    pub internal_for_leaf_dag_commit: [F; DIGEST_SIZE],

    /// Ternary flag to indicate which internal-recursive layer this Proof is for. Should be
    /// 1 for the first (i.e. index 0) internal-recursive layer, 2 for subsequent layers, and
    /// 0 everywhere else.
    pub recursion_flag: F,
    /// Cached trace commit of each subsequent (i.e. index > 0) internal-recursive layer
    /// verifier's SymbolicExpressionAir, which is derived from the internal_recursive_vk
    pub internal_recursive_dag_commit: [F; DIGEST_SIZE],
}
