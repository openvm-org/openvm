use openvm_circuit_primitives::AlignedBorrow;
use recursion_circuit::prelude::DIGEST_SIZE;

pub mod aggregation;
pub mod verify;

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct DeferralCircuitPvs<F> {
    /// Commit to the input to the deferral circuit
    pub input_commit: [F; DIGEST_SIZE],
    /// Commit to the output of the deferral circuit given the input
    pub output_commit: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct DeferralVerifierPvs<F> {
    /// Boolean that indicates if all intra-circuit Proofs have been aggregated.
    pub intra_flag: F,
    /// Boolean that indicates if all inter-circuit Proofs have been aggregated. Should only
    /// be 1 at the root.
    pub inter_flag: F,
    /// Depth of the inter-circuit subtree with this node as root.
    pub inter_depth: F,

    /// Merkle root of circuit input commits in this subtree if deferral_flag is 0, else is
    /// the Merkle root of the input/output accumulator memory subtree rooted at this node.
    pub input_commit: [F; DIGEST_SIZE],
    /// Merkle root of circuit output commits in this subtree if deferral_flag is 0, else is
    /// unset.
    pub output_commit: [F; DIGEST_SIZE],

    /// Ternary flag to indicate which continuations layer this Proof is for. Should be 0 for
    /// the leaf verifier, 1 for the internal-for-leaf verifier, and 2 for the internal-
    /// recursive verifier.
    pub internal_flag: F,
    /// When deferral_flag is 0 this is the cached trace commit of the leaf verifier circuit's
    /// SymbolicExpressionAir, which is derived from the def_vk. Else this is the def_vk_commit,
    /// computed by compressing together the def, leaf, and internal-for-leaf DAG commits.
    pub def_commit: [F; DIGEST_SIZE],
    /// When deferral_flag is 0 this is the cached trace commit of the internal-for-leaf
    /// verifier circuit's SymbolicExpressionAir, which is derived from the leaf_vk. Should be
    /// unset otherwise.
    pub leaf_dag_commit: [F; DIGEST_SIZE],
    /// When deferral_flag is 0 this is the cached trace commit of the first (i.e. index 0)
    /// internal-recursive layer verifier circuit's SymbolicExpressionAir, which is derived
    /// from the internal_for_leaf_vk. Should be unset otherwise.
    pub internal_for_leaf_dag_commit: [F; DIGEST_SIZE],

    /// Ternary flag to indicate which internal-recursive layer this Proof is for. Should be
    /// 1 for the first (i.e. index 0) internal-recursive layer, 2 for subsequent layers, and
    /// 0 everywhere else.
    pub recursion_flag: F,
    /// Cached trace commit of each subsequent (i.e. index > 0) internal-recursive layer
    /// verifier's SymbolicExpressionAir, which is derived from the internal_recursive_vk
    pub internal_recursive_dag_commit: [F; DIGEST_SIZE],
}
