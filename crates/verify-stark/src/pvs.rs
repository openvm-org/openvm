use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use stark_recursion_circuit_derive::AlignedBorrow;

pub const VERIFIER_PVS_AIR_ID: usize = 0;
pub const VM_PVS_AIR_ID: usize = 1;
pub const DEF_PVS_AIR_ID: usize = 2;
pub const CONSTRAINT_EVAL_AIR_ID: usize = 3;

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy)]
pub struct VerifierBasePvs<F> {
    //////////////////////////////////////////////////////////////////////
    /// VERIFIER-SPECIFIC PVS
    //////////////////////////////////////////////////////////////////////
    /// Ternary flag to indicate which continuations layer this Proof is for. Should be 0 for
    /// the leaf verifier, 1 for the internal-for-leaf verifier, and 2 for the internal-
    /// recursive verifier.
    pub internal_flag: F,
    /// Cached trace commit of the leaf verifier circuit's SymbolicExpressionAir, which is
    /// derived from the app_vk
    pub app_dag_commit: [F; DIGEST_SIZE],
    /// Cached trace commit of the internal-for-leaf verifier circuit's SymbolicExpressionAir,
    /// which is derived from the leaf_vk
    pub leaf_dag_commit: [F; DIGEST_SIZE],
    /// Cached trace commit of the first (i.e. index 0) internal-recursive layer verifier
    /// circuit's SymbolicExpressionAir, which is derived from the internal_for_leaf_vk
    pub internal_for_leaf_dag_commit: [F; DIGEST_SIZE],

    //////////////////////////////////////////////////////////////////////
    /// VERIFIER-SPECIFIC RECURSION PVS
    //////////////////////////////////////////////////////////////////////
    /// Ternary flag to indicate which internal-recursive layer this Proof is for. Should be
    /// 1 for the first (i.e. index 0) internal-recursive layer, 2 for subsequent layers, and
    /// 0 everywhere else.
    pub recursion_flag: F,
    /// Cached trace commit of each subsequent (i.e. index > 0) internal-recursive layer
    /// verifier's SymbolicExpressionAir, which is derived from the internal_recursive_vk
    pub internal_recursive_dag_commit: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy)]
pub struct VerifierDefPvs<F> {
    //////////////////////////////////////////////////////////////////////
    /// DEFERRAL-SPECIFIC PVS
    //////////////////////////////////////////////////////////////////////
    /// Ternary flag to indicate which public values this Proof contains. Should be 0 if it
    /// has only VM public values defined, 1 if only deferral public values, and 2 if both.
    pub deferral_flag: F,
    /// Commit to the deferral hook verifying key, computed by compressing together the app,
    /// leaf, and internal-for-leaf DAG commits when deferral_flag == 1. Is set exactly when
    /// internal_for_leaf_dag_commit is set.
    pub def_hook_vk_commit: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy)]
pub struct VmPvs<F> {
    //////////////////////////////////////////////////////////////////////
    /// PROGRAM COMMIT PVS
    //////////////////////////////////////////////////////////////////////
    /// Cached trace commit of the app verifier circuit's ProgramAir.
    pub program_commit: [F; DIGEST_SIZE],

    //////////////////////////////////////////////////////////////////////
    /// CONNECTOR PVS
    //////////////////////////////////////////////////////////////////////
    /// Starting PC value of the program (or segment) run.
    pub initial_pc: F,
    /// Final PC value of the program (or segment) run.
    pub final_pc: F,
    /// Exit code of the program run.
    pub exit_code: F,
    /// Boolean flag to determine whether or not this segment terminated the program.
    pub is_terminate: F,

    //////////////////////////////////////////////////////////////////////
    /// MEMORY MERKLE PVS
    //////////////////////////////////////////////////////////////////////
    /// Merkle root commit of the starting memory state for this program (or segment).
    pub initial_root: [F; DIGEST_SIZE],
    /// Merkle root commit of the final memory state for this program (or segment).
    pub final_root: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy)]
pub struct DeferralPvs<F> {
    /// Merkle root of all the initial hash accumulators for deferral circuit proofs that
    /// have been aggregated up to this point.
    pub initial_acc_hash: [F; DIGEST_SIZE],
    /// Merkle root of all the final hash accumulators for deferral circuit proofs that
    /// have been aggregated up to this point.
    pub final_acc_hash: [F; DIGEST_SIZE],
    /// Depth of the Merkle subtrees above.
    pub depth: F,
}
