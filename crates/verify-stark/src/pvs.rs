use stark_backend_v2::DIGEST_SIZE;
use stark_recursion_circuit_derive::AlignedBorrow;

pub const VERIFIER_PVS_AIR_ID: usize = 0;

/// Public values interpretation for the AIR at index VERIFIER_PVS_AIR_ID
#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy)]
pub struct NonRootVerifierPvs<F> {
    //////////////////////////////////////////////////////////////////////
    /// APP COMMIT PVS
    //////////////////////////////////////////////////////////////////////
    /// Merkle root commit of user-defined public values.
    pub user_pv_commit: [F; DIGEST_SIZE],
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

    //////////////////////////////////////////////////////////////////////
    /// VERIFIER-SPECIFIC PVS
    //////////////////////////////////////////////////////////////////////
    /// Ternary flag to indicate which continuations layer this Proof is for. Should be 0 for
    /// the leaf verifier, 1 for the internal-for-leaf verifier, and 2 for the internal-
    /// recursive verifier.
    pub internal_flag: F,
    /// Cached trace commit of the leaf verifier circuit's SymbolicExpressionAir, which is
    /// derived from the app_vk
    pub app_vk_commit: [F; DIGEST_SIZE],
    /// Cached trace commit of the internal-for-leaf verifier circuit's SymbolicExpressionAir,
    /// which is derived from the leaf_vk
    pub leaf_vk_commit: [F; DIGEST_SIZE],
    /// Cached trace commit of the first (i.e. index 0) internal-recursive layer verifier
    /// circuit's SymbolicExpressionAir, which is derived from the internal_for_leaf_vk
    pub internal_for_leaf_vk_commit: [F; DIGEST_SIZE],

    //////////////////////////////////////////////////////////////////////
    /// VERIFIER-SPECIFIC RECURSION PVS
    //////////////////////////////////////////////////////////////////////
    /// Ternary flag to indicate which internal-recursive layer this Proof is for. Should be
    /// 1 for the first (i.e. index 0) internal-recursive layer, 2 for subsequent layers, and
    /// 0 everywhere else.
    pub recursion_flag: F,
    /// Cached trace commit of each subsequent (i.e. index > 0) internal-recursive layer
    /// verifier's SymbolicExpressionAir, which is derived from the internal_recursive_vk
    pub internal_recursive_vk_commit: [F; DIGEST_SIZE],
}
