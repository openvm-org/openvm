use stark_backend_v2::DIGEST_SIZE;
use stark_recursion_circuit_derive::AlignedBorrow;

pub mod bus;
pub mod commit;
pub mod memory;
pub mod verifier;
#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct RootVerifierPvs<F> {
    /// Hashed combination of the app-level ProgramAir cached trace, the Merkle root commit of
    /// the starting app memory state (i.e. initial_root), and the initial app program counter
    /// (i.e. initial_pc).
    pub app_exe_commit: [F; DIGEST_SIZE],
    /// Cached trace commit of the leaf verifier circuit's SymbolicExpressionAir, which is
    /// derived from the app_vk
    pub app_vk_commit: [F; DIGEST_SIZE],
    /// Cached trace commit of the internal-for-leaf verifier circuit's SymbolicExpressionAir,
    /// which is derived from the leaf_vk
    pub leaf_vk_commit: [F; DIGEST_SIZE],
    /// Cached trace commit of the first (i.e. index 0) internal-recursive layer verifier
    /// circuit's SymbolicExpressionAir, which is derived from the internal_for_leaf_vk
    pub internal_for_leaf_vk_commit: [F; DIGEST_SIZE],
}
