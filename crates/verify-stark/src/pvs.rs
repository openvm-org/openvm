use stark_backend_v2::DIGEST_SIZE;
use stark_recursion_circuit_derive::AlignedBorrow;

pub const VERIFIER_PVS_AIR_ID: usize = 0;

/// Public values interpretation for the AIR at index VERIFIER_PVS_AIR_ID
#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy)]
pub struct NonRootVerifierPvs<F> {
    // app commit pvs
    pub user_pv_commit: [F; DIGEST_SIZE],
    pub program_commit: [F; DIGEST_SIZE],

    // connector pvs
    pub initial_pc: F,
    pub final_pc: F,
    pub exit_code: F,
    pub is_terminate: F,

    // memory merkle pvs
    pub initial_root: [F; DIGEST_SIZE],
    pub final_root: [F; DIGEST_SIZE],

    // verifier-specific pvs
    pub internal_flag: F,
    pub leaf_commit: [F; DIGEST_SIZE],
    pub internal_for_leaf_commit: [F; DIGEST_SIZE],
    pub internal_recursive_commit: [F; DIGEST_SIZE],
}
