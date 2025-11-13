use stark_backend_v2::DIGEST_SIZE;
use stark_recursion_circuit_derive::AlignedBorrow;

pub mod app {
    pub use openvm_circuit::arch::{
        CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX,
        PUBLIC_VALUES_AIR_ID,
    };
}

pub mod receiver;
pub mod verifier;

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy)]
pub struct NonRootVerifierPvs<F> {
    // app commit pvs
    pub user_pv_commit: [F; DIGEST_SIZE],
    pub app_commit: [F; DIGEST_SIZE],

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

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct RootVerifierPvs<F> {
    pub app_commit: [F; DIGEST_SIZE],
    pub leaf_commit: [F; DIGEST_SIZE],
    pub internal_for_leaf_commit: [F; DIGEST_SIZE],
    pub internal_recursive_commit: [F; DIGEST_SIZE],
}
