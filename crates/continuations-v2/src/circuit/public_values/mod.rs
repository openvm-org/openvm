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
#[derive(AlignedBorrow)]
pub struct RootVerifierPvs<F> {
    pub app_commit: [F; DIGEST_SIZE],
    pub app_vk_commit: [F; DIGEST_SIZE],
    pub leaf_vk_commit: [F; DIGEST_SIZE],
    pub internal_for_leaf_vk_commit: [F; DIGEST_SIZE],
}
