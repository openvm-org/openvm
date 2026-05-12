//! F-typed builder for the rvr deferral output hasher.
//!
//! The hasher is constructed from a Poseidon2 chip parameterized over `F` and
//! handed off to `rvr-openvm-ext-deferral`'s thread-local runtime by
//! `DeferralExtension::extend_rvr`.

use std::sync::Arc;

use openvm_circuit::arch::VmField;
use rvr_openvm_ext_deferral::DeferralHashFn;
use rvr_openvm_ext_ffi_common::DEFERRAL_COMMIT_NUM_BYTES;

use crate::{def_fn::hash_output_raw, poseidon2::deferral_poseidon2_chip};

pub fn make_deferral_hash<F: VmField>() -> DeferralHashFn {
    let hasher = deferral_poseidon2_chip::<F>();
    Arc::new(
        move |def_idx: u32, output_raw: &[u8]| -> [u8; DEFERRAL_COMMIT_NUM_BYTES] {
            hash_output_raw(&hasher, def_idx, output_raw)
                .as_slice()
                .try_into()
                .unwrap()
        },
    )
}
