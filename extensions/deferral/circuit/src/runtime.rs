//! F-typed builders for the rvr-side deferral fields on `Streams<F>`.
//!
//! TODO: drop this module once rvr execution is unified onto openvm
//! `VmState<F>`.

use std::sync::Arc;

use openvm_circuit::arch::{
    rvr::{DeferralFnPtr, DeferralHashFn},
    VmField,
};
use rvr_openvm_ext_ffi_common::DEFERRAL_COMMIT_NUM_BYTES;

use crate::{def_fn::hash_output_raw, poseidon2::deferral_poseidon2_chip, DeferralExtension};

pub fn make_deferral_fns(extension: &DeferralExtension) -> Vec<DeferralFnPtr> {
    extension
        .fns
        .iter()
        .map(|fn_arc| {
            let fn_arc = Arc::clone(fn_arc);
            Arc::new(move |input: &[u8]| fn_arc.call_raw(input)) as DeferralFnPtr
        })
        .collect()
}

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
