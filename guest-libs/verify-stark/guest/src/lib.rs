#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{vec, vec::Vec};

use openvm_deferral_guest::{deferred_compute, get_deferred_output, Commit, COMMIT_NUM_BYTES};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofOutput {
    pub app_exe_commit: Commit,
    pub app_vm_commit: Commit,
    pub user_public_values: Vec<u8>,
}

pub fn verify_stark_unchecked<const DEF_IDX: u16>(input_commit: &Commit) -> ProofOutput {
    let output_key = deferred_compute::<DEF_IDX>(input_commit);
    let output_len = output_key.output_len as usize;

    const MIN_OUTPUT_BYTES: usize = 2 * COMMIT_NUM_BYTES;
    if output_len < MIN_OUTPUT_BYTES {
        panic!("output_len too small for a ProofOutput");
    }

    let mut output_bytes = vec![0u8; output_len];
    get_deferred_output::<DEF_IDX>(&mut output_bytes, &output_key);

    let app_exe_commit = output_bytes[..COMMIT_NUM_BYTES].try_into().unwrap();
    let app_vm_commit = output_bytes[COMMIT_NUM_BYTES..MIN_OUTPUT_BYTES]
        .try_into()
        .unwrap();
    let user_public_values = output_bytes[MIN_OUTPUT_BYTES..].to_vec();

    ProofOutput {
        app_exe_commit,
        app_vm_commit,
        user_public_values,
    }
}

pub fn verify_stark<const DEF_IDX: u16>(input_commit: &Commit, expected: &ProofOutput) {
    let actual = verify_stark_unchecked::<DEF_IDX>(input_commit);
    if actual != *expected {
        panic!("Proof verification failed for commit {:?}", input_commit);
    }
}
