#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(target_os = "zkvm", no_std)]

extern crate alloc;

use alloc::vec::Vec;

use openvm::io::read;
use openvm_deferral_guest::Commit;
use openvm_verify_stark_guest::{verify_proof_output, ProofOutput};

openvm::entry!(main);

pub fn main() {
    let app_exe_commit: Commit = read();
    let app_vk_commit: Commit = read();
    let user_public_values: Vec<u8> = read();

    let expected = ProofOutput {
        app_exe_commit,
        app_vk_commit,
        user_public_values,
    };

    let input_commit: Commit = read();
    verify_proof_output::<0>(&input_commit, &expected);
}
