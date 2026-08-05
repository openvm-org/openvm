#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::mem::size_of;

use openvm::io::read;
use openvm_deferral_guest::Commit;
use openvm_verify_stark_guest::{verify_stark, ProofOutput};

openvm::entry!(main);

pub fn main() {
    let app_exe_commit: Commit = read();
    let app_vm_commit: Commit = read();
    let user_public_values: Vec<u8> = read();
    assert!(user_public_values.len().is_multiple_of(size_of::<u64>()));
    let num_public_values = u32::try_from(user_public_values.len() / size_of::<u64>())
        .expect("public-values count exceeds u32");

    let expected = ProofOutput {
        app_exe_commit,
        app_vm_commit,
        num_public_values,
        user_public_values,
    };

    let input_commit: Commit = read();
    verify_stark::<0>(&input_commit, &expected);
}
