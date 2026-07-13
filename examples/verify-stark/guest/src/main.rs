#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::format;

use openvm::io::{println, read};
use openvm_deferral_guest::Commit;
use openvm_verify_stark_guest::verify_stark_unchecked;

openvm::entry!(main);

pub fn main() {
    let input_commit: Commit = read();

    let output = verify_stark_unchecked::<0>(&input_commit);
    println(format!("app_exe_commit: {:?}", output.app_exe_commit));
    println(format!("app_vm_commit: {:?}", output.app_vm_commit));
    println(format!(
        "user_public_values: {:?}",
        output.user_public_values
    ));
}
