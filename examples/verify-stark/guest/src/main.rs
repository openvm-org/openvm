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

const MAX_NUM_DEF_CIRCUITS: u32 = 8;

fn verify_stark_at_def_idx(input_commit: &Commit, def_idx: u32) {
    let output = match def_idx {
        0 => verify_stark_unchecked::<0>(input_commit),
        #[cfg(feature = "deferral-1")]
        1 => verify_stark_unchecked::<1>(input_commit),
        #[cfg(feature = "deferral-2")]
        2 => verify_stark_unchecked::<2>(input_commit),
        #[cfg(feature = "deferral-3")]
        3 => verify_stark_unchecked::<3>(input_commit),
        #[cfg(feature = "deferral-4")]
        4 => verify_stark_unchecked::<4>(input_commit),
        #[cfg(feature = "deferral-5")]
        5 => verify_stark_unchecked::<5>(input_commit),
        #[cfg(feature = "deferral-6")]
        6 => verify_stark_unchecked::<6>(input_commit),
        #[cfg(feature = "deferral-7")]
        7 => verify_stark_unchecked::<7>(input_commit),
        _ => unreachable!(),
    };
    println(format!("app_exe_commit: {:?}", output.app_exe_commit));
    println(format!("app_vm_commit: {:?}", output.app_vm_commit));
    println(format!("user_pvs: {:?}", output.user_public_values));
}

pub fn main() {
    let input_commit: Commit = read();
    let num_def_circuits: u32 = read();
    assert!(num_def_circuits <= MAX_NUM_DEF_CIRCUITS);
    for def_idx in 0..num_def_circuits {
        let num_verifies: u32 = read();
        for _ in 0..num_verifies {
            verify_stark_at_def_idx(&input_commit, def_idx);
        }
    }
}
