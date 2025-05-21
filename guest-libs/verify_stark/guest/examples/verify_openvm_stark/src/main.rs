extern crate alloc;
use alloc::vec::Vec;

use openvm::io::read;
use openvm_verify_stark::define_verify_openvm_stark;

define_verify_openvm_stark!(
    verify_openvm_stark,
    env!("CARGO_MANIFEST_DIR"),
    "root_verifier.asm"
);

pub fn main() {
    let app_exe_commit: [u32; 8] = read();
    let app_vm_commit: [u32; 8] = read();
    let pvs: Vec<u8> = read();
    verify_openvm_stark(&app_exe_commit, &app_vm_commit, &pvs);
}
