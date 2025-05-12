extern crate alloc;
use alloc::vec::Vec;

use openvm::{define_verify_openvm_stark, io::read};

define_verify_openvm_stark!(
    verify_openvm_stark,
    env!("CARGO_MANIFEST_DIR"),
    "root_verifier.asm"
);

// const APP_EXE_COMMIT: [u32; 8] = [
//     343014587, 230645511, 1447462186, 773379336, 1182270030, 1497892484, 461820702, 353704350,
// ];
// const APP_VM_COMMIT: [u32; 8] = [
//     445134834, 1133596793, 530952192, 425228715, 1806903712, 1362083369, 295028151, 482389308,
// ];

pub fn main() {
    let app_exe_commit: [u32; 8] = read();
    let app_vm_commit: [u32; 8] = read();
    let pvs: Vec<u32> = read();
    verify_openvm_stark(&app_exe_commit, &app_vm_commit, &pvs);
}
