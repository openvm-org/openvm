#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::mem::transmute;

use base64::engine::general_purpose;

axvm::entry!(main);

fn main() {
    let data_b64 = axvm::io::read_vec();
    let data_b64 = core::str::from_utf8(&data_b64).expect("Invalid UTF-8");

    let decoded = general_purpose::STANDARD
        .decode(data_b64)
        .expect("Failed to decode base64");
    let data = core::str::from_utf8(&decoded).expect("Failed to decode base64");
    let data = data.replace("\r\n", "\n");

    let data_hash = axvm::intrinsics::keccak256(data.as_str().as_bytes());

    let data_hash = unsafe { transmute::<[u8; 32], [u32; 8]>(data_hash) };

    data_hash
        .into_iter()
        .enumerate()
        .for_each(|(i, x)| axvm::io::reveal(x, i));
}
