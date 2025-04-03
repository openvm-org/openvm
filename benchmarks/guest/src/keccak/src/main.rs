use core::hint::black_box;

use hex::FromHex;
use openvm::io::reveal_bytes32;
use openvm_keccak256_guest::keccak256;

pub fn main() {
    let input = "";

    let input = Vec::from_hex(black_box(input)).unwrap();
    let output = keccak256(&input);

    black_box(output);
}
