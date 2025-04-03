use core::hint::black_box;

use hex::FromHex;
use openvm_sha256_guest::sha256;

pub fn main() {
    let input = "";

    let input = Vec::from_hex(black_box(input)).unwrap();
    let output = sha256(&input);

    black_box(output);
}
