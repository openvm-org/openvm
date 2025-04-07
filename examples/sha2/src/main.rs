// ANCHOR: imports
use core::hint::black_box;

use hex::FromHex;
use openvm_sha2_guest::{sha256, sha384, sha512};
// ANCHOR_END: imports

// ANCHOR: main
openvm::entry!(main);

pub fn main() {
    let test_vectors = [(
        "",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
        "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b",
    )];
    for (input, expected_output_sha256, expected_output_sha512, expected_output_sha384) in
        test_vectors.iter()
    {
        let input = Vec::from_hex(input).unwrap();
        let expected_output_sha256 = Vec::from_hex(expected_output_sha256).unwrap();
        let expected_output_sha512 = Vec::from_hex(expected_output_sha512).unwrap();
        let expected_output_sha384 = Vec::from_hex(expected_output_sha384).unwrap();
        let output = sha256(black_box(&input));
        if output != *expected_output_sha256 {
            panic!();
        }
        let output = sha512(black_box(&input));
        if output != *expected_output_sha512 {
            panic!();
        }
        let output = sha384(black_box(&input));
        if output != *expected_output_sha384 {
            panic!();
        }
    }
}
// ANCHOR_END: main
