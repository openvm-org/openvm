openvm::entry!(main);

// [!region imports]
use core::hint::black_box;

use hex::FromHex;
use openvm_sha2::sha256;
// [!endregion imports]

// [!region main]
openvm::entry!(main);

pub fn main() {
    let test_vectors = [(
        "",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )];
    for (input, expected_output) in test_vectors.iter() {
        let input = Vec::from_hex(input).unwrap();
        let expected_output = Vec::from_hex(expected_output).unwrap();
        let output = sha256(&black_box(input));
        if output != *expected_output {
            panic!();
        }
    }
}
// [!endregion main]
