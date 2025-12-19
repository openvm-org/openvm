// [!region imports]
// [!endregion imports]

// [!region main]
use hex_literal::hex;
use openvm as _;
use revm_primitives::keccak256;

/// Vector of test cases for Keccak-256 hash function.
/// Each test case consists of (input_bytes, expected_hash_result).
const KECCAK_TEST_CASES: &[(&[u8], [u8; 32])] = &[
    (
        b"",
        hex!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
    )
];
// todo: call the forked tiny keccak library once that is updated instead of directly calling the
// keccak256_guest native functions
pub fn main() {
    #[cfg(target_os = "zkvm")]
    {
        for &(input, expected) in KECCAK_TEST_CASES {
            let mut input = input.to_vec();
            let mut output = [0u8; 32];
            openvm_keccak256_guest::native_keccak256(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
            );
            assert_eq!(output, expected);
        }

        // let mut expected_output = [0u8; 32];
        // openvm_keccak256::keccak256(&buffer);
        // assert_eq!(output, expected_output);
    }
}
// [!endregion main]
