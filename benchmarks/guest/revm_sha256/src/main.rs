use openvm as _;

use core::hint::black_box;

use hex_literal::hex;
use revm_precompile::{hash::sha256_run, Bytes};

const TEST_CASES: &[(&[u8], [u8; 32])] = &[
    // "ethereum/tests/Hash function SHA256"
    (
        &hex!("0000000ccccccccccccccccccccccccccccccccccccccccccccccccccc000000"),
        hex!("73f5062fb68ed2a1ec82ff8c73f9251bb9cf53a623bc93527e16bc5ae29dad74"),
    ),
    // "ethereum/tests/CALLCODESha256_0"
    (
        &hex!("0000000000000000000000000000000000000000000000000000000000000001"),
        hex!("ec4916dd28fc4c10d78e287ca5d9cc51ee1ae73cbfde08c6b37324cbfaac8bc5"),
    ),
    // "ethereum/tests/CALLCODESha256_1"
    (
        &hex!(""),
        hex!("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ),
    // "ethereum/tests/CALLCODESha256_2"
    (
        &hex!("0000000000000000000000000000000000000000000000000000000000000000f34578907f"),
        hex!("cb39b3bde22925b2f931111130c774761d8895e0e08437c9b396c1e97d10f34d"),
    ),
    // "ethereum/tests/CALLCODESha256_3"
    (
        &hex!("000000000000000000000000000000000000000000000000000000f34578907f0000000000"),
        hex!("7392925565d67be8e9620aacbcfaecd8cb6ec58d709d25da9eccf1d08a41ce35"),
    ),
    // "ethereum/tests/CALLCODESha256_3_postfix0"
    (
        &hex!("0000000000000000000000000000000000000000000000000000f34578907f000000000000"),
        hex!("3b745a1c00d035c334f358d007a430e4cf0ae63aa0556fb05529706de546464d"),
    ),
    // "ethereum/tests/CALLCODESha256_4"
    (
        &hex!("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"),
        hex!("af9613760f72635fbdb44a5a0a63c39f12af30f950a6ee5c971be188e89c4051"),
    ),
];

fn main() {
    for (input, expected) in TEST_CASES {
        let input = black_box(Bytes::from_static(input));

        let outcome = sha256_run(&input, u64::MAX).unwrap();
        assert_eq!(outcome.bytes.as_ref(), expected.as_slice());
    }
}
