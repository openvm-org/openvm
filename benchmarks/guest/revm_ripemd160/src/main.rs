use openvm as _;

use core::hint::black_box;

use hex_literal::hex;
use revm_precompile::{hash::ripemd160_run, Bytes};

const TEST_CASES: &[(&[u8], [u8; 32])] = &[
    (
        &hex!("38d18acb67d25c8bb9942764b62f18e17054f66a817bd4295423adf9ed98873e000000000000000000000000000000000000000000000000000000000000001b38d18acb67d25c8bb9942764b62f18e17054f66a817bd4295423adf9ed98873e789d1dd423d25f0772d2748d60f7e4b81bb14d086eba8e8e8efb6dcff8a4ae02"),
        hex!("0000000000000000000000009215b8d9882ff46f0dfde6684d78e831467f65e6"),
    ),
];

fn main() {
    for (input, expected) in TEST_CASES {
        let input = black_box(Bytes::from_static(input));

        let outcome = ripemd160_run(&input, u64::MAX).unwrap();
        assert_eq!(outcome.bytes.as_ref(), expected.as_slice());
    }
}
