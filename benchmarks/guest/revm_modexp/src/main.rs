extern crate openvm;

use core::hint::black_box;

use hex_literal::hex;
use revm_precompile::modexp;

const TEST_CASES: &[(&[u8], &[u8])] = &[
    (
        &hex!("00000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000002003fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2efffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"),
        &hex!("0000000000000000000000000000000000000000000000000000000000000001"),
    ),
];

fn main() {
    for (input, expected) in TEST_CASES {
        let input = black_box(input);

        let outcome = modexp::run_inner(input, u64::MAX, 0, |_, _, _, _| 0).unwrap();
        assert_eq!(&outcome.bytes.as_ref(), expected);
    }
}
