extern crate openvm;

use core::hint::black_box;

use hex_literal::hex;
use openvm_algebra_complex_macros::complex_init;
use openvm_algebra_guest::moduli_macros::moduli_init;
use openvm_ecc_guest::sw_macros::sw_init;
use revm_precompile::{kzg_point_evaluation, Bytes};
use revm_primitives::Env;

#[allow(unused_imports)]
use openvm_pairing_guest::bls12_381::Bls12_381G1Affine;

// initialize moduli
moduli_init! {
    // bls12_381
    "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", // coordinate field
    "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", // scalar field
}
// initialize complex extensions of moduli
complex_init! {
    Bls12_381Fp2 { mod_idx = 0 },
}
// initialize elliptic curves
sw_init! {
    Bls12_381G1Affine,
}

const TEST_CASES: &[(&[u8], [u8; 64])] = &[
    (
        &hex!("01e798154708fe7789429634053cbf9f99b619f9f084048927333fce637f549b564c0a11a0f704f4fc3e8acfe0f8245f0ad1347b378fbf96e206da11a5d3630624d25032e67a7e6a4910df5834b8fe70e6bcfeeac0352434196bdf4b2485d5a18f59a8d2a1a625a17f3fea0fe5eb8c896db3764f3185481bc22f91b4aaffcca25f26936857bc3a7c2539ea8ec3a952b7873033e038326e87ed3e1276fd140253fa08e9fc25fb2d9a98527fc22a2c9612fbeafdad446cbc7bcdbdcd780af2c16a"),
        hex!("000000000000000000000000000000000000000000000000000000000000100073eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"),
    ),
];

fn main() {
    {
        setup_all_moduli();
        setup_all_complex_extensions();
        setup_all_curves();
    }

    for (input, expected) in TEST_CASES {
        let input = black_box(Bytes::from_static(input));

        let outcome = kzg_point_evaluation::run(&input, u64::MAX, &Env::default()).unwrap();
        assert_eq!(outcome.bytes.as_ref(), expected.as_slice());
    }
}
