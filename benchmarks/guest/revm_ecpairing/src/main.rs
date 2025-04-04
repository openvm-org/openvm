use core::hint::black_box;
use openvm as _;

use hex_literal::hex;
use openvm_algebra_complex_macros::complex_init;
use openvm_algebra_guest::moduli_macros::moduli_init;
use openvm_ecc_guest::sw_macros::sw_init;
use revm_precompile::bn128;

#[allow(unused_imports)]
use openvm_pairing_guest::bn254::Bn254G1Affine;

// initialize moduli
moduli_init! {
    // bn254 (alt bn128)
    "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47", // coordinate field
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001", // scalar field
}
// initialize complex extensions of moduli
complex_init! {
    Bn254Fp2 { mod_idx = 0 },
}
// initialize elliptic curves
sw_init! {
    Bn254G1Affine,
}

const TEST_CASES: &[(&[u8], [u8; 32])] = &[
    (
        &hex!("1c76476f4def4bb94541d57ebba1193381ffa7aa76ada664dd31c16024c43f593034dd2920f673e204fee2811c678745fc819b55d3e9d294e45c9b03a76aef41209dd15ebff5d46c4bd888e51a93cf99a7329636c63514396b4a452003a35bf704bf11ca01483bfa8b34b43561848d28905960114c8ac04049af4b6315a416782bb8324af6cfc93537a2ad1a445cfd0ca2a71acd7ac41fadbf933c2a51be344d120a2a4cf30c1bf9845f20c6fe39e07ea2cce61f0c9bb048165fe5e4de877550111e129f1cf1097710d41c4ac70fcdfa5ba2023c6ff1cbeac322de49d1b6df7c2032c61a830e3c17286de9462bf242fca2883585b93870a73853face6a6bf411198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c21800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa"),
        hex!("0000000000000000000000000000000000000000000000000000000000000001"),
    ),
];

fn main() {
    {
        setup_all_moduli();
        setup_all_complex_extensions();
        setup_all_curves();
    }

    for (input, expected) in TEST_CASES {
        let input = black_box(input);

        let outcome = bn128::run_pair(input, 0, 0, u64::MAX).unwrap();
        assert_eq!(outcome.bytes.as_ref(), expected.as_slice());
    }
}
