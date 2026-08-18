#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::{CyclicGroup, Group};
use openvm_pairing::bn254::{Bn254G1Affine, Bn254Scalar};

openvm::init!("openvm_init_bn_ec_mul_rvr_bn254.rs");

openvm::entry!(main);

fn mul(base: &Bn254G1Affine, scalar: u64) -> Bn254G1Affine {
    base.mul_scalar(&Bn254Scalar::from_u64(scalar))
}

pub fn main() {
    let g = Bn254G1Affine::GENERATOR;
    let two_g = g.double();
    let three_g = &two_g + &g;
    let four_g = two_g.double();

    assert_eq!(mul(&g, 1), g);
    assert_eq!(mul(&g, 2), two_g);
    assert_eq!(mul(&g, 3), three_g);

    assert_eq!(mul(&two_g, 1), two_g);
    assert_eq!(mul(&two_g, 2), four_g);
    assert_eq!(mul(&two_g, 3), &four_g + &two_g);

    let identity = <Bn254G1Affine as Group>::IDENTITY;
    assert!(mul(&identity, 7).is_identity());
}
