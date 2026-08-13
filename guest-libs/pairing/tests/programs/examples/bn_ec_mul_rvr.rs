#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::{CyclicGroup, Group};
use openvm_pairing::bn254::{Bn254G1Affine, Scalar};

openvm::init!("openvm_init_bn_ec_mul_rvr_bn254.rs");

openvm::entry!(main);

fn raw_mul(base: &Bn254G1Affine, scalar: u64) -> Bn254G1Affine {
    let scalar = Scalar::from_u64(scalar);
    let bytes: [u8; 32] = scalar.as_le_bytes().try_into().unwrap();
    // SAFETY: This fixture intentionally exercises the RVR handler's raw EC_MUL contract,
    // including its defined behavior for an even scalar and the identity base.
    unsafe { base.mul_scalar_le_unchecked(&bytes) }
}

pub fn main() {
    let g = Bn254G1Affine::GENERATOR;
    let two_g = g.double();
    let three_g = &two_g + &g;
    let six_g = three_g.double();

    assert_eq!(raw_mul(&g, 1), g);
    assert_eq!(raw_mul(&g, 2), three_g);
    assert_eq!(raw_mul(&g, 3), three_g);

    assert_eq!(raw_mul(&two_g, 1), two_g);
    assert_eq!(raw_mul(&two_g, 2), six_g);
    assert_eq!(raw_mul(&two_g, 3), six_g);

    let identity = <Bn254G1Affine as Group>::IDENTITY;
    assert!(raw_mul(&identity, 7).is_identity());
}
