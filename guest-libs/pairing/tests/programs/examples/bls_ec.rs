#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

#[allow(unused_imports)]
use openvm_pairing::bls12_381::Bls12_381G1Affine;

openvm::init!("openvm_init_bls_ec_bls12_381.rs");

openvm::entry!(main);

pub fn main() {
    use openvm_algebra_guest::IntMod;
    use openvm_ecc_guest::weierstrass::WeierstrassPoint;
    use openvm_pairing::bls12_381::{Fp, G1Affine, G2Affine, Scalar};

    assert_eq!(
        G2Affine::IDENTITY.as_le_bytes(),
        G2Affine::IDENTITY.as_le_bytes_unchecked()
    );

    // This on-curve point has order 3; the scalar modulus is 1 modulo 3.
    let point = unsafe { G1Affine::from_xy_unchecked(Fp::ZERO, Fp::from_u32(2)) };
    let scalar = Scalar::from_repr(Scalar::MODULUS);
    assert_eq!(
        openvm_ecc_guest::msm(&[scalar], core::slice::from_ref(&point)),
        point
    );
}
