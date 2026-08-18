#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::{
    weierstrass::{IntrinsicCurve, WeierstrassPoint},
    CyclicGroup, Group,
};
use openvm_pairing::bls12_381::{Bls12_381, Bls12_381G1Affine, Fp, Scalar};

openvm::init!("openvm_init_bls_offsubgroup_mul_bls12_381.rs");

openvm::entry!(main);

pub fn main() {
    let zero = Scalar::ZERO;
    let one = Scalar::from_u64(1);
    let two = Scalar::from_u64(2);
    let three = Scalar::from_u64(3);

    // The generator is in the prime-order subgroup.
    let g = Bls12_381G1Affine::GENERATOR;
    assert!(Bls12_381::mul_generator(&zero).is_identity());
    assert_eq!(Bls12_381::mul_generator(&two), g.double());

    let mut q_minus_one = Scalar::MODULUS;
    q_minus_one[0] -= 1;
    let q_minus_one = Scalar::from_le_bytes_unchecked(&q_minus_one);
    let q = Scalar::from_le_bytes_unchecked(&Scalar::MODULUS);
    let mut q_plus_one = Scalar::MODULUS;
    q_plus_one[0] += 1;
    let q_plus_one = Scalar::from_le_bytes_unchecked(&q_plus_one);
    assert_eq!(Bls12_381::mul_generator(&q_minus_one), -g.clone());
    assert!(Bls12_381::mul_generator(&q).is_identity());
    assert_eq!(Bls12_381::mul_generator(&q_plus_one), g);

    // SAFETY: Both bases are the standard prime-subgroup generator.
    let msm = unsafe {
        Bls12_381::msm_prime_subgroup_unchecked(&[one.clone(), two.clone()], &[g.clone(), g])
    };
    assert_eq!(msm, Bls12_381G1Affine::GENERATOR.mul_scalar(&three));

    // P = (0, 2) is on y^2 = x^3 + 4 and has order 3.
    // SAFETY: `from_xy` accepts any on-curve point. It does not require subgroup membership.
    let p = unsafe { Bls12_381G1Affine::from_xy(Fp::ZERO, Fp::from_u8(2)) }.unwrap();
    let two_p = p.double();

    assert_eq!(p.mul_scalar(&one), p);
    assert_eq!(p.mul_scalar(&two), two_p);
    assert!(p.mul_scalar(&three).is_identity());
    assert_eq!(Bls12_381::msm(&[two], &[p]), two_p);
}
