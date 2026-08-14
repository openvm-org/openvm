#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use hex_literal::hex;
use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::{
    weierstrass::{IntrinsicCurve, WeierstrassPoint},
    CyclicGroup, Group,
};
use openvm_pairing::bls12_381::{Bls12_381, Bls12_381G1Affine, Fp, Scalar};

openvm::init!("openvm_init_bls_offsubgroup_mul_bls12_381.rs");

openvm::entry!(main);

/// `P = (4, y)` satisfies `y^2 = x^3 + 4` but lies outside the prime-order subgroup: `r * P != O`.
/// A value the public API admits, e.g. a point deserialized without a subgroup check.
const P_X_LE: [u8; 48] = hex!(
    "040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
);
const P_Y_LE: [u8; 48] = hex!(
    "6C70BE4A353EA95E5DDEE100EDB84663448384925ED89DDA266B92C988F960C79B3E76F3C3FF3CB312620DD4AD9B980A"
);

fn point(x_le: &[u8; 48], y_le: &[u8; 48]) -> Bls12_381G1Affine {
    let x = Fp::from_le_bytes(x_le).unwrap();
    let y = Fp::from_le_bytes(y_le).unwrap();
    // SAFETY: (x, y) is on the curve. No subgroup membership is claimed, which is exactly the
    // case `from_xy` is documented to allow.
    unsafe { Bls12_381G1Affine::from_xy(x, y) }.unwrap()
}

pub fn main() {
    let two = Scalar::from_u64(2);
    let three = Scalar::from_u64(3);

    // Control: inside the prime-order subgroup.
    let g = Bls12_381G1Affine::GENERATOR;
    assert_eq!(g.mul_scalar(&two), g.double(), "control: generator");

    let p = point(&P_X_LE, &P_Y_LE);

    // `2 * P` by plain doubling: exact, and it assumes nothing about the point's order. The
    // even-scalar branch computes `(k - 1) * P + P`, which must agree; the cofactor-1 curves'
    // `-((n - k) * P)` substitution would not, since `r * P` is not the identity here.
    let expected = p.double();
    assert_eq!(p.mul_scalar(&two), expected, "mul_scalar(2) == 2*P");

    // Odd scalars take the ladder directly; `3 * P = 2 * P + P` for any point.
    assert_eq!(p.mul_scalar(&three), expected.clone() + &p, "mul_scalar(3) == 3*P");

    // `Bls12_381::msm` routes every base through the same `mul_scalar`.
    assert_eq!(Bls12_381::msm(&[two], &[p]), expected, "msm(2, P) == 2*P");
}
