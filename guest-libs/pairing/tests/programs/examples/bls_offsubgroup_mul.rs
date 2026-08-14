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
///
/// BLS12-381 G1 has cofactor `h ~ 2^126`, so all but a `1/h` fraction of on-curve points are
/// outside it. `WeierstrassPoint::from_xy` documents that it "does not perform any subgroup checks
/// and only guarantees that the point is on the curve", and `G1Affine`'s own doc notes an instance
/// "may be constructed that lies on the curve but not necessarily in the prime order subgroup" --
/// so this is a value the public API admits, e.g. a deserialized point without a subgroup check.
const P_X_LE: [u8; 48] = hex!(
    "040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
);
const P_Y_LE: [u8; 48] = hex!(
    "6C70BE4A353EA95E5DDEE100EDB84663448384925ED89DDA266B92C988F960C79B3E76F3C3FF3CB312620DD4AD9B980A"
);

/// `-((r - 2) * P)`, i.e. what `mul_scalar(2)`'s even-scalar branch actually returns for the `P`
/// above. It differs from `2 * P` by exactly `-r * P`, which is the identity only inside G1.
const WRONG_X_LE: [u8; 48] = hex!(
    "F4A0460139E6D81D13F5A8796A6CF2E94E25B9EB69255EF9F53C6292291DE7859BB9E0B4B8ECAA7A81EB5D4B9E74A114"
);
const WRONG_Y_LE: [u8; 48] = hex!(
    "A251B94F3A5B38976E8833171269790F40375D5D7B4C64AAF3FF583E5A9C0177F9A9717302BFC796A1F0CAE692BB8008"
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

    // Control: inside the prime-order subgroup the even-scalar substitution is sound, so the
    // generator behaves correctly. Any failure below is therefore about the subgroup, not about
    // point construction, setup, or the EC_MUL chip.
    let g = Bls12_381G1Affine::GENERATOR;
    assert_eq!(g.mul_scalar(&two), g.double(), "control: generator");

    let p = point(&P_X_LE, &P_Y_LE);

    // `2 * P` by plain doubling: exact, and it assumes nothing about the point's order.
    let expected = p.double();

    // `mul_scalar(2)` takes the even-scalar branch and computes `-((r - 2) * P)`, which equals
    // `2 * P` only when `r * P` is the identity.
    let product = p.mul_scalar(&two);
    assert_eq!(
        product,
        point(&WRONG_X_LE, &WRONG_Y_LE),
        "predicted -((r-2)*P)"
    );
    assert_ne!(product, expected, "mul_scalar(2) should have been 2*P");

    // `Bls12_381::msm` now routes every base through that same `mul_scalar`, so a single-term MSM
    // inherits the bug.
    let via_msm = Bls12_381::msm(&[two], &[p]);
    assert_eq!(via_msm, product, "msm agrees with the buggy mul_scalar");
    assert_ne!(via_msm, expected, "msm(2, P) should have been 2*P");
}
