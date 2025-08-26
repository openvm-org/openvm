#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use elliptic_curve::{ops::LinearCombination, CurveArithmetic, Field, Group, PrimeField};
use openvm_p256::NistP256;
// clippy thinks this is unused, but it's used in the init! macro
#[allow(unused)]
use openvm_p256::{P256Point, P256Point as ProjectivePoint, P256Scalar as Scalar};

openvm::init!("openvm_init_various.rs");

openvm::entry!(main);

pub fn main() {
    // from scalar_sqrt.rs
    type Scalar = <NistP256 as CurveArithmetic>::Scalar;

    let a = Scalar::from_u128(4);
    let b = a.sqrt().unwrap();
    assert!(b == Scalar::from_u128(2) || b == -Scalar::from_u128(2));

    let a = Scalar::from_u128(5);
    let b = a.sqrt().unwrap();
    let sqrt_5 = Scalar::from_str_vartime(
        "37706888570942939511621860890978929712654002332559277021296980149138421130241",
    )
    .unwrap();
    assert!(b == sqrt_5 || b == -sqrt_5);
    assert!(b * b == a);

    let a = Scalar::from_u128(7);
    let b = a.sqrt();
    assert!(bool::from(b.is_none()));

    // from linear_combination.rs
    let g = ProjectivePoint::generator();
    let a = ProjectivePoint::lincomb(&g, &Scalar::from_u128(100), &g, &Scalar::from_u128(156));
    let mut b = g;
    for _ in 0..8 {
        b += b;
    }
    assert_eq!(a, b);
}
