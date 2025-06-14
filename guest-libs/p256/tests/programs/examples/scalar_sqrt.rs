#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use elliptic_curve::{CurveArithmetic, Field, PrimeField};
use openvm_p256::NistP256;
// clippy thinks this is unused, but it's used in the init! macro
#[allow(unused)]
use openvm_p256::P256Point;

openvm::init!("openvm_init_scalar_sqrt.rs");

openvm::entry!(main);

pub fn main() {
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
}
