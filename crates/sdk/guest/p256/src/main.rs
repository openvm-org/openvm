extern crate alloc;

use elliptic_curve::{ops::LinearCombination, CurveArithmetic, Field, Group, PrimeField};
use openvm_p256::{NistP256, P256Point, P256Scalar as Scalar};

openvm::init!();

pub fn main() {
    let g = P256Point::generator();
    let a = P256Point::lincomb(&g, &Scalar::from_u128(100), &g, &Scalar::from_u128(156));
    let mut b = g;
    for _ in 0..8 {
        b += b;
    }
    assert_eq!(a, b);

    type NistScalar = <NistP256 as CurveArithmetic>::Scalar;

    let a = NistScalar::from_u128(4);
    let b = a.sqrt().unwrap();
    assert!(b == NistScalar::from_u128(2) || b == -NistScalar::from_u128(2));

    let a = NistScalar::from_u128(5);
    let b = a.sqrt().unwrap();
    let sqrt_5 = NistScalar::from_str_vartime(
        "37706888570942939511621860890978929712654002332559277021296980149138421130241",
    )
    .unwrap();
    assert!(b == sqrt_5 || b == -sqrt_5);
    assert!(b * b == a);

    let a = NistScalar::from_u128(7);
    let b = a.sqrt();
    assert!(bool::from(b.is_none()));
}
