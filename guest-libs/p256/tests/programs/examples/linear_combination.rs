#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use elliptic_curve::{ops::LinearCombination, Group, PrimeField};
// clippy thinks this is unused, but it's used in the init! macro
#[allow(unused)]
use openvm_p256::{P256Point, P256Point as ProjectivePoint, P256Scalar as Scalar};

openvm::init!("openvm_init_linear_combination.rs");

openvm::entry!(main);

pub fn main() {
    let g = ProjectivePoint::generator();
    let a = ProjectivePoint::lincomb(&g, &Scalar::from_u128(100), &g, &Scalar::from_u128(156));
    let mut b = g;
    for _ in 0..8 {
        b += b;
    }
    assert_eq!(a, b);
}
