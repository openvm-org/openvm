#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use elliptic_curve::{group::Curve, CurveArithmetic, Group};
use openvm_p256::NistP256;
// clippy thinks this is unused, but it's used in the init! macro
#[allow(unused)]
use openvm_p256::P256Point;

openvm::init!("openvm_init_add.rs");

openvm::entry!(main);

mod test_vectors;
use test_vectors::ADD_TEST_VECTORS;

pub fn main() {
    let generator = <NistP256 as CurveArithmetic>::ProjectivePoint::generator();
    let mut p = generator;

    for test_vector in ADD_TEST_VECTORS {
        let affine = p.to_affine();

        let (expected_x, expected_y) = test_vector;
        assert_eq!(&affine.x_be_bytes(), expected_x);
        assert_eq!(&affine.y_be_bytes(), expected_y);

        p += &generator;
    }
}
