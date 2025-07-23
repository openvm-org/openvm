#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use elliptic_curve::{group::Curve, CurveArithmetic, Group};
use openvm_k256::Secp256k1;
// clippy thinks this is unused, but it's used in the init! macro
#[allow(unused)]
use openvm_k256::Secp256k1Point;

mod test_vectors;
use test_vectors::ADD_TEST_VECTORS;

openvm::init!("openvm_init_add.rs");

openvm::entry!(main);

// Taken from https://github.com/RustCrypto/elliptic-curves/blob/32343a78f1522aa5bd856556f114053d4bb938e0/k256/src/arithmetic/projective.rs#L797
pub fn main() {
    let generator = <Secp256k1 as CurveArithmetic>::ProjectivePoint::generator();
    let mut p = generator;

    for test_vector in ADD_TEST_VECTORS {
        let affine = p.to_affine();

        let (expected_x, expected_y) = test_vector;
        assert_eq!(&affine.x_be_bytes(), expected_x);
        assert_eq!(&affine.y_be_bytes(), expected_y);

        p += &generator;
    }
}
