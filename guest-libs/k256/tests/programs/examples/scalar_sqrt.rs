#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use elliptic_curve::{CurveArithmetic, Field, PrimeField};
use openvm_k256::Secp256k1;
// clippy thinks this is unused, but it's used in the init! macro
#[allow(unused)]
use openvm_k256::Secp256k1Point;
openvm::init!("openvm_init_scalar_sqrt.rs");

openvm::entry!(main);

pub fn main() {
    type Scalar = <Secp256k1 as CurveArithmetic>::Scalar;

    let a = Scalar::from_u128(4);
    let b = a.sqrt().unwrap();
    assert!(b == Scalar::from_u128(2) || b == -Scalar::from_u128(2));

    let a = Scalar::from_u128(2);
    let b = a.sqrt().unwrap();
    let sqrt_2 = Scalar::from_str_vartime(
        "2823942750030662837874242031155578187138543190171473581917399304008038956128",
    )
    .unwrap();
    assert!(b == sqrt_2 || b == -sqrt_2);
    assert!(b * b == a);

    let a = Scalar::from_u128(5);
    let b = a.sqrt();
    assert!(bool::from(b.is_none()));
}
