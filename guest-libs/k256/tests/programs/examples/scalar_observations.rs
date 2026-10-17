#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

use elliptic_curve::{bigint::U256, FieldBytes, PrimeField};
use openvm_algebra_guest::IntMod;
#[allow(unused)]
use openvm_k256::Secp256k1Point;
use openvm_k256::{Secp256k1, Secp256k1Scalar as Scalar};

openvm::init!("openvm_init_scalar_observations.rs");
openvm::entry!(main);

pub fn main() {
    let case = openvm::io::read_vec()[0];
    let scalar = if case == 0 {
        Scalar::from_u32(42)
    } else {
        Scalar::from_le_bytes_unchecked(&<Scalar as IntMod>::MODULUS)
    };
    match case {
        0 => {
            let mut expected = [0u8; 32];
            expected[31] = 42;
            assert_eq!(&scalar.to_bytes()[..], expected);
            assert_eq!(&scalar.to_repr()[..], expected);
            assert_eq!(scalar.is_odd().unwrap_u8(), 0);
            assert_eq!(U256::from(scalar), U256::from_be_slice(&expected));
            assert_eq!(&FieldBytes::<Secp256k1>::from(scalar)[..], expected);
        }
        1 => {
            black_box(scalar.to_bytes());
        }
        2 => {
            black_box(scalar.to_repr());
        }
        3 => {
            black_box(scalar.is_odd());
        }
        4 => {
            black_box(U256::from(scalar));
        }
        5 => {
            black_box(FieldBytes::<Secp256k1>::from(scalar));
        }
        6 => {
            black_box(scalar.as_le_bytes());
        }
        7 => {
            black_box(scalar.to_be_bytes());
        }
        8 => {
            black_box(openvm::serde::to_vec(&scalar).unwrap());
        }
        _ => panic!("unknown case"),
    }
}
