#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
openvm::entry!(main);

extern crate alloc;

use alloc::{vec, vec::Vec};

use openvm_ff_derive::openvm_prime_field;

/// The BLS12-381 scalar field.
#[openvm_prime_field]
#[PrimeFieldModulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
#[PrimeFieldGenerator = "7"]
#[PrimeFieldReprEndianness = "little"]
struct Bls381K12Scalar([u64; 4]);

openvm::init!("openvm_init_batch_inversion_std.rs");

fn main() {
    use ff::{BatchInverter, Field};

    let one = Bls381K12Scalar::ONE;

    // [1, 2, 3, 4]
    let values: Vec<_> = (0..4)
        .scan(one, |acc, _| {
            let ret = *acc;
            *acc += &one;
            Some(ret)
        })
        .collect();

    // Test BatchInverter::invert_with_external_scratch
    {
        let mut elements = values.clone();
        let mut scratch_space = vec![Bls381K12Scalar::ZERO; elements.len()];
        BatchInverter::invert_with_external_scratch(&mut elements, &mut scratch_space);
        for (a, a_inv) in values.iter().zip(elements.into_iter()) {
            assert_eq!(*a * a_inv, one);
        }
    }

    // Test BatchInverter::invert_with_internal_scratch
    {
        let mut items: Vec<_> = values.iter().cloned().map(|p| (p, one)).collect();
        BatchInverter::invert_with_internal_scratch(
            &mut items,
            |item| &mut item.0,
            |item| &mut item.1,
        );
        for (a, (a_inv, _)) in values.iter().zip(items.into_iter()) {
            assert_eq!(*a * a_inv, one);
        }
    }

    // Test BatchInvert trait
    #[cfg(feature = "std")]
    {
        use ff::BatchInvert;
        let mut elements = values.clone();
        elements.iter_mut().batch_invert();
        for (a, a_inv) in values.iter().zip(elements.into_iter()) {
            assert_eq!(*a * a_inv, one);
        }
    }
}
