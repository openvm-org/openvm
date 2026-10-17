#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

#[allow(unused_imports)]
#[cfg(feature = "bn254")]
use openvm_pairing::bn254::Bn254G1Affine;

openvm::init!("openvm_init_bn_ec_bn254.rs");
openvm::entry!(main);

pub fn main() {
    #[cfg(feature = "bn254")]
    {
        extern crate alloc;
        use alloc::vec;

        use hex_literal::hex;
        use openvm_algebra_guest::IntMod;
        use openvm_ecc_guest::{
            weierstrass::{IntrinsicCurve, WeierstrassPoint},
            CyclicGroup,
        };
        use openvm_pairing::bn254::{Bn254, G1Affine, G2Affine, Scalar};

        assert_eq!(
            G2Affine::IDENTITY.as_le_bytes(),
            G2Affine::IDENTITY.as_le_bytes_unchecked()
        );

        let mut n_plus_one = Scalar::MODULUS;
        n_plus_one[0] += 1;
        for (bytes, expected) in [
            (Scalar::MODULUS, Scalar::ZERO),
            (n_plus_one, Scalar::ONE),
            (
                [255; 32],
                Scalar::from_repr(hex!(
                    "faffff4f1c3496ac29cd609f9576fc362e4679786fa36e662fdf079ac1770a0e"
                )),
            ),
        ] {
            // Exercise both the windowed and Pippenger paths.
            for len in [1, 25] {
                let mut scalars = vec![Scalar::ZERO; len];
                scalars[0] = Scalar::from_repr(bytes);
                let bases = vec![G1Affine::GENERATOR; len];
                assert_eq!(
                    Bn254::msm(&scalars, &bases),
                    openvm_ecc_guest::msm(core::slice::from_ref(&expected), &[G1Affine::GENERATOR])
                );
            }
        }
    }
}
