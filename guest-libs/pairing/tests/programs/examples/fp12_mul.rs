#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unused_imports)]

use openvm::io::read_vec;
use openvm_algebra_guest::{field::FieldExtension, IntMod};

openvm::entry!(main);

#[cfg(feature = "bn254")]
mod bn254 {
    use openvm_pairing::bn254::{Fp, Fp12};

    use super::*;

    openvm::init!("openvm_init_fp12_mul_bn254.rs");

    pub fn test_fp12_mul(io: &[u8]) {
        assert_eq!(io.len(), 32 * 36);

        let f0 = &io[0..32 * 12];
        let f1 = &io[32 * 12..32 * 24];
        let r_cmp = &io[32 * 24..32 * 36];

        let f0_cast = Fp12::from_bytes(f0);
        let f1_cast = Fp12::from_bytes(f1);

        let r = f0_cast * f1_cast;
        let mut r_bytes = [0u8; 32 * 12];
        r.to_coeffs()
            .iter()
            .flat_map(|fp2| fp2.clone().to_coeffs())
            .enumerate()
            .for_each(|(i, fp)| r_bytes[i * 32..(i + 1) * 32].copy_from_slice(fp.as_le_bytes()));

        assert_eq!(r_bytes, r_cmp);
    }
}

#[cfg(feature = "bls12_381")]
mod bls12_381 {
    use openvm_pairing::bls12_381::{Fp, Fp12};

    use super::*;

    openvm::init!("openvm_init_fp12_mul_bls12_381.rs");

    pub fn test_fp12_mul(io: &[u8]) {
        assert_eq!(io.len(), 48 * 36);

        let f0 = &io[0..48 * 12];
        let f1 = &io[48 * 12..48 * 24];
        let r_cmp = &io[48 * 24..48 * 36];

        let f0_cast = Fp12::from_bytes(f0);
        let f1_cast = Fp12::from_bytes(f1);

        let r = f0_cast * f1_cast;
        let mut r_bytes = [0u8; 48 * 12];
        r.to_coeffs()
            .iter()
            .flat_map(|fp2| fp2.clone().to_coeffs())
            .enumerate()
            .for_each(|(i, fp)| r_bytes[i * 48..(i + 1) * 48].copy_from_slice(fp.as_le_bytes()));

        assert_eq!(r_bytes, r_cmp);
    }
}

pub fn main() {
    #[allow(unused_variables)]
    let io = read_vec();

    #[cfg(feature = "bn254")]
    {
        bn254::test_fp12_mul(&io)
    }
    #[cfg(feature = "bls12_381")]
    {
        bls12_381::test_fp12_mul(&io)
    }
}
