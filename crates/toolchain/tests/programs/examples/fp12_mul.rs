#![feature(cfg_match)]
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::io::read_vec;
use axvm_algebra_guest::{field::FieldExtension, IntMod};

axvm::entry!(main);

#[cfg(feature = "bn254")]
mod bn254 {
    use axvm_pairing_guest::bn254::Fp12;

    use super::*;

    axvm_algebra_moduli_setup::moduli_init! {
        "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
        "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
    }

    pub fn test_fp12_mul(io: &[u8]) {
        setup_0();
        assert_eq!(io.len(), 32 * 36);

        let f0 = &io[0..32 * 12];
        let f1 = &io[32 * 12..32 * 24];
        let r_cmp = &io[32 * 24..32 * 36];

        let f0_cast = unsafe { &*(f0.as_ptr() as *const Fp12) };
        let f1_cast = unsafe { &*(f1.as_ptr() as *const Fp12) };

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
    use axvm_pairing_guest::bls12_381::Fp12;

    use super::*;

    axvm_algebra_moduli_setup::moduli_init! {
        "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
        "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
    }

    pub fn test_fp12_mul(io: &[u8]) {
        setup_0();
        assert_eq!(io.len(), 48 * 36);

        let f0 = &io[0..48 * 12];
        let f1 = &io[48 * 12..48 * 24];
        let r_cmp = &io[48 * 24..48 * 36];

        let f0_cast = unsafe { &*(f0.as_ptr() as *const Fp12) };
        let f1_cast = unsafe { &*(f1.as_ptr() as *const Fp12) };

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

    cfg_match! {
        cfg(feature = "bn254") => { bn254::test_fp12_mul(&io) }
        cfg(feature = "bls12_381") => { bls12_381::test_fp12_mul(&io) }
        _ => { panic!("No curve feature enabled") }
    }
}
