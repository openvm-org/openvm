#![feature(cfg_match)]
#![allow(unused_imports)]
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use openvm::io::read_vec;
use openvm_algebra_guest::{field::FieldExtension, IntMod};
use openvm_ecc_guest::AffinePoint;
use openvm_pairing_guest::pairing::MultiMillerLoop;

openvm::entry!(main);

#[cfg(feature = "bn254")]
mod bn254 {
    use openvm_pairing_guest::bn254::{Bn254, Fp, Fp2};

    use super::*;

    openvm_algebra_moduli_setup::moduli_init! {
        "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
        "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
    }

    openvm_algebra_complex_macros::complex_init! {
        Fp2 { mod_idx = 0 },
    }

    openvm_ecc_sw_setup::sw_init! {
        Fp,
    }

    pub fn test_miller_loop(io: &[u8]) {
        setup_0();
        setup_all_complex_extensions();
        let s0 = &io[0..32 * 2];
        let s1 = &io[32 * 2..32 * 4];
        let q0 = &io[32 * 4..32 * 8];
        let q1 = &io[32 * 8..32 * 12];
        let f_cmp = &io[32 * 12..32 * 24];

        let s0_cast = unsafe { &*(s0.as_ptr() as *const AffinePoint<Fp>) };
        let s1_cast = unsafe { &*(s1.as_ptr() as *const AffinePoint<Fp>) };
        let q0_cast = unsafe { &*(q0.as_ptr() as *const AffinePoint<Fp2>) };
        let q1_cast = unsafe { &*(q1.as_ptr() as *const AffinePoint<Fp2>) };

        let f = Bn254::multi_miller_loop(
            &[s0_cast.clone(), s1_cast.clone()],
            &[q0_cast.clone(), q1_cast.clone()],
        );
        let mut f_bytes = [0u8; 32 * 12];
        f.to_coeffs()
            .iter()
            .flat_map(|fp2| fp2.clone().to_coeffs())
            .enumerate()
            .for_each(|(i, fp)| f_bytes[i * 32..(i + 1) * 32].copy_from_slice(fp.as_le_bytes()));

        assert_eq!(f_bytes, f_cmp);
    }
}

#[cfg(feature = "bls12_381")]
mod bls12_381 {
    use openvm_pairing_guest::bls12_381::{Bls12_381, Fp, Fp2};

    use super::*;

    openvm_algebra_moduli_setup::moduli_init! {
        "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
        "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
    }

    openvm_algebra_complex_macros::complex_init! {
        Fp2 { mod_idx = 0 },
    }

    openvm_ecc_sw_setup::sw_init! {
        Fp,
    }

    pub fn test_miller_loop(io: &[u8]) {
        setup_0();
        setup_all_complex_extensions();
        let s0 = &io[0..48 * 2];
        let s1 = &io[48 * 2..48 * 4];
        let q0 = &io[48 * 4..48 * 8];
        let q1 = &io[48 * 8..48 * 12];
        let f_cmp = &io[48 * 12..48 * 24];

        let s0_cast = unsafe { &*(s0.as_ptr() as *const AffinePoint<Fp>) };
        let s1_cast = unsafe { &*(s1.as_ptr() as *const AffinePoint<Fp>) };
        let q0_cast = unsafe { &*(q0.as_ptr() as *const AffinePoint<Fp2>) };
        let q1_cast = unsafe { &*(q1.as_ptr() as *const AffinePoint<Fp2>) };

        let f = Bls12_381::multi_miller_loop(
            &[s0_cast.clone(), s1_cast.clone()],
            &[q0_cast.clone(), q1_cast.clone()],
        );
        let mut f_bytes = [0u8; 48 * 12];
        f.to_coeffs()
            .iter()
            .flat_map(|fp2| fp2.clone().to_coeffs())
            .enumerate()
            .for_each(|(i, fp)| f_bytes[i * 48..(i + 1) * 48].copy_from_slice(fp.as_le_bytes()));

        assert_eq!(f_bytes, f_cmp);
    }
}

pub fn main() {
    #[allow(unused_variables)]
    let io = read_vec();

    cfg_match! {
        cfg(feature = "bn254") => { bn254::test_miller_loop(&io); }
        cfg(feature = "bls12_381") => { bls12_381::test_miller_loop(&io); }
        _ => { panic!("No curve feature enabled") }
    }
}
