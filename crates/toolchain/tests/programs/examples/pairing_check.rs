#![feature(cfg_match)]
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use axvm::io::read_vec;
use axvm_ecc_guest::AffinePoint;
use axvm_pairing_guest::pairing::PairingCheck;

axvm::entry!(main);

#[cfg(feature = "bn254")]
mod bn254 {
    use alloc::format;

    use axvm_algebra_guest::IntMod;
    use axvm_pairing_guest::bn254::{Bn254, Fp, Fp2};

    use super::*;

    axvm_algebra_moduli_setup::moduli_init!(
        "21888242871839275222246405745257275088696311157297823662689037894645226208583"
    );

    axvm_algebra_complex_macros::complex_init! {
        Fp2 { mod_idx = 0 },
    }

    pub fn test_pairing_check(io: &[u8]) {
        setup_all_moduli();
        setup_all_complex_extensions();
        let s0 = &io[0..32 * 2];
        let s1 = &io[32 * 2..32 * 4];
        let q0 = &io[32 * 4..32 * 8];
        let q1 = &io[32 * 8..32 * 12];

        let s0_cast = unsafe { &*(s0.as_ptr() as *const AffinePoint<Fp>) };
        let s1_cast = unsafe { &*(s1.as_ptr() as *const AffinePoint<Fp>) };
        let q0_cast = unsafe { &*(q0.as_ptr() as *const AffinePoint<Fp2>) };
        let q1_cast = unsafe { &*(q1.as_ptr() as *const AffinePoint<Fp2>) };

        let f = Bn254::pairing_check(
            &[s0_cast.clone(), s1_cast.clone()],
            &[q0_cast.clone(), q1_cast.clone()],
        );
        assert_eq!(f, Ok(()));
    }
}

#[cfg(feature = "bls12_381")]
mod bls12_381 {

    use alloc::format;

    use axvm_algebra_guest::IntMod;
    use axvm_pairing_guest::bls12_381::{Bls12_381, Fp, Fp2};

    use super::*;

    axvm_algebra_moduli_setup::moduli_init!("0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");

    axvm_algebra_complex_macros::complex_init! {
        Fp2 { mod_idx = 0 },
    }

    pub fn test_pairing_check(io: &[u8]) {
        setup_all_moduli();
        setup_all_complex_extensions();
        let s0 = &io[0..48 * 2];
        let s1 = &io[48 * 2..48 * 4];
        let q0 = &io[48 * 4..48 * 8];
        let q1 = &io[48 * 8..48 * 12];

        let s0_cast = unsafe { &*(s0.as_ptr() as *const AffinePoint<Fp>) };
        let s1_cast = unsafe { &*(s1.as_ptr() as *const AffinePoint<Fp>) };
        let q0_cast = unsafe { &*(q0.as_ptr() as *const AffinePoint<Fp2>) };
        let q1_cast = unsafe { &*(q1.as_ptr() as *const AffinePoint<Fp2>) };

        let f = Bls12_381::pairing_check(
            &[s0_cast.clone(), s1_cast.clone()],
            &[q0_cast.clone(), q1_cast.clone()],
        );
        assert_eq!(f, Ok(()));
    }
}

pub fn main() {
    #[allow(unused_variables)]
    let io = read_vec();

    cfg_match! {
        cfg(feature = "bn254") => { bn254::test_pairing_check(&io); }
        cfg(feature = "bls12_381") => { bls12_381::test_pairing_check(&io); }
        _ => { panic!("No curve feature enabled") }
    }
}
