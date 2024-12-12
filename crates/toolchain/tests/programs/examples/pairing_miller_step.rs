#![feature(cfg_match)]
#![allow(unused_imports)]
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::io::read_vec;
use axvm_algebra_guest::IntMod;
use axvm_ecc_guest::AffinePoint;
use axvm_pairing_guest::pairing::MillerStep;

axvm::entry!(main);

#[cfg(feature = "bn254")]
mod bn254 {
    use axvm_pairing_guest::bn254::{Bn254, Fp, Fp2};

    use super::*;

    axvm_algebra_moduli_setup::moduli_init! {
        "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
        "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
    }

    axvm_algebra_complex_macros::complex_init! {
        Fp2 { mod_idx = 0 },
    }

    axvm_ecc_sw_setup::sw_init! {
        Fp,
    }

    pub fn test_miller_step(io: &[u8]) {
        assert_eq!(io.len(), 32 * 12);
        let s = &io[..32 * 4];
        let pt = &io[32 * 4..32 * 8];
        let l = &io[32 * 8..32 * 12];

        let s_cast = unsafe { &*(s.as_ptr() as *const AffinePoint<Fp2>) };

        let (pt_cmp, l_cmp) = Bn254::miller_double_step(s_cast);
        let mut pt_bytes = [0u8; 32 * 4];
        let mut l_bytes = [0u8; 32 * 4];

        // TODO: if we ever need to change this, we should switch to using `bincode` to serialize
        //       for us and use `read()` instead of `read_vec()`
        pt_bytes[0..32].copy_from_slice(pt_cmp.x.c0.as_le_bytes());
        pt_bytes[32..2 * 32].copy_from_slice(pt_cmp.x.c1.as_le_bytes());
        pt_bytes[2 * 32..3 * 32].copy_from_slice(pt_cmp.y.c0.as_le_bytes());
        pt_bytes[3 * 32..4 * 32].copy_from_slice(pt_cmp.y.c1.as_le_bytes());
        l_bytes[0..32].copy_from_slice(l_cmp.b.c0.as_le_bytes());
        l_bytes[32..2 * 32].copy_from_slice(l_cmp.b.c1.as_le_bytes());
        l_bytes[2 * 32..3 * 32].copy_from_slice(l_cmp.c.c0.as_le_bytes());
        l_bytes[3 * 32..4 * 32].copy_from_slice(l_cmp.c.c1.as_le_bytes());

        assert_eq!(pt_bytes, pt);
        assert_eq!(l_bytes, l);
    }

    pub fn test_miller_double_and_add_step(io: &[u8]) {
        assert_eq!(io.len(), 32 * 20);
        let s = &io[0..32 * 4];
        let q = &io[32 * 4..32 * 8];
        let pt = &io[32 * 8..32 * 12];
        let l0 = &io[32 * 12..32 * 16];
        let l1 = &io[32 * 16..32 * 20];

        let s_cast = unsafe { &*(s.as_ptr() as *const AffinePoint<Fp2>) };
        let q_cast = unsafe { &*(q.as_ptr() as *const AffinePoint<Fp2>) };
        let (pt_cmp, l0_cmp, l1_cmp) = Bn254::miller_double_and_add_step(s_cast, q_cast);
        let mut pt_bytes = [0u8; 32 * 4];
        let mut l0_bytes = [0u8; 32 * 4];
        let mut l1_bytes = [0u8; 32 * 4];

        // TODO: if we ever need to change this, we should switch to using `bincode` to serialize
        //       for us and use `read()` instead of `read_vec()`
        pt_bytes[0..32].copy_from_slice(pt_cmp.x.c0.as_le_bytes());
        pt_bytes[32..2 * 32].copy_from_slice(pt_cmp.x.c1.as_le_bytes());
        pt_bytes[2 * 32..3 * 32].copy_from_slice(pt_cmp.y.c0.as_le_bytes());
        pt_bytes[3 * 32..4 * 32].copy_from_slice(pt_cmp.y.c1.as_le_bytes());
        l0_bytes[0..32].copy_from_slice(l0_cmp.b.c0.as_le_bytes());
        l0_bytes[32..2 * 32].copy_from_slice(l0_cmp.b.c1.as_le_bytes());
        l0_bytes[2 * 32..3 * 32].copy_from_slice(l0_cmp.c.c0.as_le_bytes());
        l0_bytes[3 * 32..4 * 32].copy_from_slice(l0_cmp.c.c1.as_le_bytes());
        l1_bytes[0..32].copy_from_slice(l1_cmp.b.c0.as_le_bytes());
        l1_bytes[32..2 * 32].copy_from_slice(l1_cmp.b.c1.as_le_bytes());
        l1_bytes[2 * 32..3 * 32].copy_from_slice(l1_cmp.c.c0.as_le_bytes());
        l1_bytes[3 * 32..4 * 32].copy_from_slice(l1_cmp.c.c1.as_le_bytes());

        assert_eq!(pt_bytes, pt);
        assert_eq!(l0_bytes, l0);
        assert_eq!(l1_bytes, l1);
    }
}

#[cfg(feature = "bls12_381")]
mod bls12_381 {
    use axvm_pairing_guest::bls12_381::{Bls12_381, Fp, Fp2};

    use super::*;

    axvm_algebra_moduli_setup::moduli_init! {
        "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
        "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
    }

    axvm_algebra_complex_macros::complex_init! {
        Fp2 { mod_idx = 0 },
    }

    axvm_ecc_sw_setup::sw_init! {
        Fp,
    }

    pub fn test_miller_step(io: &[u8]) {
        assert_eq!(io.len(), 48 * 12);
        let s = &io[..48 * 4];
        let pt = &io[48 * 4..48 * 8];
        let l = &io[48 * 8..48 * 12];

        let s_cast = unsafe { &*(s.as_ptr() as *const AffinePoint<Fp2>) };

        let (pt_cmp, l_cmp) = Bls12_381::miller_double_step(s_cast);
        let mut pt_bytes = [0u8; 48 * 4];
        let mut l_bytes = [0u8; 48 * 4];

        pt_bytes[0..48].copy_from_slice(pt_cmp.x.c0.as_le_bytes());
        pt_bytes[48..2 * 48].copy_from_slice(pt_cmp.x.c1.as_le_bytes());
        pt_bytes[2 * 48..3 * 48].copy_from_slice(pt_cmp.y.c0.as_le_bytes());
        pt_bytes[3 * 48..4 * 48].copy_from_slice(pt_cmp.y.c1.as_le_bytes());
        l_bytes[0..48].copy_from_slice(l_cmp.b.c0.as_le_bytes());
        l_bytes[48..2 * 48].copy_from_slice(l_cmp.b.c1.as_le_bytes());
        l_bytes[2 * 48..3 * 48].copy_from_slice(l_cmp.c.c0.as_le_bytes());
        l_bytes[3 * 48..4 * 48].copy_from_slice(l_cmp.c.c1.as_le_bytes());

        assert_eq!(pt_bytes, pt);
        assert_eq!(l_bytes, l);
    }

    pub fn test_miller_double_and_add_step(io: &[u8]) {
        assert_eq!(io.len(), 48 * 20);
        let s = &io[0..48 * 4];
        let q = &io[48 * 4..48 * 8];
        let pt = &io[48 * 8..48 * 12];
        let l0 = &io[48 * 12..48 * 16];
        let l1 = &io[48 * 16..48 * 20];

        let s_cast = unsafe { &*(s.as_ptr() as *const AffinePoint<Fp2>) };
        let q_cast = unsafe { &*(q.as_ptr() as *const AffinePoint<Fp2>) };
        let (pt_cmp, l0_cmp, l1_cmp) = Bls12_381::miller_double_and_add_step(s_cast, q_cast);
        let mut pt_bytes = [0u8; 48 * 4];
        let mut l0_bytes = [0u8; 48 * 4];
        let mut l1_bytes = [0u8; 48 * 4];

        pt_bytes[0..48].copy_from_slice(pt_cmp.x.c0.as_le_bytes());
        pt_bytes[48..2 * 48].copy_from_slice(pt_cmp.x.c1.as_le_bytes());
        pt_bytes[2 * 48..3 * 48].copy_from_slice(pt_cmp.y.c0.as_le_bytes());
        pt_bytes[3 * 48..4 * 48].copy_from_slice(pt_cmp.y.c1.as_le_bytes());
        l0_bytes[0..48].copy_from_slice(l0_cmp.b.c0.as_le_bytes());
        l0_bytes[48..2 * 48].copy_from_slice(l0_cmp.b.c1.as_le_bytes());
        l0_bytes[2 * 48..3 * 48].copy_from_slice(l0_cmp.c.c0.as_le_bytes());
        l0_bytes[3 * 48..4 * 48].copy_from_slice(l0_cmp.c.c1.as_le_bytes());
        l1_bytes[0..48].copy_from_slice(l1_cmp.b.c0.as_le_bytes());
        l1_bytes[48..2 * 48].copy_from_slice(l1_cmp.b.c1.as_le_bytes());
        l1_bytes[2 * 48..3 * 48].copy_from_slice(l1_cmp.c.c0.as_le_bytes());
        l1_bytes[3 * 48..4 * 48].copy_from_slice(l1_cmp.c.c1.as_le_bytes());

        assert_eq!(pt_bytes, pt);
        assert_eq!(l0_bytes, l0);
        assert_eq!(l1_bytes, l1);
    }
}

pub fn main() {
    #[allow(unused_variables)]
    let io = read_vec();

    cfg_match! {
        cfg(feature = "bn254") => {
            bn254::setup_0();
            bn254::setup_all_complex_extensions();
            bn254::test_miller_step(&io[..32 * 12]);
            bn254::test_miller_double_and_add_step(&io[32 * 12..]);
        }
        cfg(feature = "bls12_381") => {
            bls12_381::setup_0();
            bls12_381::setup_all_complex_extensions();
            bls12_381::test_miller_step(&io[..48 * 12]);
            bls12_381::test_miller_double_and_add_step(&io[48 * 12..]);
        }
        _ => { panic!("No curve feature enabled") }
    }
}
