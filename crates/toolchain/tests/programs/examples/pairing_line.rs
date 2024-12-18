#![feature(cfg_match)]
#![allow(unused_imports)]
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::read_vec;
use openvm_algebra_guest::{field::FieldExtension, IntMod};
use openvm_pairing_guest::pairing::{EvaluatedLine, LineMulDType, LineMulMType};

openvm::entry!(main);

#[cfg(feature = "bn254")]
mod bn254 {
    use openvm_pairing_guest::bn254::{Bn254, Fp, Fp12, Fp2};

    use super::*;

    openvm_algebra_moduli_setup::moduli_init! {
        "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
        "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
    }

    pub fn test_mul_013_by_013(io: &[u8]) {
        assert_eq!(io.len(), 32 * 18);
        let l0 = &io[..32 * 4];
        let l1 = &io[32 * 4..32 * 8];
        let expected = &io[32 * 8..32 * 18];

        let l0_cast = unsafe { &*(l0.as_ptr() as *const EvaluatedLine<Fp2>) };
        let l1_cast = unsafe { &*(l1.as_ptr() as *const EvaluatedLine<Fp2>) };

        let r = Bn254::mul_013_by_013(l0_cast, l1_cast);
        let mut r_bytes = [0u8; 32 * 10];
        let mut i = 0;
        for x in r {
            r_bytes[i..i + 32].copy_from_slice(x.c0.as_le_bytes());
            r_bytes[i + 32..i + 64].copy_from_slice(x.c1.as_le_bytes());
            i += 64;
        }
        assert_eq!(r_bytes, expected);
    }

    pub fn test_mul_by_01234(io: &[u8]) {
        assert_eq!(io.len(), 32 * 34);
        let f = &io[..32 * 12];
        let x = &io[32 * 12..32 * 22];
        let expected = &io[32 * 22..32 * 34];

        let f_cast = unsafe { &*(f.as_ptr() as *const Fp12) };
        let x_cast = unsafe { &*(x.as_ptr() as *const [Fp2; 5]) };

        let r = Bn254::mul_by_01234(f_cast, x_cast);
        let mut r_bytes = [0u8; 32 * 12];
        let mut i = 0;
        for x in r.to_coeffs() {
            r_bytes[i..i + 32].copy_from_slice(x.c0.as_le_bytes());
            r_bytes[i + 32..i + 64].copy_from_slice(x.c1.as_le_bytes());
            i += 64;
        }
        assert_eq!(r_bytes, expected);
    }
}

#[cfg(feature = "bls12_381")]
mod bls12_381 {
    use openvm_pairing_guest::bls12_381::{Bls12_381, Fp, Fp12, Fp2};

    use super::*;

    openvm_algebra_moduli_setup::moduli_init! {
        "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
        "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
    }

    pub fn test_mul_023_by_023(io: &[u8]) {
        assert_eq!(io.len(), 48 * 18);
        let l0 = &io[..48 * 4];
        let l1 = &io[48 * 4..48 * 8];
        let expected = &io[48 * 8..48 * 18];

        let l0_cast = unsafe { &*(l0.as_ptr() as *const EvaluatedLine<Fp2>) };
        let l1_cast = unsafe { &*(l1.as_ptr() as *const EvaluatedLine<Fp2>) };

        let r = Bls12_381::mul_023_by_023(l0_cast, l1_cast);
        let mut r_bytes = [0u8; 48 * 10];
        let mut i = 0;
        for x in r {
            r_bytes[i..i + 48].copy_from_slice(x.c0.as_le_bytes());
            r_bytes[i + 48..i + 96].copy_from_slice(x.c1.as_le_bytes());
            i += 96;
        }
        assert_eq!(r_bytes, expected);
    }

    pub fn test_mul_by_02345(io: &[u8]) {
        assert_eq!(io.len(), 48 * 34);
        let f = &io[..48 * 12];
        let x = &io[48 * 12..48 * 22];
        let expected = &io[48 * 22..48 * 34];

        let f_cast = unsafe { &*(f.as_ptr() as *const Fp12) };
        let x_cast = unsafe { &*(x.as_ptr() as *const [Fp2; 5]) };

        let r = Bls12_381::mul_by_02345(f_cast, x_cast);
        let mut r_bytes = [0u8; 48 * 12];
        let mut i = 0;
        for x in r.to_coeffs() {
            r_bytes[i..i + 48].copy_from_slice(x.c0.as_le_bytes());
            r_bytes[i + 48..i + 96].copy_from_slice(x.c1.as_le_bytes());
            i += 96;
        }
        assert_eq!(r_bytes, expected);
    }
}

pub fn main() {
    #[allow(unused_variables)]
    let io = read_vec();

    cfg_match! {
        cfg(feature = "bn254") => {
            bn254::setup_0();
            bn254::test_mul_013_by_013(&io[..32 * 18]);
            bn254::test_mul_by_01234(&io[32 * 18..32 * 52]);
        }
        cfg(feature = "bls12_381") => {
            bls12_381::setup_0();
            bls12_381::test_mul_023_by_023(&io[..48 * 18]);
            bls12_381::test_mul_by_02345(&io[48 * 18..48 * 52]);
        }
        _ => { panic!("No curve feature enabled") }
    }
}
