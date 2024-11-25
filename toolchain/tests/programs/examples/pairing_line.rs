#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::io::read_vec;
use axvm_algebra::{field::FieldExtension, IntMod};
use axvm_ecc::pairing::{EvaluatedLine, LineMulDType, LineMulMType};

axvm::entry!(main);

mod bn254 {
    use axvm_ecc::bn254::{Bn254, Fp12, Fp2};

    use super::*;

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

mod bls12_381 {
    use axvm_ecc::bls12_381::{Bls12_381, Fp12, Fp2};

    use super::*;

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
    let io = read_vec();
    const BN254_SIZE: usize = 32 * 52;
    const BLS12_381_SIZE: usize = 48 * 52;

    match io.len() {
        BN254_SIZE => {
            bn254::test_mul_013_by_013(&io[..32 * 18]);
            bn254::test_mul_by_01234(&io[32 * 18..32 * 52]);
        }
        BLS12_381_SIZE => {
            bls12_381::test_mul_023_by_023(&io[..48 * 18]);
            bls12_381::test_mul_by_02345(&io[48 * 18..48 * 52]);
        }
        _ => panic!("Invalid input length"),
    }
}
