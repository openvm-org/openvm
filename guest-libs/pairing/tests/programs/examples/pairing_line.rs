#![allow(unused_imports)]
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::read_vec;
use openvm_algebra_guest::{field::FieldExtension, IntMod};
use openvm_pairing_guest::pairing::{EvaluatedLine, LineMulDType, LineMulMType};

openvm::entry!(main);

#[cfg(feature = "bn254")]
mod bn254 {
    use openvm_pairing::bn254::{Bn254, Fp, Fp12, Fp2};

    use super::*;

    openvm::init!("openvm_init_pairing_line_bn254.rs");

    pub fn test_mul_013_by_013(io: &[u8]) {
        assert_eq!(io.len(), 32 * 18);
        let l0 = &io[..32 * 4];
        let l1 = &io[32 * 4..32 * 8];
        let expected = &io[32 * 8..32 * 18];

        let l0_cast = EvaluatedLine {
            b: Fp2::from_bytes(&l0[..64]),
            c: Fp2::from_bytes(&l0[64..128]),
        };
        let l1_cast = EvaluatedLine {
            b: Fp2::from_bytes(&l1[..64]),
            c: Fp2::from_bytes(&l1[64..128]),
        };

        let r = Bn254::mul_013_by_013(&l0_cast, &l1_cast);
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

        let f_cast = Fp12::from_bytes(f);
        let x_cast = [
            Fp2::from_bytes(&x[..64]),
            Fp2::from_bytes(&x[64..128]),
            Fp2::from_bytes(&x[128..192]),
            Fp2::from_bytes(&x[192..256]),
            Fp2::from_bytes(&x[256..320]),
        ];

        let r = Bn254::mul_by_01234(&f_cast, &x_cast);
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
    use openvm_pairing::bls12_381::{Bls12_381, Fp, Fp12, Fp2};

    use super::*;

    openvm::init!("openvm_init_pairing_line_bls12_381.rs");

    pub fn test_mul_023_by_023(io: &[u8]) {
        assert_eq!(io.len(), 48 * 18);
        let l0 = &io[..48 * 4];
        let l1 = &io[48 * 4..48 * 8];
        let expected = &io[48 * 8..48 * 18];

        let l0_cast = EvaluatedLine {
            b: Fp2::from_bytes(&l0[..48 * 2]),
            c: Fp2::from_bytes(&l0[48 * 2..48 * 4]),
        };
        let l1_cast = EvaluatedLine {
            b: Fp2::from_bytes(&l1[..48 * 2]),
            c: Fp2::from_bytes(&l1[48 * 2..48 * 4]),
        };

        let r = Bls12_381::mul_023_by_023(&l0_cast, &l1_cast);
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

        let f_cast = Fp12::from_bytes(f);
        let x_cast = [
            Fp2::from_bytes(&x[..48 * 2]),
            Fp2::from_bytes(&x[48 * 2..48 * 4]),
            Fp2::from_bytes(&x[48 * 4..48 * 6]),
            Fp2::from_bytes(&x[48 * 6..48 * 8]),
            Fp2::from_bytes(&x[48 * 8..48 * 10]),
        ];

        let r = Bls12_381::mul_by_02345(&f_cast, &x_cast);
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

    #[cfg(feature = "bn254")]
    {
        bn254::test_mul_013_by_013(&io[..32 * 18]);
        bn254::test_mul_by_01234(&io[32 * 18..32 * 52]);
    }
    #[cfg(feature = "bls12_381")]
    {
        bls12_381::test_mul_023_by_023(&io[..48 * 18]);
        bls12_381::test_mul_by_02345(&io[48 * 18..48 * 52]);
    }
}
