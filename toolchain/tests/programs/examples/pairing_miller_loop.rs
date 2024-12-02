#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use axvm::io::read_vec;
use axvm_algebra::{field::FieldExtension, IntMod};
use axvm_ecc::{pairing::MultiMillerLoop, AffinePoint};

axvm::entry!(main);

mod bn254 {
    use axvm_ecc::bn254::{Bn254, Fp, Fp2};

    use super::*;

    pub fn test_miller_loop(io: &[u8]) {
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

mod bls12_381 {
    use axvm_ecc::bls12_381::{Bls12_381, Fp, Fp2};

    use super::*;

    pub fn test_miller_loop(io: &[u8]) {
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
    let io = read_vec();
    const BN254_SIZE: usize = 32 * 24;
    const BLS12_381_SIZE: usize = 48 * 24;

    match io.len() {
        BN254_SIZE => bn254::test_miller_loop(&io),
        BLS12_381_SIZE => bls12_381::test_miller_loop(&io),
        _ => panic!("Invalid input length"),
    }
}
