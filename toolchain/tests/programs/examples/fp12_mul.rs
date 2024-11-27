#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::io::read_vec;
use axvm_algebra::{field::FieldExtension, IntMod};

axvm::entry!(main);

mod bn254 {
    use axvm_ecc::bn254::Fp12;

    use super::*;

    pub fn test_fp12_mul(io: &[u8]) {
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

mod bls12_381 {
    use axvm_ecc::bls12_381::Fp12;

    use super::*;

    pub fn test_fp12_mul(io: &[u8]) {
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
    let io = read_vec();
    const BN254_SIZE: usize = 32 * 36;
    const BLS12_381_SIZE: usize = 48 * 36;

    match io.len() {
        BN254_SIZE => bn254::test_fp12_mul(&io),
        BLS12_381_SIZE => bls12_381::test_fp12_mul(&io),
        _ => panic!("Invalid input size"),
    }
}
