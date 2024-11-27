#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use axvm::io::read_vec;
use axvm_ecc::{pairing::PairingCheck, AffinePoint};

axvm::entry!(main);

mod bn254 {
    use axvm_ecc::bn254::{Bn254, Fp, Fp2};

    use super::*;

    pub fn test_pairing_check(io: &[u8]) {
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

mod bls12_381 {
    use axvm_ecc::bls12_381::{Bls12_381, Fp, Fp2};

    use super::*;

    pub fn test_pairing_check(io: &[u8]) {
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
    let io = read_vec();
    const BN254_SIZE: usize = 32 * 12;
    const BLS12_381_SIZE: usize = 48 * 12;

    match io.len() {
        BN254_SIZE => bn254::test_pairing_check(&io),
        BLS12_381_SIZE => bls12_381::test_pairing_check(&io),
        _ => panic!("Invalid input length"),
    }
}
