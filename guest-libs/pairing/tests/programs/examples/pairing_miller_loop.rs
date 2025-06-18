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
    use openvm_pairing::bn254::{Bn254, Fp, Fp2};

    use super::*;

    openvm::init!("openvm_init_pairing_miller_loop_bn254.rs");

    pub fn test_miller_loop(io: &[u8]) {
        let s0 = &io[0..32 * 2];
        let s1 = &io[32 * 2..32 * 4];
        let q0 = &io[32 * 4..32 * 8];
        let q1 = &io[32 * 8..32 * 12];
        let f_cmp = &io[32 * 12..32 * 24];

        let s0_cast = AffinePoint::new(
            Fp::from_le_bytes_unchecked(&s0[..32]),
            Fp::from_le_bytes_unchecked(&s0[32..64]),
        );
        let s1_cast = AffinePoint::new(
            Fp::from_le_bytes_unchecked(&s1[..32]),
            Fp::from_le_bytes_unchecked(&s1[32..64]),
        );
        let q0_cast = AffinePoint::new(Fp2::from_bytes(&q0[..64]), Fp2::from_bytes(&q0[64..128]));
        let q1_cast = AffinePoint::new(Fp2::from_bytes(&q1[..64]), Fp2::from_bytes(&q1[64..128]));

        let f = Bn254::multi_miller_loop(&[s0_cast, s1_cast], &[q0_cast, q1_cast]);
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
    use openvm_pairing::bls12_381::{Bls12_381, Fp, Fp2};

    use super::*;

    openvm::init!("openvm_init_pairing_miller_loop_bls12_381.rs");

    pub fn test_miller_loop(io: &[u8]) {
        let s0 = &io[0..48 * 2];
        let s1 = &io[48 * 2..48 * 4];
        let q0 = &io[48 * 4..48 * 8];
        let q1 = &io[48 * 8..48 * 12];
        let f_cmp = &io[48 * 12..48 * 24];

        let s0_cast = AffinePoint::new(
            Fp::from_le_bytes_unchecked(&s0[..48]),
            Fp::from_le_bytes_unchecked(&s0[48..96]),
        );
        let s1_cast = AffinePoint::new(
            Fp::from_le_bytes_unchecked(&s1[..48]),
            Fp::from_le_bytes_unchecked(&s1[48..96]),
        );
        let q0_cast = AffinePoint::new(Fp2::from_bytes(&q0[..96]), Fp2::from_bytes(&q0[96..192]));
        let q1_cast = AffinePoint::new(Fp2::from_bytes(&q1[..96]), Fp2::from_bytes(&q1[96..192]));

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

    #[cfg(feature = "bn254")]
    {
        bn254::test_miller_loop(&io);
    }
    #[cfg(feature = "bls12_381")]
    {
        bls12_381::test_miller_loop(&io);
    }
}
