#![feature(cfg_match)]
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unused_imports)]

extern crate alloc;

use openvm::io::read_vec;
use openvm_ecc_guest::AffinePoint;
use openvm_pairing_guest::pairing::PairingCheck;

openvm::entry!(main);

#[cfg(feature = "bn254")]
mod bn254 {
    use alloc::format;

    use openvm_algebra_guest::{field::FieldExtension, IntMod};
    use openvm_pairing_guest::bn254::{Bn254, Fp, Fp2};

    use super::*;

    openvm_algebra_moduli_macros::moduli_init! {
        "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
        "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
    }

    openvm_algebra_complex_macros::complex_init! {
        Bn254Fp2 { mod_idx = 0 },
    }

    pub fn test_pairing_check(io: &[u8]) {
        setup_0();
        setup_all_complex_extensions();
        let s0 = &io[0..32 * 2];
        let s1 = &io[32 * 2..32 * 4];
        let q0 = &io[32 * 4..32 * 8];
        let q1 = &io[32 * 8..32 * 12];

        let s0_cast =
            AffinePoint::new(Fp::from_le_bytes(&s0[..32]), Fp::from_le_bytes(&s0[32..64]));
        let s1_cast =
            AffinePoint::new(Fp::from_le_bytes(&s1[..32]), Fp::from_le_bytes(&s1[32..64]));
        let q0_cast = AffinePoint::new(Fp2::from_bytes(&q0[..64]), Fp2::from_bytes(&q0[64..128]));
        let q1_cast = AffinePoint::new(Fp2::from_bytes(&q1[..64]), Fp2::from_bytes(&q1[64..128]));

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

    use openvm_algebra_guest::{field::FieldExtension, IntMod};
    use openvm_pairing_guest::bls12_381::{Bls12_381, Fp, Fp2};

    use super::*;

    openvm_algebra_moduli_macros::moduli_init! {
        "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
        "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
    }

    openvm_algebra_complex_macros::complex_init! {
        Bls12_381Fp2 { mod_idx = 0 },
    }

    pub fn test_pairing_check(io: &[u8]) {
        setup_0();
        setup_all_complex_extensions();
        let s0 = &io[0..48 * 2];
        let s1 = &io[48 * 2..48 * 4];
        let q0 = &io[48 * 4..48 * 8];
        let q1 = &io[48 * 8..48 * 12];

        let s0_cast =
            AffinePoint::new(Fp::from_le_bytes(&s0[..48]), Fp::from_le_bytes(&s0[48..96]));
        let s1_cast =
            AffinePoint::new(Fp::from_le_bytes(&s1[..48]), Fp::from_le_bytes(&s1[48..96]));
        let q0_cast = AffinePoint::new(Fp2::from_bytes(&q0[..96]), Fp2::from_bytes(&q0[96..192]));
        let q1_cast = AffinePoint::new(Fp2::from_bytes(&q1[..96]), Fp2::from_bytes(&q1[96..192]));

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
