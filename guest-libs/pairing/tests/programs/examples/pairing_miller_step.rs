#![allow(unused_imports)]
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::read_vec;
use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::AffinePoint;
use openvm_pairing_guest::pairing::MillerStep;

openvm::entry!(main);

#[cfg(feature = "bn254")]
mod bn254 {
    use openvm_algebra_guest::field::FieldExtension;
    use openvm_pairing::bn254::{Bn254, Fp, Fp2};

    use super::*;

    openvm::init!("openvm_init_pairing_miller_step_bn254.rs");

    pub fn test_miller_step(io: &[u8]) {
        assert_eq!(io.len(), 32 * 12);
        let s = &io[..32 * 4];
        let pt = &io[32 * 4..32 * 8];
        let l = &io[32 * 8..32 * 12];

        let s_cast = AffinePoint::new(Fp2::from_bytes(&s[..64]), Fp2::from_bytes(&s[64..128]));

        let (pt_cmp, l_cmp) = Bn254::miller_double_step(&s_cast);
        let mut pt_bytes = [0u8; 32 * 4];
        let mut l_bytes = [0u8; 32 * 4];

        // TODO: if we ever need to change this, we should switch to using `StdIn::write` to
        // serialize       for us and use `read()` instead of `read_vec()`
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

        let s_cast = AffinePoint::new(Fp2::from_bytes(&s[..64]), Fp2::from_bytes(&s[64..128]));
        let q_cast = AffinePoint::new(Fp2::from_bytes(&q[..64]), Fp2::from_bytes(&q[64..128]));
        let (pt_cmp, l0_cmp, l1_cmp) = Bn254::miller_double_and_add_step(&s_cast, &q_cast);
        let mut pt_bytes = [0u8; 32 * 4];
        let mut l0_bytes = [0u8; 32 * 4];
        let mut l1_bytes = [0u8; 32 * 4];

        // TODO: if we ever need to change this, we should switch to using `StdIn::write` to
        // serialize       for us and use `read()` instead of `read_vec()`
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
    use openvm_algebra_guest::field::FieldExtension;
    use openvm_pairing::bls12_381::{Bls12_381, Fp, Fp2};

    use super::*;

    openvm::init!("openvm_init_pairing_miller_step_bls12_381.rs");

    pub fn test_miller_step(io: &[u8]) {
        assert_eq!(io.len(), 48 * 12);
        let s = &io[..48 * 4];
        let pt = &io[48 * 4..48 * 8];
        let l = &io[48 * 8..48 * 12];

        let s_cast = AffinePoint::new(Fp2::from_bytes(&s[..96]), Fp2::from_bytes(&s[96..192]));

        let (pt_cmp, l_cmp) = Bls12_381::miller_double_step(&s_cast);
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

        let s_cast = AffinePoint::new(Fp2::from_bytes(&s[..96]), Fp2::from_bytes(&s[96..192]));
        let q_cast = AffinePoint::new(Fp2::from_bytes(&q[..96]), Fp2::from_bytes(&q[96..192]));
        let (pt_cmp, l0_cmp, l1_cmp) = Bls12_381::miller_double_and_add_step(&s_cast, &q_cast);
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

    #[cfg(feature = "bn254")]
    {
        bn254::test_miller_step(&io[..32 * 12]);
        bn254::test_miller_double_and_add_step(&io[32 * 12..]);
    }
    #[cfg(feature = "bls12_381")]
    {
        bls12_381::test_miller_step(&io[..48 * 12]);
        bls12_381::test_miller_double_and_add_step(&io[48 * 12..]);
    }
}
