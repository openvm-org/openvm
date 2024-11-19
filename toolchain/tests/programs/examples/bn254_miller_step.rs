#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::io::read_vec;
use axvm_algebra::IntMod;
use axvm_ecc::{
    bn254::{Bn254, Fp2},
    pairing::{MillerStep, PairingIntrinsics},
    AffinePoint,
};

axvm::entry!(main);

fn test_miller_step(io: &[u8]) {
    assert_eq!(io.len(), 32 * 12);
    let s = &io[..32 * 4];
    let pt = &io[32 * 4..32 * 8];
    let l = &io[32 * 8..32 * 12];

    let s_cast = unsafe { &*(s.as_ptr() as *const AffinePoint<Fp2>) };

    let (pt_cmp, l_cmp) = Bn254::miller_double_step(&s_cast);
    let mut pt_bytes = [0u8; 32 * 4];
    let mut l_bytes = [0u8; 32 * 4];

    pt_bytes[0 * 32..1 * 32].copy_from_slice(&pt_cmp.x.c0.as_le_bytes());
    pt_bytes[1 * 32..2 * 32].copy_from_slice(&pt_cmp.x.c1.as_le_bytes());
    pt_bytes[2 * 32..3 * 32].copy_from_slice(&pt_cmp.y.c0.as_le_bytes());
    pt_bytes[3 * 32..4 * 32].copy_from_slice(&pt_cmp.y.c1.as_le_bytes());
    l_bytes[0 * 32..1 * 32].copy_from_slice(&l_cmp.b.c0.as_le_bytes());
    l_bytes[1 * 32..2 * 32].copy_from_slice(&l_cmp.b.c1.as_le_bytes());
    l_bytes[2 * 32..3 * 32].copy_from_slice(&l_cmp.c.c0.as_le_bytes());
    l_bytes[3 * 32..4 * 32].copy_from_slice(&l_cmp.c.c1.as_le_bytes());

    assert_eq!(pt_bytes, pt);
    assert_eq!(l_bytes, l);
}

fn test_miller_double_and_add_step(io: &[u8]) {
    let s = &io[32 * 12..32 * 16];
    let q = &io[32 * 16..32 * 20];
    let pt = &io[32 * 20..32 * 24];
    let l0 = &io[32 * 24..32 * 28];
    let l1 = &io[32 * 28..32 * 32];

    let s_cast = unsafe { &*(s.as_ptr() as *const AffinePoint<Fp2>) };
    let q_cast = unsafe { &*(q.as_ptr() as *const AffinePoint<Fp2>) };
    let (pt_cmp, l0_cmp, l1_cmp) = Bn254::miller_double_and_add_step(&s_cast, &q_cast);
    let mut pt_bytes = [0u8; 32 * 4];
    let mut l0_bytes = [0u8; 32 * 4];
    let mut l1_bytes = [0u8; 32 * 4];

    pt_bytes[0 * 32..1 * 32].copy_from_slice(&pt_cmp.x.c0.as_le_bytes());
    pt_bytes[1 * 32..2 * 32].copy_from_slice(&pt_cmp.x.c1.as_le_bytes());
    pt_bytes[2 * 32..3 * 32].copy_from_slice(&pt_cmp.y.c0.as_le_bytes());
    pt_bytes[3 * 32..4 * 32].copy_from_slice(&pt_cmp.y.c1.as_le_bytes());
    l0_bytes[0 * 32..1 * 32].copy_from_slice(&l0_cmp.b.c0.as_le_bytes());
    l0_bytes[1 * 32..2 * 32].copy_from_slice(&l0_cmp.b.c1.as_le_bytes());
    l0_bytes[2 * 32..3 * 32].copy_from_slice(&l0_cmp.c.c0.as_le_bytes());
    l0_bytes[3 * 32..4 * 32].copy_from_slice(&l0_cmp.c.c1.as_le_bytes());
    l1_bytes[0 * 32..1 * 32].copy_from_slice(&l1_cmp.b.c0.as_le_bytes());
    l1_bytes[1 * 32..2 * 32].copy_from_slice(&l1_cmp.b.c1.as_le_bytes());
    l1_bytes[2 * 32..3 * 32].copy_from_slice(&l1_cmp.c.c0.as_le_bytes());
    l1_bytes[3 * 32..4 * 32].copy_from_slice(&l1_cmp.c.c1.as_le_bytes());

    assert_eq!(pt_bytes, pt);
    assert_eq!(l0_bytes, l0);
    assert_eq!(l1_bytes, l1);
}

pub fn main() {
    let io = read_vec();
    assert_eq!(io.len(), 32 * 32);

    test_miller_step(&io);
    test_miller_double_and_add_step(&io);
}
