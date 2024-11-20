#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::io::read_vec;
use axvm_algebra::IntMod;
use axvm_ecc::{
    bn254::{Bn254, Fp, Fp2},
    pairing::MultiMillerLoop,
    AffinePoint,
};

axvm::entry!(main);

fn test_miller_loop(io: &[u8]) {
    let s = &io[0..32 * 2];
    let q = &io[32 * 2..32 * 6];
    let f = &io[32 * 6..32 * 18];

    let s_cast = unsafe { &*(s.as_ptr() as *const AffinePoint<Fp>) };
    let q_cast = unsafe { &*(q.as_ptr() as *const AffinePoint<Fp2>) };

    let f_cmp = Bn254::multi_miller_loop(&[s_cast.clone()], &[q_cast.clone()]);

    let mut f_bytes = [0u8; 32 * 12];
    f_bytes[0..32].copy_from_slice(f_cmp.c[0].c0.as_le_bytes());
    f_bytes[32..2 * 32].copy_from_slice(f_cmp.c[0].c1.as_le_bytes());
    f_bytes[2 * 32..3 * 32].copy_from_slice(f_cmp.c[1].c0.as_le_bytes());
    f_bytes[3 * 32..4 * 32].copy_from_slice(f_cmp.c[1].c1.as_le_bytes());
    f_bytes[4 * 32..5 * 32].copy_from_slice(f_cmp.c[2].c0.as_le_bytes());
    f_bytes[5 * 32..6 * 32].copy_from_slice(f_cmp.c[2].c1.as_le_bytes());
    f_bytes[6 * 32..7 * 32].copy_from_slice(f_cmp.c[3].c0.as_le_bytes());
    f_bytes[7 * 32..8 * 32].copy_from_slice(f_cmp.c[3].c1.as_le_bytes());
    f_bytes[8 * 32..9 * 32].copy_from_slice(f_cmp.c[4].c0.as_le_bytes());
    f_bytes[9 * 32..10 * 32].copy_from_slice(f_cmp.c[4].c1.as_le_bytes());
    f_bytes[10 * 32..11 * 32].copy_from_slice(f_cmp.c[5].c0.as_le_bytes());
    f_bytes[11 * 32..12 * 32].copy_from_slice(f_cmp.c[5].c1.as_le_bytes());

    assert_eq!(f, f_bytes);
}

pub fn main() {
    let io = read_vec();
    assert_eq!(io.len(), 32 * 18);
    test_miller_loop(&io);
}
