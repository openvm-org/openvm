#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use axvm::io::read_vec;
use axvm_algebra::{field::FieldExtension, IntMod};
use axvm_ecc::{
    bn254::{Bn254, Fp, Fp2},
    pairing::MultiMillerLoop,
    AffinePoint,
};

axvm::entry!(main);
axvm::moduli_setup! {
    Bn254Modulus { modulus = "21888242871839275222246405745257275088696311157297823662689037894645226208583" },
}

fn test_pairing_check(io: &[u8]) {
    let s0 = &io[0..32 * 2];
    let s1 = &io[32 * 2..32 * 4];
    let q0 = &io[32 * 4..32 * 8];
    let q1 = &io[32 * 8..32 * 12];
    let f_cmp = &io[32 * 12..32 * 24];

    let s0_cast = unsafe { &*(s0.as_ptr() as *const AffinePoint<Fp>) };
    let s1_cast = unsafe { &*(s1.as_ptr() as *const AffinePoint<Fp>) };
    let q0_cast = unsafe { &*(q0.as_ptr() as *const AffinePoint<Fp2>) };
    let q1_cast = unsafe { &*(q1.as_ptr() as *const AffinePoint<Fp2>) };

    let f = Bn254::pairing_check(
        &[s0_cast.clone(), s1_cast.clone()],
        &[q0_cast.clone(), q1_cast.clone()],
    );


pub fn main() {
    setup_Bn254Modulus();
    let io = read_vec();
    assert_eq!(io.len(), 32 * 12);
    test_pairing_check(&io);
}
