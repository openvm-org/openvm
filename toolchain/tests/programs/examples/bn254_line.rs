#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::io::read_vec;
use axvm_algebra::{field::FieldExtension, IntMod};
use axvm_ecc::{
    bn254::{Bn254, Fp12, Fp2},
    pairing::{EvaluatedLine, LineMulDType},
};

axvm::entry!(main);

fn test_mul_013_by_013(io: &[u8]) {
    assert_eq!(io.len(), 32 * 18);
    let l0 = &io[..32 * 4];
    let l1 = &io[32 * 4..32 * 8];
    let expected = &io[32 * 8..32 * 18];

    let l0_cast = unsafe { &*(l0.as_ptr() as *const EvaluatedLine<Fp2>) };
    let l1_cast = unsafe { &*(l1.as_ptr() as *const EvaluatedLine<Fp2>) };

    let r = Bn254::mul_013_by_013(&l0_cast, &l1_cast);
    let mut r_bytes = [0u8; 32 * 10];
    let mut i = 0;
    for x in r {
        r_bytes[i..i + 32].copy_from_slice(&x.c0.as_le_bytes());
        r_bytes[i + 32..i + 64].copy_from_slice(&x.c1.as_le_bytes());
        i += 64;
    }
    assert_eq!(r_bytes, expected);
}

fn test_mul_by_01234(io: &[u8]) {
    assert_eq!(io.len(), 32 * 34);
    let f = &io[..32 * 12];
    let x = &io[32 * 12..32 * 22];
    let expected = &io[32 * 22..32 * 34];

    let f_cast = unsafe { &*(f.as_ptr() as *const Fp12) };
    let x_cast = unsafe { &*(x.as_ptr() as *const [Fp2; 5]) };

    let r = Bn254::mul_by_01234(&f_cast, &x_cast);
    let mut r_bytes = [0u8; 32 * 12];
    let mut i = 0;
    for x in r.to_coeffs() {
        r_bytes[i..i + 32].copy_from_slice(&x.c0.as_le_bytes());
        r_bytes[i + 32..i + 64].copy_from_slice(&x.c1.as_le_bytes());
        i += 64;
    }
    assert_eq!(r_bytes, expected);
}

pub fn main() {
    let io = read_vec();
    assert_eq!(io.len(), 32 * 52);

    test_mul_013_by_013(&io[..32 * 18]);
    test_mul_by_01234(&io[32 * 18..32 * 52]);
}
