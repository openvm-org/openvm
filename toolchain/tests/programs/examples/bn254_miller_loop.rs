#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::io::read_vec;
use axvm_algebra::{field::FieldExtension, IntMod};
use axvm_ecc::{
    bn254::{Bn254, Fp, Fp2},
    pairing::MultiMillerLoop,
    AffinePoint,
};

axvm::entry!(main);

fn test_miller_loop(io: &[u8]) {
    let s = &io[0..32 * 2];
    let q = &io[32 * 2..32 * 6];
    let f_cmp = &io[32 * 6..32 * 18];

    let s_cast = unsafe { &*(s.as_ptr() as *const AffinePoint<Fp>) };
    let q_cast = unsafe { &*(q.as_ptr() as *const AffinePoint<Fp2>) };

    let f = Bn254::multi_miller_loop(&[s_cast.clone()], &[q_cast.clone()]);
    let mut f_bytes = [0u8; 32 * 12];
    f.to_coeffs()
        .iter()
        .flat_map(|fp2| fp2.clone().to_coeffs())
        .enumerate()
        .for_each(|(i, fp)| f_bytes[i * 32..(i + 1) * 32].copy_from_slice(fp.as_le_bytes()));

    assert_eq!(f_bytes, f_cmp);
}

pub fn main() {
    let io = read_vec();
    assert_eq!(io.len(), 32 * 18);
    test_miller_loop(&io);
}
