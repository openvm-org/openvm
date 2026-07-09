#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use openvm::io::read_vec;
use openvm_algebra_guest::{field::FieldExtension, IntMod};
use openvm_ecc_guest::AffinePoint;
use openvm_pairing_guest::pairing::MultiMillerLoop;

openvm::entry!(main);

#[cfg(feature = "bls12_381")]
mod bls12_381 {
    use openvm_pairing::bls12_381::{Bls12_381, Fp, Fp2};

    use super::*;

    openvm::init!("openvm_init_rvr_pairing_miller_loop_bls12_381.rs");

    pub fn test_miller_loop(io: &[u8]) {
        let p = AffinePoint::new(
            Fp::from_le_bytes_unchecked(&io[..48]),
            Fp::from_le_bytes_unchecked(&io[48..96]),
        );
        let q = AffinePoint::new(
            Fp2::from_bytes(&io[96..192]),
            Fp2::from_bytes(&io[192..288]),
        );
        let expected = &io[288..864];
        let expected_base_sum = Fp::from_le_bytes_unchecked(&io[864..912]);

        // Close the 48-byte VecHeap coverage gap in the same BLS12-381
        // fixture that exercises the 96-byte Fp2 path below.
        assert_eq!(p.x.clone() + p.y.clone(), expected_base_sum);

        let f = Bls12_381::multi_miller_loop(&[p], &[q]);
        let mut actual = [0u8; 48 * 12];
        f.to_coeffs()
            .iter()
            .flat_map(|fp2| fp2.clone().to_coeffs())
            .enumerate()
            .for_each(|(i, fp)| actual[i * 48..(i + 1) * 48].copy_from_slice(fp.as_le_bytes()));
        assert_eq!(actual, expected);
    }
}

pub fn main() {
    let io = read_vec();

    #[cfg(feature = "bls12_381")]
    bls12_381::test_miller_loop(&io);
}
