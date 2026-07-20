#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use openvm::io::reveal_bytes32;
use openvm_algebra_guest::{DivUnsafe, IntMod};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    Bls12_381Fp { modulus = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab" },
    Bls12_381Scalar { modulus = "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001" },
}

openvm_algebra_complex_macros::complex_declare! {
    Bls12_381Fp2 { mod_type = Bls12_381Fp },
}

openvm::init!("openvm_init_bls12_381_rvr.rs");

const EXPECTED_DIGEST: [u8; 32] = [
    0x84, 0xf2, 0xb5, 0x9b, 0xb5, 0xe0, 0x6a, 0xc9, 0x0a, 0x96, 0xe4, 0x6d, 0xe0, 0xd8, 0x97, 0x71,
    0xf1, 0x59, 0x23, 0x46, 0x31, 0x39, 0x39, 0x40, 0xa0, 0x9d, 0x57, 0x79, 0x61, 0xbf, 0xf4, 0x5a,
];

fn increment<const N: usize>(mut bytes: [u8; N]) -> [u8; N] {
    for byte in &mut bytes {
        let (next, carry) = byte.overflowing_add(1);
        *byte = next;
        if !carry {
            break;
        }
    }
    bytes
}

fn absorb(digest: &mut [u8; 32], position: &mut usize, bytes: &[u8]) {
    for &byte in bytes {
        let slot = *position % digest.len();
        digest[slot] = digest[slot].rotate_left(1) ^ byte;
        *position += 1;
    }
}

pub fn main() {
    let fp_modulus = Bls12_381Fp::from_le_bytes_unchecked(&Bls12_381Fp::MODULUS);
    let fp_modulus_plus_one =
        Bls12_381Fp::from_le_bytes_unchecked(&increment(Bls12_381Fp::MODULUS));
    let fp_two = Bls12_381Fp::from_u8(2);
    let fp_three = Bls12_381Fp::from_u8(3);
    let fp_sum = fp_modulus_plus_one.clone() + &fp_two;
    let fp_equal = fp_sum == fp_three;
    let fp_results = [
        fp_sum,
        fp_modulus_plus_one.clone() - &fp_two,
        fp_modulus_plus_one.clone() * &fp_three,
        fp_modulus_plus_one.clone().div_unsafe(&fp_three),
        Bls12_381Fp::from_le_bytes_unchecked(&[u8::MAX; 48]) + &Bls12_381Fp::ONE,
    ];

    let scalar_modulus_plus_one =
        Bls12_381Scalar::from_le_bytes_unchecked(&increment(Bls12_381Scalar::MODULUS));
    let scalar_two = Bls12_381Scalar::from_u8(2);
    let scalar_three = Bls12_381Scalar::from_u8(3);
    let scalar_sum = scalar_modulus_plus_one.clone() + &scalar_two;
    let scalar_equal = scalar_sum == scalar_three;
    let scalar_results = [
        scalar_sum,
        scalar_modulus_plus_one.clone() - &scalar_two,
        scalar_modulus_plus_one.clone() * &scalar_three,
        scalar_modulus_plus_one.clone().div_unsafe(&scalar_three),
        Bls12_381Scalar::from_le_bytes_unchecked(&[u8::MAX; 32]) + &Bls12_381Scalar::ONE,
    ];

    let fp2_left = Bls12_381Fp2::new(fp_modulus_plus_one, fp_modulus.clone());
    let fp2_right = Bls12_381Fp2::new(fp_two, fp_three);
    let fp2_results = [
        fp2_left.clone() + &fp2_right,
        fp2_left.clone() - &fp2_right,
        fp2_left.clone() * &fp2_right,
        fp2_left.div_unsafe(&fp2_right),
    ];

    let mut digest = [0u8; 32];
    let mut position = 0;
    for value in fp_results {
        absorb(&mut digest, &mut position, value.as_le_bytes());
    }
    absorb(&mut digest, &mut position, &[fp_equal as u8]);
    for value in scalar_results {
        absorb(&mut digest, &mut position, value.as_le_bytes());
    }
    absorb(&mut digest, &mut position, &[scalar_equal as u8]);
    for value in fp2_results {
        absorb(&mut digest, &mut position, value.c0.as_le_bytes());
        absorb(&mut digest, &mut position, value.c1.as_le_bytes());
    }
    assert_eq!(digest, EXPECTED_DIGEST);
    reveal_bytes32(digest);
}
