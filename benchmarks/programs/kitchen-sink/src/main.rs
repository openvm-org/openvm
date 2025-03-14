use std::{hint::black_box, mem::transmute};

use openvm_bigint_guest::I256;
use openvm_keccak256_guest::keccak256;
use openvm_sha256_guest::sha256;
#[allow(unused_imports)]
use {
    openvm_ecc_guest::{
        k256::Secp256k1Point, p256::P256Point, weierstrass::WeierstrassPoint, AffinePoint,
    },
    openvm_pairing_guest::{
        bls12_381::{Bls12_381G1Affine, G2Affine as Bls12_381G2Affine},
        bn254::{Bn254, Bn254G1Affine, Fp, Fp2, G2Affine as Bn254G2Affine},
        pairing::PairingCheck,
    },
};

openvm_algebra_guest::moduli_macros::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F", // secp256k1 Coordinate field
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141", // secp256k1 Scalar field
    "0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff", // secp256r1=p256 Coordinate field
    "0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551", // secp256r1=p256 Scalar field
    "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47", // Bn254Fp Coordinate field
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001", // Bn254 Scalar
    "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", // BLS12-381 Coordinate field
    "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001" // BLS12-381 Scalar field
}

openvm_ecc_guest::sw_macros::sw_init! {
    Secp256k1Point, P256Point,
    Bn254G1Affine, Bls12_381G1Affine
}

openvm_algebra_guest::complex_macros::complex_init! {
    Bn254Fp2 { mod_idx = 4 },
    Bls12_381Fp2 { mod_idx = 6 },
}

pub fn main() {
    // Setup will materialize every chip
    setup_all_moduli();
    setup_all_complex_extensions();
    setup_all_curves();

    let mut hash = vec![];
    for _ in 0..200 {
        let digest1 = keccak256(&hash);
        hash.extend_from_slice(&digest1);
        let digest2 = sha256(&hash);
        hash.extend_from_slice(&digest2);

        // SAFETY: internally I256 is represented as [u8; 32]
        let i1 = unsafe { transmute::<[u8; 32], I256>(digest1) };
        let i2 = unsafe { transmute::<[u8; 32], I256>(digest2) };

        black_box(&i1 + &i2);
        black_box(&i1 - &i2);
        black_box(&i1 * &i2);
        black_box(i1 == i2);
        black_box(i1 < i2);
        black_box(i1 <= i2);
        black_box(&i1 & &i2);
        black_box(&i1 ^ &i2);
        black_box(&i1 << &i2);
        black_box(&i1 >> &i2);
    }
}
