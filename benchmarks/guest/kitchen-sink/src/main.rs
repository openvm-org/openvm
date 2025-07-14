extern crate alloc;

use std::hint::black_box;

use openvm_algebra_guest::IntMod;
use openvm_keccak256::keccak256;
use openvm_ruint::aliases::U256;
use openvm_sha2::sha256;
#[allow(unused_imports)]
use {
    openvm_ecc_guest::{weierstrass::WeierstrassPoint, CyclicGroup},
    openvm_k256::{Secp256k1Coord, Secp256k1Point, Secp256k1Scalar},
    openvm_p256::{P256Coord, P256Point, P256Scalar},
    openvm_pairing::{
        bls12_381::{
            Bls12_381Fp, Bls12_381Fp2, Bls12_381G1Affine, Bls12_381Scalar,
            G2Affine as Bls12_381G2Affine,
        },
        bn254::{Bn254, Bn254Fp, Bn254Fp2, Bn254G1Affine, Bn254Scalar, G2Affine as Bn254G2Affine},
        PairingCheck,
    },
};

// Note: these will all currently be represented as bytes32 even though they could be smaller
openvm_algebra_guest::moduli_macros::moduli_declare! {
    Seven { modulus = "7" },
    Mod1e18 { modulus = "1000000000000000003" },
    Mersenne61 { modulus = "0x1fffffffffffffff" },
}

openvm::init!();

fn materialize_modular_chip<T: IntMod>() {
    // ensure the compiler doesn't optimize out the operations
    // add/sub chip
    black_box(T::ZERO + T::ZERO);
    // mul/div chip
    black_box(T::ZERO * T::ZERO);
    // is_equal chip
    black_box(T::ZERO.assert_reduced());
}

// making this a macro since there's no complex extension trait
macro_rules! materialize_complex_chip {
    ($complex_type:ty, $modular_type:ty) => {
        // ensure the compiler doesn't optimize out the operations
        let zero = <$complex_type>::new(
            <$modular_type as IntMod>::ZERO,
            <$modular_type as IntMod>::ZERO,
        );
        // add/sub chip
        black_box(&zero + &zero);
        // mul/div chip
        black_box(&zero * &zero);
    };
}

fn materialize_ecc_chip<T: WeierstrassPoint + CyclicGroup>() {
    // add chip
    // it is important that neither operand is identity, otherwise the chip will not be materialized
    black_box(T::GENERATOR + T::GENERATOR);
    // double chip
    // it is important that the operand is not identity, otherwise the chip will not be materialized
    black_box(T::GENERATOR.double());
}

pub fn main() {
    // Since we don't explicitly call setup functions anymore, we must ensure every declared modulus
    // and curve chip is materialized.
    materialize_modular_chip::<Mod1e18>();
    materialize_modular_chip::<Secp256k1Coord>();
    materialize_modular_chip::<Secp256k1Scalar>();
    materialize_modular_chip::<P256Coord>();
    materialize_modular_chip::<P256Scalar>();
    materialize_modular_chip::<Bn254Fp>();
    materialize_modular_chip::<Bn254Scalar>();
    materialize_modular_chip::<Bls12_381Fp>();
    materialize_modular_chip::<Bls12_381Scalar>();
    materialize_modular_chip::<Mersenne61>();
    materialize_modular_chip::<Seven>();

    materialize_complex_chip!(Bn254Fp2, Bn254Fp);
    materialize_complex_chip!(Bls12_381Fp2, Bls12_381Fp);

    materialize_ecc_chip::<Secp256k1Point>();
    materialize_ecc_chip::<P256Point>();
    materialize_ecc_chip::<Bn254G1Affine>();
    materialize_ecc_chip::<Bls12_381G1Affine>();

    let [one, six] = [1, 6].map(Seven::from_u32);
    assert_eq!(one + six, Seven::ZERO);

    let y = Mod1e18::from_u32(100);
    let y = (&y * &y) * &y;
    let y = y.clone() * y.clone() * y.clone();
    assert_eq!(y + Mod1e18::from_u32(3), Mod1e18::ZERO);

    let mut bytes = [0u8; 32];
    bytes[7] = 1 << 5; // 2^61 = modulus + 1
    let mut res = Mersenne61::from_le_bytes_unchecked(&bytes); // No need to start from reduced representation
    for _ in 0..61 {
        res += res.clone();
    }
    assert_eq!(res, Mersenne61::from_u32(1));
    let two = Mersenne61::from_u32(2);
    for _ in 0..61 {
        res *= &two;
    }
    assert_eq!(res, Mersenne61::from_u32(1));

    let mut hash = vec![];
    for _ in 0..200 {
        let digest1 = keccak256(&hash);
        hash.extend_from_slice(&digest1);
        let digest2 = sha256(&hash);
        hash.extend_from_slice(&digest2);

        // SAFETY: internally U256 is represented as [u8; 32]
        let i1 = U256::from_le_bytes(digest1);
        let i2 = U256::from_le_bytes(digest2);

        black_box(&i1 + &i2);
        black_box(&i1 - &i2);
        black_box(&i1 * &i2);
        black_box(i1 == i2);
        black_box(i1 < i2);
        black_box(i1 <= i2);
        black_box(&i1 & &i2);
        black_box(&i1 ^ &i2);
        black_box(i1 << &i2);
        black_box(i1 >> &i2);
    }
}
