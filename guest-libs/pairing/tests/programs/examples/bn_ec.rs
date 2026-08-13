#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "bn254")]
use {
    openvm_algebra_guest::IntMod,
    openvm_ecc_guest::{weierstrass::CachedMulTable, CyclicGroup, Group},
    openvm_pairing::bn254::{Bn254, Bn254G1Affine, Bn254Scalar},
};

openvm::init!("openvm_init_bn_ec_bn254.rs");

openvm::entry!(main);

#[cfg(feature = "bn254")]
fn windowed_reference(base: Bn254G1Affine, scalar: Bn254Scalar) -> Bn254G1Affine {
    let bases = [base];
    CachedMulTable::<Bn254>::new_with_prime_order(&bases, 4).windowed_mul(&[scalar])
}

pub fn main() {
    #[cfg(feature = "bn254")]
    {
        let generator = Bn254G1Affine::GENERATOR;
        let generic = generator.double();

        for k in [0u64, 1, 2, 3, 255, 0x1234_5678, u32::MAX as u64] {
            let scalar = Bn254Scalar::from_u64(k);
            for base in [generator.clone(), generic.clone()] {
                assert_eq!(
                    base.mul_scalar(&scalar),
                    windowed_reference(base, scalar.clone())
                );
            }
        }

        let identity = <Bn254G1Affine as Group>::IDENTITY;
        assert!(identity.mul_scalar(&Bn254Scalar::from_u64(7)).is_identity());
    }
}
