extern crate alloc;

use alloc::vec::Vec;
use core::ops::{MulAssign, Neg};

use openvm_algebra_guest::IntMod;
use openvm_algebra_moduli_macros::moduli_declare;
use openvm_ecc_guest::{
    weierstrass::{IntrinsicCurve, ScalarMul, WeierstrassPoint},
    CyclicGroup, Group,
};

mod fp12;
mod fp2;
mod pairing;
#[cfg(all(
    feature = "halo2curves",
    not(any(openvm_intrinsics, target_os = "openvm"))
))]
pub(crate) mod utils;

pub use fp12::*;
pub use fp2::*;
use hex_literal::hex;
use openvm_ecc_sw_macros::sw_declare;
use openvm_pairing_guest::pairing::PairingIntrinsics;

#[cfg(all(
    test,
    feature = "halo2curves",
    not(any(openvm_intrinsics, target_os = "openvm"))
))]
mod tests;

moduli_declare! {
    Bls12_381Fp { modulus = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab" },
    Bls12_381Scalar { modulus = "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001" },
}

const CURVE_B: Bls12_381Fp = Bls12_381Fp::from_const_u8(4);

sw_declare! {
    Bls12_381G1Affine { mod_type = Bls12_381Fp, b = CURVE_B },
}

pub type Fp = Bls12_381Fp;
pub type Scalar = Bls12_381Scalar;
/// Affine point representation of `Fp` points of BLS12-381.
/// **Note**: an instance of this type may be constructed that lies
/// on the curve but not necessarily in the prime order subgroup
/// because the group has cofactors.
pub type G1Affine = Bls12_381G1Affine;
pub use g2::G2Affine;

const G1_X: [u64; 1] = [0xd201000000010000];
const G1_BETA: Fp = Fp::from_const_bytes(hex!(
    "fefffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
));
const G1_H_EFF: [u64; 1] = [G1_X[0] + 1];
const G1_H_EFF_INV: Scalar = Scalar::from_const_bytes(hex!(
    "010000000000ffff0000fd890148001435d39e93390be8a5477d9d2953a7ed73"
));

fn g1_small_scalar_mul<const CHECK_SETUP: bool>(base: &G1Affine, scalar: &[u64]) -> G1Affine {
    let mut result = <G1Affine as Group>::IDENTITY;
    let mut temp = base.clone();
    for limb in scalar {
        for bit_idx in 0..64 {
            if (limb >> bit_idx) & 1 == 1 {
                result.add_assign_impl::<CHECK_SETUP>(&temp);
            }
            temp.double_assign_impl::<CHECK_SETUP>();
        }
    }
    result
}

// https://hackmd.io/@benjaminion/bls12-381#Cofactor
// BLS12-381: The from_xy function will allow constructing elements that lie on the curve
// but aren't actually in the cyclic subgroup of prime order that is usually called G1.
impl CyclicGroup for G1Affine {
    // https://github.com/zcash/librustzcash/blob/6e0364cd42a2b3d2b958a54771ef51a8db79dd29/pairing/src/bls12_381/README.md#generators
    const GENERATOR: Self = G1Affine {
        x: Bls12_381Fp::from_const_bytes(hex!(
            "BBC622DB0AF03AFBEF1A7AF93FE8556C58AC1B173F3A4EA105B974974F8C68C30FACA94F8C63952694D79731A7D3F117"
        )),
        y: Bls12_381Fp::from_const_bytes(hex!(
            "E1E7C5462923AA0CE48A88A244C73CD0EDB3042CCB18DB00F60AD0D595E0F5FCE48A1D74ED309EA0F1A0AAE381F4B308"
        )),
    };
    const NEG_GENERATOR: Self = G1Affine {
        x: Bls12_381Fp::from_const_bytes(hex!(
            "BBC622DB0AF03AFBEF1A7AF93FE8556C58AC1B173F3A4EA105B974974F8C68C30FACA94F8C63952694D79731A7D3F117"
        )),
        y: Bls12_381Fp::from_const_bytes(hex!(
            "CAC239B9D6DC54AD1B75CB0EBA386F4E3642ACCAD5B95566C907B51DEF6A8167F2212ECFC8767DAAA845D555681D4D11"
        )),
    };
}

/// Reduces a 256-bit scalar without using the scalar arithmetic extension.
fn reduced_scalar_bytes(scalar: &Scalar) -> [u8; 32] {
    let mut bytes: [u8; 32] = scalar.as_le_bytes().try_into().unwrap();
    let modulus = Scalar::MODULUS;

    // Every 256-bit value is less than 3q. Two subtractions are enough.
    for _ in 0..2 {
        if bytes.iter().rev().cmp(modulus.iter().rev()).is_ge() {
            sub_assign_bytes(&mut bytes, &modulus);
        }
    }
    bytes
}

fn reduced_scalar(scalar: &Scalar) -> Scalar {
    Scalar::from_le_bytes_unchecked(&reduced_scalar_bytes(scalar))
}

/// Subtracts `rhs` from `lhs` as little-endian integers. Requires `lhs >= rhs`.
fn sub_assign_bytes(lhs: &mut [u8; 32], rhs: &[u8; 32]) {
    let mut borrow = 0u16;
    for (lhs_byte, &rhs_byte) in lhs.iter_mut().zip(rhs) {
        let lhs_word = *lhs_byte as u16;
        let rhs_word = rhs_byte as u16 + borrow;
        if lhs_word >= rhs_word {
            *lhs_byte = (lhs_word - rhs_word) as u8;
            borrow = 0;
        } else {
            *lhs_byte = (lhs_word + 256 - rhs_word) as u8;
            borrow = 1;
        }
    }
    debug_assert_eq!(borrow, 0);
}

impl G1Affine {
    /// Multiplies an on-curve BLS12-381 G1 point by a scalar.
    ///
    /// This function also supports the identity and points outside the prime-order subgroup.
    pub fn mul_scalar(&self, scalar: &Scalar) -> Self {
        let reduced = reduced_scalar(scalar);
        openvm_ecc_guest::msm(core::slice::from_ref(&reduced), core::slice::from_ref(self))
    }

    /// Checks prime-subgroup membership with the BLS12-381 G1 endomorphism.
    pub fn is_in_prime_subgroup_via_endomorphism(&self) -> bool {
        if self.is_identity() {
            return true;
        }

        let x_times_point = g1_small_scalar_mul::<true>(self, &G1_X);
        if self == &x_times_point {
            return false;
        }

        // The first multiplication performed curve setup.
        let minus_x_squared = -g1_small_scalar_mul::<false>(&x_times_point, &G1_X);
        let mut endomorphism = self.clone();
        endomorphism.x_mut().mul_assign(&G1_BETA);
        minus_x_squared == endomorphism
    }

    /// Checks prime-subgroup membership by projecting onto the prime-order subgroup.
    pub fn is_in_prime_subgroup_via_projection(&self) -> bool {
        if self.is_identity() {
            return true;
        }

        let cleared = g1_small_scalar_mul::<true>(self, &G1_H_EFF);
        // SAFETY: Multiplication by the effective cofactor puts `cleared` in the prime-order
        // subgroup.
        let projected = unsafe { cleared.mul_scalar_prime_subgroup_unchecked(&G1_H_EFF_INV) };
        projected == *self
    }

    /// Multiplies a prime-subgroup point using the `EC_MUL` intrinsic.
    ///
    /// This function reduces the scalar and handles zero, even scalars, and the identity point.
    ///
    /// # Safety
    ///
    /// `self` must be on the BLS12-381 G1 curve and in its prime-order subgroup. This function
    /// does not check these requirements.
    pub unsafe fn mul_scalar_prime_subgroup_unchecked(&self, scalar: &Scalar) -> Self {
        if self.is_identity() {
            return <Self as Group>::IDENTITY;
        }
        let mut reduced = reduced_scalar_bytes(scalar);
        if reduced.iter().all(|&byte| byte == 0) {
            return <Self as Group>::IDENTITY;
        }

        // EC_MUL needs an odd scalar below the subgroup order. For even k, q - k is odd and
        // (q - k) * P = -(k * P) because P is in the order-q subgroup.
        let odd = reduced[0] & 1 == 1;
        if !odd {
            let mut negated = Scalar::MODULUS;
            sub_assign_bytes(&mut negated, &reduced);
            reduced = negated;
        }
        // SAFETY: The caller guarantees subgroup membership. `reduced` is odd, nonzero, and less
        // than the subgroup order. The order is 1 modulo 4. EC_MUL setup runs on first use.
        let product = unsafe { self.mul_scalar_le_unchecked::<true>(&reduced) };
        if odd {
            product
        } else {
            -product
        }
    }
}

impl ScalarMul<Scalar> for G1Affine {
    fn mul_scalar(&self, scalar: &Scalar) -> Self {
        G1Affine::mul_scalar(self, scalar)
    }
}

pub struct Bls12_381;

impl Bls12_381 {
    /// Multiplies the standard G1 generator using the `EC_MUL` intrinsic.
    pub fn mul_generator(scalar: &Scalar) -> G1Affine {
        // SAFETY: `GENERATOR` is in the BLS12-381 G1 prime-order subgroup.
        unsafe { G1Affine::GENERATOR.mul_scalar_prime_subgroup_unchecked(scalar) }
    }

    /// Computes an MSM of points that have already passed a subgroup check using `EC_MUL`.
    ///
    /// # Safety
    ///
    /// Every base must be an on-curve point in the BLS12-381 G1 prime-order subgroup. This
    /// function does not check subgroup membership.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs` and `bases` have different lengths.
    pub unsafe fn msm_prime_subgroup_unchecked(coeffs: &[Scalar], bases: &[G1Affine]) -> G1Affine {
        assert_eq!(
            coeffs.len(),
            bases.len(),
            "msm requires matching scalar/base lengths"
        );

        let mut acc = <G1Affine as Group>::IDENTITY;
        for (coeff, base) in coeffs.iter().zip(bases) {
            // SAFETY: The caller guarantees subgroup membership for every base.
            acc += unsafe { base.mul_scalar_prime_subgroup_unchecked(coeff) };
        }
        acc
    }
}

impl IntrinsicCurve for Bls12_381 {
    type Scalar = Scalar;
    type Point = G1Affine;

    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point {
        let coeffs: Vec<_> = coeffs.iter().map(reduced_scalar).collect();
        openvm_ecc_guest::msm(&coeffs, bases)
    }
}

// Define a G2Affine struct that implements curve operations using `Fp2` intrinsics
// but not special E(Fp2) intrinsics.
mod g2 {
    use openvm_algebra_guest::Field;
    use openvm_ecc_guest::{
        impl_sw_affine, impl_sw_group_ops, weierstrass::WeierstrassPoint, AffinePoint, Group,
    };

    use super::{Fp, Fp2};

    const THREE: Fp2 = Fp2::new(Fp::from_const_u8(3), Fp::ZERO);
    const B: Fp2 = Fp2::new(Fp::from_const_u8(4), Fp::from_const_u8(4));
    impl_sw_affine!(G2Affine, Fp2, THREE, B);
    impl_sw_group_ops!(G2Affine, Fp2);
}

impl PairingIntrinsics for Bls12_381 {
    type Fp = Fp;
    type Fp2 = Fp2;
    type Fp12 = Fp12;

    const PAIRING_IDX: usize = 1;
    // The sextic extension `Fp12` is `Fp2[X] / (X^6 - \xi)`, where `\xi` is a non-residue.
    const XI: Fp2 = Fp2::new(Fp::from_const_u8(1), Fp::from_const_u8(1));
    const FP2_TWO: Fp2 = Fp2::new(Fp::from_const_u8(2), Fp::from_const_u8(0));
    const FP2_THREE: Fp2 = Fp2::new(Fp::from_const_u8(3), Fp::from_const_u8(0));

    // Multiplication constants for the Frobenius map for coefficients in Fp2 c1..=c5 for powers
    // 0..12 FROBENIUS_COEFFS\[i\]\[j\] = \xi^{(j + 1) * (p^i - 1)/6} when p = 1 (mod 6)
    // These are validated against `halo2curves::bls12_381::FROBENIUS_COEFF_FQ12_C1` in tests.rs
    const FROBENIUS_COEFFS: [[Self::Fp2; 5]; 12] = [
        [
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
                    ")),
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
                    ")),
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
                    ")),
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
                    ")),
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
                    ")),
            },
        ],
        [
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "b85f2392ed75078d3d81e7633da57ef6c4b9ba84d743247b4f5fbd3cfd03d60f1f0d2c20b4be31c26706bb02bfd30419"
                )),
                c1: Bls12_381Fp(hex!(
                    "f34adc6d128af72cc27e6c4dc15a2d285f3cf671c98e0cec6fb3c7b68747a154b89f1f2302e9e98832e0c4362b3efc00"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "acaa00000000fd8bfdff494feb2794409b5fb80f65297d89d49a75897d850daa85ded463864002ec99e67f39ea11011a"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "09cce3edfb8410c8f405ec722f9967eec5419200176ef7775e43d3c2ab5d3948fe7fd16b6de331680b40ff37040eaf06"
                )),
                c1: Bls12_381Fp(hex!(
                    "09cce3edfb8410c8f405ec722f9967eec5419200176ef7775e43d3c2ab5d3948fe7fd16b6de331680b40ff37040eaf06"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "adaa00000000fd8bfdff494feb2794409b5fb80f65297d89d49a75897d850daa85ded463864002ec99e67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "16810780e9fa189b32877f256e3e3ac666059c8e4ddfea8bee8f0b0c241698f345e0b1486bfa47dfd85f3a01d9cfb205"
                )),
                c1: Bls12_381Fp(hex!(
                    "9529f87f1605e61ecd78d48b90c17158bdf0146853f345dbd08279e76035df7091cc99fa4aadd36bc186453811424e14"
                ))
            },
        ],
        [
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "fffffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "fefffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "aaaafffffffffeb9ffff53b1feffab1e24f6b0f6a0d23067bf1285f3844b7764d7ac4b43b6a71b4b9ae67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "acaa00000000fd8bfdff494feb2794409b5fb80f65297d89d49a75897d850daa85ded463864002ec99e67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "adaa00000000fd8bfdff494feb2794409b5fb80f65297d89d49a75897d850daa85ded463864002ec99e67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
        ],
        [
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "a2de1b12047beef10afa673ecf6644305eb41ef6896439ef60cfb130d9ed3d1cd92c7ad748c4e9e28ea68001e6035213"
                )),
                c1: Bls12_381Fp(hex!(
                    "09cce3edfb8410c8f405ec722f9967eec5419200176ef7775e43d3c2ab5d3948fe7fd16b6de331680b40ff37040eaf06"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "a2de1b12047beef10afa673ecf6644305eb41ef6896439ef60cfb130d9ed3d1cd92c7ad748c4e9e28ea68001e6035213"
                )),
                c1: Bls12_381Fp(hex!(
                    "a2de1b12047beef10afa673ecf6644305eb41ef6896439ef60cfb130d9ed3d1cd92c7ad748c4e9e28ea68001e6035213"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "aaaafffffffffeb9ffff53b1feffab1e24f6b0f6a0d23067bf1285f3844b7764d7ac4b43b6a71b4b9ae67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "09cce3edfb8410c8f405ec722f9967eec5419200176ef7775e43d3c2ab5d3948fe7fd16b6de331680b40ff37040eaf06"
                )),
                c1: Bls12_381Fp(hex!(
                    "a2de1b12047beef10afa673ecf6644305eb41ef6896439ef60cfb130d9ed3d1cd92c7ad748c4e9e28ea68001e6035213"
                ))
            },
        ],
        [
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "fefffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "acaa00000000fd8bfdff494feb2794409b5fb80f65297d89d49a75897d850daa85ded463864002ec99e67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "fefffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "acaa00000000fd8bfdff494feb2794409b5fb80f65297d89d49a75897d850daa85ded463864002ec99e67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
        ],
        [
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "9529f87f1605e61ecd78d48b90c17158bdf0146853f345dbd08279e76035df7091cc99fa4aadd36bc186453811424e14"
                )),
                c1: Bls12_381Fp(hex!(
                    "16810780e9fa189b32877f256e3e3ac666059c8e4ddfea8bee8f0b0c241698f345e0b1486bfa47dfd85f3a01d9cfb205"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "fefffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "09cce3edfb8410c8f405ec722f9967eec5419200176ef7775e43d3c2ab5d3948fe7fd16b6de331680b40ff37040eaf06"
                )),
                c1: Bls12_381Fp(hex!(
                    "09cce3edfb8410c8f405ec722f9967eec5419200176ef7775e43d3c2ab5d3948fe7fd16b6de331680b40ff37040eaf06"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "fffffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "f34adc6d128af72cc27e6c4dc15a2d285f3cf671c98e0cec6fb3c7b68747a154b89f1f2302e9e98832e0c4362b3efc00"
                )),
                c1: Bls12_381Fp(hex!(
                    "b85f2392ed75078d3d81e7633da57ef6c4b9ba84d743247b4f5fbd3cfd03d60f1f0d2c20b4be31c26706bb02bfd30419"
                ))
            },
        ],
        [
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "aaaafffffffffeb9ffff53b1feffab1e24f6b0f6a0d23067bf1285f3844b7764d7ac4b43b6a71b4b9ae67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "aaaafffffffffeb9ffff53b1feffab1e24f6b0f6a0d23067bf1285f3844b7764d7ac4b43b6a71b4b9ae67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "aaaafffffffffeb9ffff53b1feffab1e24f6b0f6a0d23067bf1285f3844b7764d7ac4b43b6a71b4b9ae67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
        ],
        [
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "f34adc6d128af72cc27e6c4dc15a2d285f3cf671c98e0cec6fb3c7b68747a154b89f1f2302e9e98832e0c4362b3efc00"
                )),
                c1: Bls12_381Fp(hex!(
                    "b85f2392ed75078d3d81e7633da57ef6c4b9ba84d743247b4f5fbd3cfd03d60f1f0d2c20b4be31c26706bb02bfd30419"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "acaa00000000fd8bfdff494feb2794409b5fb80f65297d89d49a75897d850daa85ded463864002ec99e67f39ea11011a"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "a2de1b12047beef10afa673ecf6644305eb41ef6896439ef60cfb130d9ed3d1cd92c7ad748c4e9e28ea68001e6035213"
                )),
                c1: Bls12_381Fp(hex!(
                    "a2de1b12047beef10afa673ecf6644305eb41ef6896439ef60cfb130d9ed3d1cd92c7ad748c4e9e28ea68001e6035213"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "adaa00000000fd8bfdff494feb2794409b5fb80f65297d89d49a75897d850daa85ded463864002ec99e67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "9529f87f1605e61ecd78d48b90c17158bdf0146853f345dbd08279e76035df7091cc99fa4aadd36bc186453811424e14"
                )),
                c1: Bls12_381Fp(hex!(
                    "16810780e9fa189b32877f256e3e3ac666059c8e4ddfea8bee8f0b0c241698f345e0b1486bfa47dfd85f3a01d9cfb205"
                ))
            },
        ],
        [
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "acaa00000000fd8bfdff494feb2794409b5fb80f65297d89d49a75897d850daa85ded463864002ec99e67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "fefffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "acaa00000000fd8bfdff494feb2794409b5fb80f65297d89d49a75897d850daa85ded463864002ec99e67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "fefffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
        ],
        [
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "09cce3edfb8410c8f405ec722f9967eec5419200176ef7775e43d3c2ab5d3948fe7fd16b6de331680b40ff37040eaf06"
                )),
                c1: Bls12_381Fp(hex!(
                    "a2de1b12047beef10afa673ecf6644305eb41ef6896439ef60cfb130d9ed3d1cd92c7ad748c4e9e28ea68001e6035213"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "09cce3edfb8410c8f405ec722f9967eec5419200176ef7775e43d3c2ab5d3948fe7fd16b6de331680b40ff37040eaf06"
                )),
                c1: Bls12_381Fp(hex!(
                    "09cce3edfb8410c8f405ec722f9967eec5419200176ef7775e43d3c2ab5d3948fe7fd16b6de331680b40ff37040eaf06"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "aaaafffffffffeb9ffff53b1feffab1e24f6b0f6a0d23067bf1285f3844b7764d7ac4b43b6a71b4b9ae67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "a2de1b12047beef10afa673ecf6644305eb41ef6896439ef60cfb130d9ed3d1cd92c7ad748c4e9e28ea68001e6035213"
                )),
                c1: Bls12_381Fp(hex!(
                    "09cce3edfb8410c8f405ec722f9967eec5419200176ef7775e43d3c2ab5d3948fe7fd16b6de331680b40ff37040eaf06"
                ))
            },
        ],
        [
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "adaa00000000fd8bfdff494feb2794409b5fb80f65297d89d49a75897d850daa85ded463864002ec99e67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "acaa00000000fd8bfdff494feb2794409b5fb80f65297d89d49a75897d850daa85ded463864002ec99e67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "aaaafffffffffeb9ffff53b1feffab1e24f6b0f6a0d23067bf1285f3844b7764d7ac4b43b6a71b4b9ae67f39ea11011a"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "fefffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "fffffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
        ],
        [
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "16810780e9fa189b32877f256e3e3ac666059c8e4ddfea8bee8f0b0c241698f345e0b1486bfa47dfd85f3a01d9cfb205"
                )),
                c1: Bls12_381Fp(hex!(
                    "9529f87f1605e61ecd78d48b90c17158bdf0146853f345dbd08279e76035df7091cc99fa4aadd36bc186453811424e14"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "fefffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "a2de1b12047beef10afa673ecf6644305eb41ef6896439ef60cfb130d9ed3d1cd92c7ad748c4e9e28ea68001e6035213"
                )),
                c1: Bls12_381Fp(hex!(
                    "a2de1b12047beef10afa673ecf6644305eb41ef6896439ef60cfb130d9ed3d1cd92c7ad748c4e9e28ea68001e6035213"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "fffffeffffff012e02000a6213d817de8896f8e63ba9b3ddea770f6a07c669ba51ce76df2f67195f0000000000000000"
                )),
                c1: Bls12_381Fp(hex!(
                    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                ))
            },
            Fp2 {
                c0: Bls12_381Fp(hex!(
                    "b85f2392ed75078d3d81e7633da57ef6c4b9ba84d743247b4f5fbd3cfd03d60f1f0d2c20b4be31c26706bb02bfd30419"
                )),
                c1: Bls12_381Fp(hex!(
                    "f34adc6d128af72cc27e6c4dc15a2d285f3cf671c98e0cec6fb3c7b68747a154b89f1f2302e9e98832e0c4362b3efc00"
                ))
            },
        ],
    ];
}

impl Bls12_381 {
    // FINAL_EXPONENT = (p^12 - 1) / r in big-endian
    // Validated by a test in test.rs
    pub const FINAL_EXPONENT: [u8; 540] = hex!(
        "02ee1db5dcc825b7e1bda9c0496a1c0a89ee0193d4977b3f7d4507d07363baa13f8d14a917848517badc3a43d1073776ab353f2c30698e8cc7deada9c0aadff5e9cfee9a074e43b9a660835cc872ee83ff3a0f0f1c0ad0d6106feaf4e347aa68ad49466fa927e7bb9375331807a0dce2630d9aa4b113f414386b0e8819328148978e2b0dd39099b86e1ab656d2670d93e4d7acdd350da5359bc73ab61a0c5bf24c374693c49f570bcd2b01f3077ffb10bf24dde41064837f27611212596bc293c8d4c01f25118790f4684d0b9c40a68eb74bb22a40ee7169cdc1041296532fef459f12438dfc8e2886ef965e61a474c5c85b0129127a1b5ad0463434724538411d1676a53b5a62eb34c05739334f46c02c3f0bd0c55d3109cd15948d0a1fad20044ce6ad4c6bec3ec03ef19592004cedd556952c6d8823b19dadd7c2498345c6e5308f1c511291097db60b1749bf9b71a9f9e0100418a3ef0bc627751bbd81367066bca6a4c1b6dcfc5cceb73fc56947a403577dfa9e13c24ea820b09c1d9f7c31759c3635de3f7a3639991708e88adce88177456c49637fd7961be1a4c7e79fb02faa732e2f3ec2bea83d196283313492caa9d4aff1c910e9622d2a73f62537f2701aaef6539314043f7bbce5b78c7869aeb2181a67e49eeed2161daf3f881bd88592d767f67c4717489119226c2f011d4cab803e9d71650a6f80698e2f8491d12191a04406fbc8fbd5f48925f98630e68bfb24c0bcb9b55df57510"
    );
}
