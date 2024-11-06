use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use hex_literal::hex;
#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::BigUint;

#[cfg(not(target_os = "zkvm"))]
use crate::intrinsics::{biguint_to_limbs, uint_mod_inverse};

/// Number of limbs to represent the BN256 prime.
pub const BN256_LIMBS: usize = 32;
/// Number of limbs to represent the BLS12-381 prime.
pub const BLS12_381_LIMBS: usize = 48;

/// Trait definition for AXVM Fp2s, which take the form c0 + c1 * u where field
/// Fp2 = Fp[u]/(u^2 + 1).
pub trait Fp2<const LIMBS: usize>: Clone + Sized {
    /// Creates a new Fp2 from two arrays of bytes.
    fn from_bytes(bytes: ([u8; LIMBS], [u8; LIMBS])) -> Self;

    /// Creates a new Fp2 from two u32s of bytes.
    fn from_u32(vals: (u32, u32)) -> Self;

    /// Value of c0 and c1 as arrays of bytes.
    fn as_bytes(&self) -> (&[u8; LIMBS], &[u8; LIMBS]);

    /// Returns MODULUS (i.e. p) as an array of bytes.
    fn modulus() -> [u8; LIMBS];

    /// Returns the funct3 code for this Fp2 implementation.
    #[cfg(target_os = "zkvm")]
    fn get_funct3() -> usize;

    /// Returns MODULUS (i.e. p) as a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    fn modulus_biguint() -> BigUint;

    /// Creates a new Fp2 from two BigUints.
    #[cfg(not(target_os = "zkvm"))]
    fn from_biguint((c0, c1): (BigUint, BigUint)) -> Self {
        Self::from_bytes((biguint_to_limbs(&c0), biguint_to_limbs(&c1)))
    }

    /// Value of c0 and c1 as BigUints.
    #[cfg(not(target_os = "zkvm"))]
    fn as_biguint(&self) -> (BigUint, BigUint) {
        let (c0_bytes, c1_bytes) = self.as_bytes();
        (
            BigUint::from_bytes_le(c0_bytes),
            BigUint::from_bytes_le(c1_bytes),
        )
    }

    /// Implementation of AddAssign.
    #[inline(always)]
    fn add_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let (c0, c1) = self.as_biguint();
            let (d0, d1) = other.as_biguint();
            let modulus = Self::modulus_biguint();
            *self = Self::from_biguint(((&c0 + &d0) % &modulus, (&c1 + &d1) % &modulus));
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    /// /// Implementation of SubAssign.
    #[inline(always)]
    fn sub_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let (c0, c1) = self.as_biguint();
            let (d0, d1) = other.as_biguint();
            let modulus = Self::modulus_biguint();
            *self = Self::from_biguint((
                (&c0 + &modulus - &d0) % &modulus,
                (&c1 + &modulus - &d1) % &modulus,
            ));
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    /// Implementation of MulAssign.
    #[inline(always)]
    fn mul_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let (c0, c1) = self.as_biguint();
            let (d0, d1) = other.as_biguint();
            let modulus = Self::modulus_biguint();
            *self = Self::from_biguint((
                ((&c0 * &d0) % &modulus + &modulus - (&c1 * &d1) % &modulus) % &modulus,
                (&c0 * &d1 + &c1 * &d0) % &modulus,
            ));
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    /// Implementation of DivAssign.
    #[inline(always)]
    fn div_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let (c0, c1) = self.as_biguint();
            let (d0, d1) = other.as_biguint();
            let modulus = Self::modulus_biguint();

            let denom = uint_mod_inverse(&((&d0 * &d0 + &d1 * &d1) % &modulus), &modulus);

            *self = Self::from_biguint((
                &denom * (&c0 * &d0 + &c1 * &d1) % &modulus,
                &denom * ((&c1 * &d0) % &modulus + &modulus - (&c0 * &d1) % &modulus) % &modulus,
            ));
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    /// Implementation of Add that doesn't cause zkvm to use an additional store.
    #[inline(always)]
    fn add_no_store_impl(&self, other: &Self) -> Self {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res.add_assign_impl(other);
            res
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    /// Implementation of Sub that doesn't cause zkvm to use an additional store.
    #[inline(always)]
    fn sub_no_store_impl(&self, other: &Self) -> Self {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res.sub_assign_impl(other);
            res
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    /// Implementation of Mul that doesn't cause zkvm to use an additional store.
    #[inline(always)]
    fn mul_no_store_impl(&self, other: &Self) -> Self {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res.mul_assign_impl(other);
            res
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    /// Implementation of Div that doesn't cause zkvm to use an additional store.
    #[inline(always)]
    fn div_no_store_impl(&self, other: &Self) -> Self {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res.div_assign_impl(other);
            res
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }
}

/// Fp2 implementation that uses the BN254 prime.
#[derive(Clone)]
#[repr(C, align(64))]
pub struct Fp2Bn254([[u8; BN256_LIMBS]; 2]);

impl Fp2Bn254 {
    const MODULUS: [u8; BN256_LIMBS] =
        hex!("47FD7CD8 168C203C 8DCA7168 916A8197 5D588181 B64550B8 29A031E1 724E6430");

    /// Zero element of this field.
    pub const ZERO: Self = Self([[0; BN256_LIMBS]; 2]);
}

impl Fp2<BN256_LIMBS> for Fp2Bn254 {
    fn from_bytes((c0_bytes, c1_bytes): ([u8; BN256_LIMBS], [u8; BN256_LIMBS])) -> Self {
        Self([c0_bytes, c1_bytes])
    }

    fn from_u32((c0, c1): (u32, u32)) -> Self {
        let mut c = [[0; BN256_LIMBS]; 2];
        c[0][..4].copy_from_slice(&c0.to_le_bytes());
        c[1][..4].copy_from_slice(&c1.to_le_bytes());
        Self(c)
    }

    fn as_bytes(&self) -> (&[u8; BN256_LIMBS], &[u8; BN256_LIMBS]) {
        (&self.0[0], &self.0[1])
    }

    fn modulus() -> [u8; BN256_LIMBS] {
        Self::MODULUS
    }

    #[cfg(target_os = "zkvm")]
    fn get_funct3() -> usize {
        Custom2Funct3::Fp2Bn254 as usize
    }

    #[cfg(not(target_os = "zkvm"))]
    fn modulus_biguint() -> BigUint {
        BigUint::from_bytes_le(&Self::MODULUS)
    }
}

impl<'a> AddAssign<&'a Fp2Bn254> for Fp2Bn254 {
    #[inline(always)]
    fn add_assign(&mut self, other: &'a Fp2Bn254) {
        self.add_assign_impl(other);
    }
}

impl AddAssign for Fp2Bn254 {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        self.add_assign_impl(&other);
    }
}

impl Add for Fp2Bn254 {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl<'a> Add<&'a Fp2Bn254> for Fp2Bn254 {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: &'a Fp2Bn254) -> Self::Output {
        self += other;
        self
    }
}

impl<'a> Add<&'a Fp2Bn254> for &Fp2Bn254 {
    type Output = Fp2Bn254;
    #[inline(always)]
    fn add(self, other: &'a Fp2Bn254) -> Self::Output {
        self.add_no_store_impl(other)
    }
}

impl<'a> SubAssign<&'a Fp2Bn254> for Fp2Bn254 {
    #[inline(always)]
    fn sub_assign(&mut self, other: &'a Fp2Bn254) {
        self.sub_assign_impl(other);
    }
}

impl SubAssign for Fp2Bn254 {
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign_impl(&other);
    }
}

impl Sub for Fp2Bn254 {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, other: Self) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a> Sub<&'a Fp2Bn254> for Fp2Bn254 {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, other: &'a Fp2Bn254) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a> Sub<&'a Fp2Bn254> for &Fp2Bn254 {
    type Output = Fp2Bn254;
    #[inline(always)]
    fn sub(self, other: &'a Fp2Bn254) -> Self::Output {
        self.sub_no_store_impl(other)
    }
}

impl<'a> MulAssign<&'a Fp2Bn254> for Fp2Bn254 {
    #[inline(always)]
    fn mul_assign(&mut self, other: &'a Fp2Bn254) {
        self.mul_assign_impl(other);
    }
}

impl MulAssign for Fp2Bn254 {
    #[inline(always)]
    fn mul_assign(&mut self, other: Self) {
        self.mul_assign_impl(&other);
    }
}

impl Mul for Fp2Bn254 {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: Self) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a> Mul<&'a Fp2Bn254> for Fp2Bn254 {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: &'a Fp2Bn254) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a> Mul<&'a Fp2Bn254> for &Fp2Bn254 {
    type Output = Fp2Bn254;
    #[inline(always)]
    fn mul(self, other: &'a Fp2Bn254) -> Self::Output {
        self.mul_no_store_impl(other)
    }
}

impl<'a> DivAssign<&'a Fp2Bn254> for Fp2Bn254 {
    #[inline(always)]
    fn div_assign(&mut self, other: &'a Fp2Bn254) {
        self.div_assign_impl(other);
    }
}

impl DivAssign for Fp2Bn254 {
    #[inline(always)]
    fn div_assign(&mut self, other: Self) {
        self.div_assign_impl(&other);
    }
}

impl Div for Fp2Bn254 {
    type Output = Self;
    #[inline(always)]
    fn div(mut self, other: Self) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a> Div<&'a Fp2Bn254> for Fp2Bn254 {
    type Output = Self;
    #[inline(always)]
    fn div(mut self, other: &'a Fp2Bn254) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a> Div<&'a Fp2Bn254> for &Fp2Bn254 {
    type Output = Fp2Bn254;
    #[inline(always)]
    fn div(self, other: &'a Fp2Bn254) -> Self::Output {
        self.div_no_store_impl(other)
    }
}

/// Fp2 implementation that uses the BLS12-381 prime.
#[derive(Clone)]
#[repr(C, align(64))]
pub struct Fp2Bls12381([[u8; BLS12_381_LIMBS]; 2]);

impl Fp2Bls12381 {
    const MODULUS: [u8; BLS12_381_LIMBS] =
        hex!("ABAAFFFF FFFFFEB9 FFFF53B1 FEFFAB1E 24F6B0F6 A0D23067 BF1285F3 844B7764 D7AC4B43 B6A71B4B 9AE67F39 EA11011A");

    /// Zero element of this field.
    pub const ZERO: Self = Self([[0; BLS12_381_LIMBS]; 2]);
}

impl Fp2<BLS12_381_LIMBS> for Fp2Bls12381 {
    fn from_bytes((c0_bytes, c1_bytes): ([u8; BLS12_381_LIMBS], [u8; BLS12_381_LIMBS])) -> Self {
        Self([c0_bytes, c1_bytes])
    }

    fn from_u32((c0, c1): (u32, u32)) -> Self {
        let mut c = [[0; BLS12_381_LIMBS]; 2];
        c[0][..4].copy_from_slice(&c0.to_le_bytes());
        c[1][..4].copy_from_slice(&c1.to_le_bytes());
        Self(c)
    }

    fn as_bytes(&self) -> (&[u8; BLS12_381_LIMBS], &[u8; BLS12_381_LIMBS]) {
        (&self.0[0], &self.0[1])
    }

    fn modulus() -> [u8; BLS12_381_LIMBS] {
        Self::MODULUS
    }

    #[cfg(target_os = "zkvm")]
    fn get_funct3() -> usize {
        Custom2Funct3::Fp2Bls12381 as usize
    }

    #[cfg(not(target_os = "zkvm"))]
    fn modulus_biguint() -> BigUint {
        BigUint::from_bytes_le(&Self::MODULUS)
    }
}

impl<'a> AddAssign<&'a Fp2Bls12381> for Fp2Bls12381 {
    #[inline(always)]
    fn add_assign(&mut self, other: &'a Fp2Bls12381) {
        self.add_assign_impl(other);
    }
}

impl AddAssign for Fp2Bls12381 {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        self.add_assign_impl(&other);
    }
}

impl Add for Fp2Bls12381 {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl<'a> Add<&'a Fp2Bls12381> for Fp2Bls12381 {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: &'a Fp2Bls12381) -> Self::Output {
        self += other;
        self
    }
}

impl<'a> Add<&'a Fp2Bls12381> for &Fp2Bls12381 {
    type Output = Fp2Bls12381;
    #[inline(always)]
    fn add(self, other: &'a Fp2Bls12381) -> Self::Output {
        self.add_no_store_impl(other)
    }
}

impl<'a> SubAssign<&'a Fp2Bls12381> for Fp2Bls12381 {
    #[inline(always)]
    fn sub_assign(&mut self, other: &'a Fp2Bls12381) {
        self.sub_assign_impl(other);
    }
}

impl SubAssign for Fp2Bls12381 {
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign_impl(&other);
    }
}

impl Sub for Fp2Bls12381 {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, other: Self) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a> Sub<&'a Fp2Bls12381> for Fp2Bls12381 {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, other: &'a Fp2Bls12381) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a> Sub<&'a Fp2Bls12381> for &Fp2Bls12381 {
    type Output = Fp2Bls12381;
    #[inline(always)]
    fn sub(self, other: &'a Fp2Bls12381) -> Self::Output {
        self.sub_no_store_impl(other)
    }
}

impl<'a> MulAssign<&'a Fp2Bls12381> for Fp2Bls12381 {
    #[inline(always)]
    fn mul_assign(&mut self, other: &'a Fp2Bls12381) {
        self.mul_assign_impl(other);
    }
}

impl MulAssign for Fp2Bls12381 {
    #[inline(always)]
    fn mul_assign(&mut self, other: Self) {
        self.mul_assign_impl(&other);
    }
}

impl Mul for Fp2Bls12381 {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: Self) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a> Mul<&'a Fp2Bls12381> for Fp2Bls12381 {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: &'a Fp2Bls12381) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a> Mul<&'a Fp2Bls12381> for &Fp2Bls12381 {
    type Output = Fp2Bls12381;
    #[inline(always)]
    fn mul(self, other: &'a Fp2Bls12381) -> Self::Output {
        self.mul_no_store_impl(other)
    }
}

impl<'a> DivAssign<&'a Fp2Bls12381> for Fp2Bls12381 {
    #[inline(always)]
    fn div_assign(&mut self, other: &'a Fp2Bls12381) {
        self.div_assign_impl(other);
    }
}

impl DivAssign for Fp2Bls12381 {
    #[inline(always)]
    fn div_assign(&mut self, other: Self) {
        self.div_assign_impl(&other);
    }
}

impl Div for Fp2Bls12381 {
    type Output = Self;
    #[inline(always)]
    fn div(mut self, other: Self) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a> Div<&'a Fp2Bls12381> for Fp2Bls12381 {
    type Output = Self;
    #[inline(always)]
    fn div(mut self, other: &'a Fp2Bls12381) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a> Div<&'a Fp2Bls12381> for &Fp2Bls12381 {
    type Output = Fp2Bls12381;
    #[inline(always)]
    fn div(self, other: &'a Fp2Bls12381) -> Self::Output {
        self.div_no_store_impl(other)
    }
}
