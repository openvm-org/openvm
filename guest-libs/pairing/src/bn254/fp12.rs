extern crate alloc;

use alloc::vec::Vec;
use core::ops::{Mul, MulAssign, Neg};

use openvm_algebra_guest::{
    field::{ComplexConjugate, FieldExtension},
    DivAssignUnsafe, DivUnsafe, Field,
};
use openvm_pairing_guest::pairing::PairingIntrinsics;

use super::{Bn254, Fp, Fp2};
use crate::operations::{fp12_invert_assign, SexticExtField};

pub type Fp12 = SexticExtField<Fp2>;

impl Fp12 {
    pub fn invert(&self) -> Self {
        let mut s = self.clone();
        fp12_invert_assign::<Fp, Fp2>(&mut s.c, &Bn254::XI);
        s
    }
}

impl Field for Fp12 {
    type SelfRef<'a> = &'a Self;
    const ZERO: Self = Self::new([Fp2::ZERO; 6]);
    const ONE: Self = Self::new([
        Fp2::ONE,
        Fp2::ZERO,
        Fp2::ZERO,
        Fp2::ZERO,
        Fp2::ZERO,
        Fp2::ZERO,
    ]);

    fn double_assign(&mut self) {
        *self += self.clone();
    }

    fn square_assign(&mut self) {
        *self *= self.clone();
    }
}

impl FieldExtension<Fp2> for Fp12 {
    const D: usize = 6;
    type Coeffs = [Fp2; 6];

    fn from_coeffs(coeffs: Self::Coeffs) -> Self {
        Self::new(coeffs)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 384);
        Self::from_coeffs([
            Fp2::from_bytes(&bytes[0..64]),
            Fp2::from_bytes(&bytes[64..128]),
            Fp2::from_bytes(&bytes[128..192]),
            Fp2::from_bytes(&bytes[192..256]),
            Fp2::from_bytes(&bytes[256..320]),
            Fp2::from_bytes(&bytes[320..384]),
        ])
    }

    fn to_coeffs(self) -> Self::Coeffs {
        self.c
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(384);
        for coeff in self.clone().to_coeffs() {
            bytes.extend_from_slice(&coeff.to_bytes());
        }
        bytes
    }

    fn embed(c0: Fp2) -> Self {
        Self::new([c0, Fp2::ZERO, Fp2::ZERO, Fp2::ZERO, Fp2::ZERO, Fp2::ZERO])
    }

    /// We assume that the frobenius map power is < 12
    fn frobenius_map(&self, power: usize) -> Self {
        if power & 1 != 0 {
            let c0 = self.c[0].clone().conjugate();
            let c1 = self.c[1].clone().conjugate() * &Bn254::FROBENIUS_COEFFS[power][0];
            let c2 = self.c[2].clone().conjugate() * &Bn254::FROBENIUS_COEFFS[power][1];
            let c3 = self.c[3].clone().conjugate() * &Bn254::FROBENIUS_COEFFS[power][2];
            let c4 = self.c[4].clone().conjugate() * &Bn254::FROBENIUS_COEFFS[power][3];
            let c5 = self.c[5].clone().conjugate() * &Bn254::FROBENIUS_COEFFS[power][4];
            Self::new([c0, c1, c2, c3, c4, c5])
        } else {
            let c0 = self.c[0].clone();
            let c1 = &self.c[1] * &Bn254::FROBENIUS_COEFFS[power][0];
            let c2 = &self.c[2] * &Bn254::FROBENIUS_COEFFS[power][1];
            let c3 = &self.c[3] * &Bn254::FROBENIUS_COEFFS[power][2];
            let c4 = &self.c[4] * &Bn254::FROBENIUS_COEFFS[power][3];
            let c5 = &self.c[5] * &Bn254::FROBENIUS_COEFFS[power][4];
            Self::new([c0, c1, c2, c3, c4, c5])
        }
    }

    fn mul_base(&self, rhs: &Fp2) -> Self {
        Self::new([
            &self.c[0] * rhs,
            &self.c[1] * rhs,
            &self.c[2] * rhs,
            &self.c[3] * rhs,
            &self.c[4] * rhs,
            &self.c[5] * rhs,
        ])
    }
}

// This is ambiguous. It is conjugation for Fp12 over Fp6.
impl ComplexConjugate for Fp12 {
    #[inline(always)]
    fn conjugate(self) -> Self {
        let [c0, c1, c2, c3, c4, c5] = self.c;
        Self::new([c0, -c1, c2, -c3, c4, -c5])
    }

    fn conjugate_assign(&mut self) {
        self.c[1].neg_assign();
        self.c[3].neg_assign();
        self.c[5].neg_assign();
    }
}

impl<'a> MulAssign<&'a Fp12> for Fp12 {
    #[inline(always)]
    fn mul_assign(&mut self, other: &'a Fp12) {
        *self = crate::operations::sextic_tower_mul(self, other, &Bn254::XI);
    }
}

impl<'a> Mul<&'a Fp12> for &'a Fp12 {
    type Output = Fp12;
    #[inline(always)]
    fn mul(self, other: &'a Fp12) -> Self::Output {
        crate::operations::sextic_tower_mul(self, other, &Bn254::XI)
    }
}

impl MulAssign for Fp12 {
    #[inline(always)]
    fn mul_assign(&mut self, other: Self) {
        self.mul_assign(&other);
    }
}

impl Mul for Fp12 {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: Self) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a> Mul<&'a Fp12> for Fp12 {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: &'a Fp12) -> Fp12 {
        self *= other;
        self
    }
}

impl<'a> DivAssignUnsafe<&'a Fp12> for Fp12 {
    #[inline(always)]
    fn div_assign_unsafe(&mut self, other: &'a Fp12) {
        *self *= other.invert();
    }
}

impl<'a> DivUnsafe<&'a Fp12> for &'a Fp12 {
    type Output = Fp12;
    #[inline(always)]
    fn div_unsafe(self, other: &'a Fp12) -> Self::Output {
        let mut res = self.clone();
        res.div_assign_unsafe(other);
        res
    }
}

impl DivAssignUnsafe for Fp12 {
    #[inline(always)]
    fn div_assign_unsafe(&mut self, other: Self) {
        *self *= other.invert();
    }
}

impl DivUnsafe for Fp12 {
    type Output = Self;
    #[inline(always)]
    fn div_unsafe(mut self, other: Self) -> Self::Output {
        self.div_assign_unsafe(other);
        self
    }
}

impl<'a> DivUnsafe<&'a Fp12> for Fp12 {
    type Output = Self;
    #[inline(always)]
    fn div_unsafe(mut self, other: &'a Fp12) -> Self::Output {
        self.div_assign_unsafe(other);
        self
    }
}

impl Neg for Fp12 {
    type Output = Fp12;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::ZERO - &self
    }
}
