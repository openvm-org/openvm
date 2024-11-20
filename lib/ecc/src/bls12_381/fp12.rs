#[cfg(target_os = "zkvm")]
use core::mem::MaybeUninit;
use core::ops::{Mul, MulAssign, Neg};

use axvm_algebra::{
    field::{ComplexConjugate, FieldExtension},
    DivAssignUnsafe, DivUnsafe, Field,
};

use super::{Bls12_381, Fp, Fp2};
#[cfg(not(target_os = "zkvm"))]
use crate::pairing::PairingIntrinsics;
use crate::pairing::{fp12_invert_assign, SexticExtField};

pub type Fp12 = SexticExtField<Fp2>;

impl Fp12 {
    pub fn invert(&self) -> Self {
        let mut s = self.clone();
        fp12_invert_assign::<Fp, Fp2>(&mut s.c, &Bls12_381::XI);
        SexticExtField::new(s.c)
    }

    pub fn div_assign_unsafe_impl(&mut self, other: &Self) {
        *self *= other.invert();
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

    fn to_coeffs(self) -> Self::Coeffs {
        self.c
    }

    fn embed(c0: Fp2) -> Self {
        Self::new([c0, Fp2::ZERO, Fp2::ZERO, Fp2::ZERO, Fp2::ZERO, Fp2::ZERO])
    }

    fn frobenius_map(&self, _power: usize) -> Self {
        todo!()
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
    fn mul_assign(&mut self, other: &'a Fp12) {
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = crate::pairing::sextic_tower_mul_host(self, other, &Bls12_381::XI);
        }
        #[cfg(target_os = "zkvm")]
        {
            crate::pairing::sextic_tower_mul_intrinsic::<Bls12_381>(
                self as *mut Fp12 as *mut u8,
                self as *const Fp12 as *const u8,
                other as *const Fp12 as *const u8,
            );
        }
    }
}

impl<'a> Mul<&'a Fp12> for &'a Fp12 {
    type Output = Fp12;

    fn mul(self, other: &'a Fp12) -> Self::Output {
        #[cfg(not(target_os = "zkvm"))]
        {
            crate::pairing::sextic_tower_mul_host(self, other, &Bls12_381::XI)
        }
        #[cfg(target_os = "zkvm")]
        unsafe {
            let mut uninit: MaybeUninit<Self::Output> = MaybeUninit::uninit();
            crate::pairing::sextic_tower_mul_intrinsic::<Bls12_381>(
                uninit.as_mut_ptr() as *mut u8,
                self as *const Fp12 as *const u8,
                other as *const Fp12 as *const u8,
            );
            uninit.assume_init()
        }
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
    fn mul(mut self, other: &'a Fp12) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a> DivAssignUnsafe<&'a Fp12> for Fp12 {
    fn div_assign_unsafe(&mut self, other: &'a Fp12) {
        self.div_assign_unsafe_impl(other);
    }
}

impl<'a> DivUnsafe<&'a Fp12> for &'a Fp12 {
    type Output = Fp12;

    fn div_unsafe(self, other: &'a Fp12) -> Self::Output {
        let mut res = self.clone();
        res.div_assign_unsafe_impl(other);
        res
    }
}

impl DivAssignUnsafe for Fp12 {
    #[inline(always)]
    fn div_assign_unsafe(&mut self, other: Self) {
        self.div_assign_unsafe_impl(&other);
    }
}

impl DivUnsafe for Fp12 {
    type Output = Self;
    #[inline(always)]
    fn div_unsafe(self, other: Self) -> Self::Output {
        let mut res = self.clone();
        res.div_assign_unsafe_impl(&other);
        res
    }
}

impl<'a> DivUnsafe<&'a Fp12> for Fp12 {
    type Output = Self;
    #[inline(always)]
    fn div_unsafe(self, other: &'a Fp12) -> Self::Output {
        let mut res = self.clone();
        res.div_assign_unsafe_impl(other);
        res
    }
}

impl Neg for Fp12 {
    type Output = Fp12;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::ZERO - &self
    }
}
