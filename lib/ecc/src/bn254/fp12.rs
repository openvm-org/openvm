#[cfg(target_os = "zkvm")]
use core::mem::MaybeUninit;
use core::ops::{Mul, MulAssign, Neg};

use axvm_algebra::{
    field::{ComplexConjugate, FieldExtension},
    DivAssignUnsafe, DivUnsafe, Field,
};

use super::{
    fp6_invert_assign, fp6_mul_assign, fp6_mul_by_nonresidue_assign, fp6_neg_assign,
    fp6_square_assign, fp6_sub_assign, Bn254, Fp2,
};
#[cfg(not(target_os = "zkvm"))]
use crate::pairing::PairingIntrinsics;
use crate::pairing::SexticExtField;

pub type Fp12 = SexticExtField<Fp2>;

impl Fp12 {
    pub fn invert(&self) -> Self {
        let mut c0s = [self.c[0].clone(), self.c[1].clone(), self.c[2].clone()];
        let mut c1s = [self.c[3].clone(), self.c[4].clone(), self.c[5].clone()];

        fp6_square_assign(&mut c0s);
        fp6_square_assign(&mut c1s);
        fp6_mul_by_nonresidue_assign(&mut c1s);
        fp6_sub_assign(&mut c0s, &c1s);

        fp6_invert_assign(&mut c0s);
        let mut t0 = c0s.clone();
        let mut t1 = c0s;
        fp6_mul_assign(
            &mut t0,
            &[self.c[0].clone(), self.c[1].clone(), self.c[2].clone()],
        );
        fp6_mul_assign(
            &mut t1,
            &[self.c[3].clone(), self.c[4].clone(), self.c[5].clone()],
        );
        fp6_neg_assign(&mut t1);
        Fp12::new([
            t0[0].clone(),
            t0[1].clone(),
            t0[2].clone(),
            t1[0].clone(),
            t1[1].clone(),
            t1[2].clone(),
        ])
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
            *self = crate::pairing::sextic_tower_mul_host(self, other, &Bn254::XI);
        }
        #[cfg(target_os = "zkvm")]
        {
            crate::pairing::sextic_tower_mul_intrinsic::<Bn254>(
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
            crate::pairing::sextic_tower_mul_host(self, other, &Bn254::XI)
        }
        #[cfg(target_os = "zkvm")]
        unsafe {
            let mut uninit: MaybeUninit<Self::Output> = MaybeUninit::uninit();
            crate::pairing::sextic_tower_mul_intrinsic::<Bn254>(
                uninit.as_mut_ptr() as *mut u8,
                self as *const Fp12 as *const u8,
                other as *const Fp12 as *const u8,
            );
            uninit.assume_init()
        }
    }
}

// TODO[jpw]: make this into a macro

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
    fn div_assign_unsafe(&mut self, other: &'a Fp12) {
        self.div_assign_unsafe_impl(other);
    }
}

impl<'a> DivUnsafe<&'a Fp12> for &'a Fp12 {
    type Output = Fp12;

    fn div_unsafe(self, other: &'a Fp12) -> Self::Output {
        let mut res = self.clone();
        res.div_assign_unsafe_impl(&other);
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
