use core::ops::{Mul, MulAssign};

use axvm_algebra::{
    field::{ComplexConjugate, FieldExtension},
    Field,
};
#[cfg(target_os = "zkvm")]
use {
    axvm_platform::constants::{Custom1Funct3, PairingBaseFunct7, CUSTOM_1},
    axvm_platform::custom_insn_r,
    core::mem::MaybeUninit,
};

use super::{Bn254, Fp2};
use crate::pairing::{PairingIntrinsics as _, SexticExtField};

pub type Fp12 = SexticExtField<Fp2>;

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
        unsafe {
            crate::pairing::sextic_tower_mul_intrinsic::<Bn254>(
                self as *mut u8,
                self as *const u8,
                other as *const u8,
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
            crate::pairing::sextic_tower_mul_host::<Bn254>(
                uninit.as_mut_ptr() as *mut u8,
                self as *const u8,
                other as *const u8,
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
