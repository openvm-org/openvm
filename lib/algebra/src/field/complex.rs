use core::{
    fmt::{Debug, Formatter, Result},
    iter::{Product, Sum},
    mem::MaybeUninit,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[cfg(target_os = "zkvm")]
use axvm_platform::{
    constants::{ComplexExtFieldBaseFunct7, Custom1Funct3, COMPLEX_EXT_FIELD_MAX_KINDS, CUSTOM_1},
    custom_insn_r,
};

use super::{ComplexConjugate, Field};
use crate::{DivAssignUnsafe, DivUnsafe, IntMod};

/// Quadratic extension field of `F` with irreducible polynomial `X^2 + 1`.
/// Elements are represented as `c0 + c1 * u` where `u^2 = -1`.
///
/// Memory alignment follows alignment of `F`.
/// Memory layout is concatenation of `c0` and `c1`.
#[derive(Clone, PartialEq, Eq)]
#[repr(C)]
pub struct Complex<F> {
    /// Real coordinate
    pub c0: F,
    /// Imaginary coordinate
    pub c1: F,
}

impl<F> Complex<F> {
    pub const fn new(c0: F, c1: F) -> Self {
        Self { c0, c1 }
    }
}

impl<F: IntMod> Complex<F> {
    // Zero element (i.e. additive identity)
    pub const ZERO: Self = Self::new(F::ZERO, F::ZERO);

    // One element (i.e. multiplicative identity)
    pub const ONE: Self = Self::new(F::ONE, F::ZERO);

    pub fn neg_assign(&mut self) {
        self.c0.neg_assign();
        self.c1.neg_assign();
    }

    /// Implementation of AddAssign.
    #[inline(always)]
    fn add_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            self.c0 += &other.c0;
            self.c1 += &other.c1;
        }
        #[cfg(target_os = "zkvm")]
        {
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ComplexExtField as usize,
                ComplexExtFieldBaseFunct7::Add as usize
                    + F::MOD_IDX * (COMPLEX_EXT_FIELD_MAX_KINDS as usize),
                self as *mut Self,
                self as *const Self,
                other as *const Self
            )
        }
    }

    /// Implementation of SubAssign.
    #[inline(always)]
    fn sub_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            self.c0 -= &other.c0;
            self.c1 -= &other.c1;
        }
        #[cfg(target_os = "zkvm")]
        {
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ComplexExtField as usize,
                ComplexExtFieldBaseFunct7::Sub as usize
                    + F::MOD_IDX * (COMPLEX_EXT_FIELD_MAX_KINDS as usize),
                self as *mut Self,
                self as *const Self,
                other as *const Self
            )
        }
    }

    /// Implementation of MulAssign.
    #[inline(always)]
    fn mul_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let (c0, c1) = (&self.c0, &self.c1);
            let (d0, d1) = (&other.c0, &other.c1);
            *self = Self::new(
                c0.clone() * d0 - c1.clone() * d1,
                c0.clone() * d1 + c1.clone() * d0,
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ComplexExtField as usize,
                ComplexExtFieldBaseFunct7::Mul as usize
                    + F::MOD_IDX * (COMPLEX_EXT_FIELD_MAX_KINDS as usize),
                self as *mut Self,
                self as *const Self,
                other as *const Self
            )
        }
    }

    /// Implementation of DivAssignUnsafe.
    #[inline(always)]
    fn div_assign_unsafe_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let (c0, c1) = (&self.c0, &self.c1);
            let (d0, d1) = (&other.c0, &other.c1);
            let denom = F::ONE.div_unsafe(d0.square() + d1.square());
            *self = Self::new(
                denom.clone() * (c0.clone() * d0 + c1.clone() * d1),
                denom * &(c1.clone() * d0 - c0.clone() * d1),
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ComplexExtField as usize,
                ComplexExtFieldBaseFunct7::Div as usize
                    + F::MOD_IDX * (COMPLEX_EXT_FIELD_MAX_KINDS as usize),
                self as *mut Self,
                self as *const Self,
                other as *const Self
            )
        }
    }

    /// Implementation of Add that doesn't cause zkvm to use an additional store.
    fn add_refs_impl(&self, other: &Self) -> Self {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res.add_assign_impl(other);
            res
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<Self> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ComplexExtField as usize,
                ComplexExtFieldBaseFunct7::Add as usize
                    + F::MOD_IDX * (COMPLEX_EXT_FIELD_MAX_KINDS as usize),
                uninit.as_mut_ptr(),
                self as *const Self,
                other as *const Self
            );
            unsafe { uninit.assume_init() }
        }
    }

    /// Implementation of Sub that doesn't cause zkvm to use an additional store.
    #[inline(always)]
    fn sub_refs_impl(&self, other: &Self) -> Self {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res.sub_assign_impl(other);
            res
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<Self> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ComplexExtField as usize,
                ComplexExtFieldBaseFunct7::Sub as usize
                    + F::MOD_IDX * (COMPLEX_EXT_FIELD_MAX_KINDS as usize),
                uninit.as_mut_ptr(),
                self as *const Self,
                other as *const Self
            );
            unsafe { uninit.assume_init() }
        }
    }

    /// Implementation of Mul that doesn't cause zkvm to use an additional store.
    ///
    /// SAFETY: dst_ptr must be pointer for `&mut Self`.
    /// It will only be written to at the end of the function.
    #[inline(always)]
    unsafe fn mul_refs_impl(&self, other: &Self, dst_ptr: *mut Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res.mul_assign_impl(other);
            let dst = unsafe { &mut *dst_ptr };
            *dst = res;
        }
        #[cfg(target_os = "zkvm")]
        {
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ComplexExtField as usize,
                ComplexExtFieldBaseFunct7::Mul as usize
                    + F::MOD_IDX * (COMPLEX_EXT_FIELD_MAX_KINDS as usize),
                dst_ptr,
                self as *const Self,
                other as *const Self
            );
        }
    }

    /// Implementation of DivUnsafe that doesn't cause zkvm to use an additional store.
    #[inline(always)]
    fn div_unsafe_refs_impl(&self, other: &Self) -> Self {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res.div_assign_unsafe_impl(other);
            res
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<Self> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ComplexExtField as usize,
                ComplexExtFieldBaseFunct7::Div as usize
                    + F::MOD_IDX * (COMPLEX_EXT_FIELD_MAX_KINDS as usize),
                uninit.as_mut_ptr(),
                self as *const Self,
                other as *const Self
            );
            unsafe { uninit.assume_init() }
        }
    }
}

impl<F: IntMod> ComplexConjugate for Complex<F> {
    fn conjugate(self) -> Self {
        Self {
            c0: self.c0,
            c1: -self.c1,
        }
    }

    fn conjugate_assign(&mut self) {
        self.c1.neg_assign();
    }
}

impl<'a, F: IntMod> AddAssign<&'a Complex<F>> for Complex<F> {
    #[inline(always)]
    fn add_assign(&mut self, other: &'a Complex<F>) {
        self.add_assign_impl(other);
    }
}

impl<F: IntMod> AddAssign for Complex<F> {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        self.add_assign_impl(&other);
    }
}

impl<F: IntMod> Add for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl<'a, F: IntMod> Add<&'a Complex<F>> for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: &'a Complex<F>) -> Self::Output {
        self += other;
        self
    }
}

impl<'a, F: IntMod> Add<&'a Complex<F>> for &Complex<F> {
    type Output = Complex<F>;
    #[inline(always)]
    fn add(self, other: &'a Complex<F>) -> Self::Output {
        self.add_refs_impl(other)
    }
}

impl<'a, F: IntMod> SubAssign<&'a Complex<F>> for Complex<F> {
    #[inline(always)]
    fn sub_assign(&mut self, other: &'a Complex<F>) {
        self.sub_assign_impl(other);
    }
}

impl<F: IntMod> SubAssign for Complex<F> {
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign_impl(&other);
    }
}

impl<F: IntMod> Sub for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, other: Self) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a, F: IntMod> Sub<&'a Complex<F>> for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, other: &'a Complex<F>) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a, F: IntMod> Sub<&'a Complex<F>> for &Complex<F> {
    type Output = Complex<F>;
    #[inline(always)]
    fn sub(self, other: &'a Complex<F>) -> Self::Output {
        self.sub_refs_impl(other)
    }
}

impl<'a, F: IntMod> MulAssign<&'a Complex<F>> for Complex<F> {
    #[inline(always)]
    fn mul_assign(&mut self, other: &'a Complex<F>) {
        self.mul_assign_impl(other);
    }
}

impl<F: IntMod> MulAssign for Complex<F> {
    #[inline(always)]
    fn mul_assign(&mut self, other: Self) {
        self.mul_assign_impl(&other);
    }
}

impl<F: IntMod> Mul for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: Self) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a, F: IntMod> Mul<&'a Complex<F>> for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: &'a Complex<F>) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a, F: IntMod> Mul<&'a Complex<F>> for &'a Complex<F> {
    type Output = Complex<F>;
    #[inline(always)]
    fn mul(self, other: &'a Complex<F>) -> Self::Output {
        let mut uninit: MaybeUninit<Complex<F>> = MaybeUninit::uninit();
        unsafe {
            self.mul_refs_impl(other, uninit.as_mut_ptr());
            uninit.assume_init()
        }
    }
}

impl<'a, F: IntMod> DivAssignUnsafe<&'a Complex<F>> for Complex<F> {
    #[inline(always)]
    fn div_assign_unsafe(&mut self, other: &'a Complex<F>) {
        self.div_assign_unsafe_impl(other);
    }
}

impl<F: IntMod> DivAssignUnsafe for Complex<F> {
    #[inline(always)]
    fn div_assign_unsafe(&mut self, other: Self) {
        self.div_assign_unsafe_impl(&other);
    }
}

impl<F: IntMod> DivUnsafe for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn div_unsafe(mut self, other: Self) -> Self::Output {
        self = self.div_unsafe_refs_impl(&other);
        self
    }
}

impl<'a, F: IntMod> DivUnsafe<&'a Complex<F>> for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn div_unsafe(mut self, other: &'a Complex<F>) -> Self::Output {
        self = self.div_unsafe_refs_impl(other);
        self
    }
}

impl<'a, F: IntMod> DivUnsafe<&'a Complex<F>> for &Complex<F> {
    type Output = Complex<F>;
    #[inline(always)]
    fn div_unsafe(self, other: &'a Complex<F>) -> Self::Output {
        self.div_unsafe_refs_impl(other)
    }
}

impl<'a, F: IntMod> Sum<&'a Complex<F>> for Complex<F> {
    fn sum<I: Iterator<Item = &'a Complex<F>>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| &acc + x)
    }
}

impl<F: IntMod> Sum for Complex<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| &acc + &x)
    }
}

impl<'a, F: IntMod> Product<&'a Complex<F>> for Complex<F> {
    fn product<I: Iterator<Item = &'a Complex<F>>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| &acc * x)
    }
}

impl<F: IntMod> Product for Complex<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| &acc * &x)
    }
}

impl<F: IntMod> Neg for Complex<F> {
    type Output = Complex<F>;
    fn neg(self) -> Self::Output {
        Self::ZERO - &self
    }
}

impl<F: IntMod> Neg for &Complex<F> {
    type Output = Complex<F>;
    fn neg(self) -> Self::Output {
        Complex::ZERO - self
    }
}

impl<F: IntMod> Debug for Complex<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:?} + {:?} * u", self.c0, self.c1)
    }
}

impl<F: Field + IntMod> Field for Complex<F> {
    type SelfRef<'a>
        = &'a Self
    where
        Self: 'a;

    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    fn double_assign(&mut self) {
        *self += self.clone();
    }

    fn square_assign(&mut self) {
        unsafe {
            self.mul_refs_impl(self, self as *const Self as *mut Self);
        }
    }
}
