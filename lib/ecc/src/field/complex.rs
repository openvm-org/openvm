use core::{
    fmt::{Debug, Formatter, Result},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[cfg(target_os = "zkvm")]
use {
    axvm_platform::{
        constants::{
            ComplexExtFieldBaseFunct7, Custom1Funct3, COMPLEX_EXT_FIELD_MAX_KINDS, CUSTOM_1,
        },
        custom_insn_r,
    },
    core::mem::MaybeUninit,
};

use super::{DivAssignUnsafe, DivUnsafe, Field};

/// Quadratic extension field of `F` with irreducible polynomial `X^2 + 1`.
/// Elements are represented as `c0 + c1 * u` where `u^2 = -1`.
///
/// Memory alignment follows alignment of `F`.
/// Memory layout is concatenation of `c0` and `c1`.
#[derive(Clone, PartialEq, Eq)]
#[repr(C)]
pub struct Complex<F: Field> {
    /// Real coordinate
    pub c0: F,
    /// Imaginary coordinate
    pub c1: F,
}

impl<F: Field> Complex<F> {
    pub const fn new(c0: F, c1: F) -> Self {
        Self { c0, c1 }
    }
}

impl<F: Field> Complex<F> {
    // Zero element (i.e. additive identity)
    pub const ZERO: Self = Self::new(F::ZERO, F::ZERO);

    // One element (i.e. multiplicative identity)
    pub const ONE: Self = Self::new(F::ONE, F::ZERO);

    /// Set this complex number to be its conjugate
    pub fn conjugate_assign(&mut self) {
        self.c1 *= -F::ONE
    }

    /// Conjugate of this complex number
    pub fn conjugate(&self) -> Self {
        Self::new(self.c0.clone(), -self.c1.clone())
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
    #[inline(always)]
    fn mul_refs_impl(&self, other: &Self) -> Self {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self.clone();
            res.mul_assign_impl(other);
            res
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<Self> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::ComplexExtField as usize,
                ComplexExtFieldBaseFunct7::Mul as usize
                    + F::MOD_IDX * (COMPLEX_EXT_FIELD_MAX_KINDS as usize),
                uninit.as_mut_ptr(),
                self as *const Self,
                other as *const Self
            );
            unsafe { uninit.assume_init() }
        }
    }
}

impl<'a, F: Field> AddAssign<&'a Complex<F>> for Complex<F> {
    #[inline(always)]
    fn add_assign(&mut self, other: &'a Complex<F>) {
        self.add_assign_impl(other);
    }
}

impl<F: Field> AddAssign for Complex<F> {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        self.add_assign_impl(&other);
    }
}

impl<F: Field> Add for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl<'a, F: Field> Add<&'a Complex<F>> for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: &'a Complex<F>) -> Self::Output {
        self += other;
        self
    }
}

impl<'a, F: Field> Add<&'a Complex<F>> for &Complex<F> {
    type Output = Complex<F>;
    #[inline(always)]
    fn add(self, other: &'a Complex<F>) -> Self::Output {
        self.add_refs_impl(other)
    }
}

impl<'a, F: Field> SubAssign<&'a Complex<F>> for Complex<F> {
    #[inline(always)]
    fn sub_assign(&mut self, other: &'a Complex<F>) {
        self.sub_assign_impl(other);
    }
}

impl<F: Field> SubAssign for Complex<F> {
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign_impl(&other);
    }
}

impl<F: Field> Sub for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, other: Self) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a, F: Field> Sub<&'a Complex<F>> for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, other: &'a Complex<F>) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a, F: Field> Sub<&'a Complex<F>> for &Complex<F> {
    type Output = Complex<F>;
    #[inline(always)]
    fn sub(self, other: &'a Complex<F>) -> Self::Output {
        self.sub_refs_impl(other)
    }
}

impl<'a, F: Field> MulAssign<&'a Complex<F>> for Complex<F> {
    #[inline(always)]
    fn mul_assign(&mut self, other: &'a Complex<F>) {
        self.mul_assign_impl(other);
    }
}

impl<F: Field> MulAssign for Complex<F> {
    #[inline(always)]
    fn mul_assign(&mut self, other: Self) {
        self.mul_assign_impl(&other);
    }
}

impl<F: Field> Mul for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: Self) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a, F: Field> Mul<&'a Complex<F>> for Complex<F> {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: &'a Complex<F>) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a, F: Field> Mul<&'a Complex<F>> for &Complex<F> {
    type Output = Complex<F>;
    #[inline(always)]
    fn mul(self, other: &'a Complex<F>) -> Self::Output {
        self.mul_refs_impl(other)
    }
}

impl<'a, F: Field> Sum<&'a Complex<F>> for Complex<F> {
    fn sum<I: Iterator<Item = &'a Complex<F>>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| &acc + x)
    }
}

impl<F: Field> Sum for Complex<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| &acc + &x)
    }
}

impl<'a, F: Field> Product<&'a Complex<F>> for Complex<F> {
    fn product<I: Iterator<Item = &'a Complex<F>>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| &acc * x)
    }
}

impl<F: Field> Product for Complex<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| &acc * &x)
    }
}

impl<F: Field> Neg for Complex<F> {
    type Output = Complex<F>;
    fn neg(self) -> Self::Output {
        Self::ZERO - &self
    }
}

impl<F: Field> Debug for Complex<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:?} + {:?} * u", self.c0, self.c1)
    }
}

impl<F: Field> DivUnsafe for Complex<F> {
    type Output = Self;
    /// Undefined behavior when denominator is not coprime to N.
    #[inline(always)]
    fn div_unsafe(self, other: &Self) -> Self::Output {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut res = self;
            let (c0, c1) = (res.c0, res.c1);
            let (d0, d1) = (&other.c0, &other.c1);
            let denom = (d0.square() + d1.square()).invert().unwrap();
            res.c0 = denom.clone() * (c0.clone() * d0 + c1.clone() * d1);
            res.c1 = denom * &(c1.clone() * d0 - c0.clone() * d1);
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

impl<F: Field> DivAssignUnsafe for Complex<F> {
    // TODO[jpw]: add where clause when Self: Field
    /// Implementation of DivAssign.
    #[inline(always)]
    fn div_assign_unsafe(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let (c0, c1) = (&self.c0, &self.c1);
            let (d0, d1) = (&other.c0, &other.c1);
            let denom = (d0.square() + d1.square()).invert().unwrap();
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
}
