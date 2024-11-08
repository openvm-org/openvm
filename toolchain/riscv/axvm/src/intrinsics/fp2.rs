use core::{
    fmt::{Debug, Formatter, Result},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::IntMod;

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
    const fn new(c0: F, c1: F) -> Self {
        Self { c0, c1 }
    }
}

impl<F: IntMod> Complex<F> {
    const ZERO: Self = Self::new(F::ZERO, F::ZERO);
    const ONE: Self = Self::new(F::ONE, F::ZERO);

    #[inline(always)]
    fn add_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            self.c0 += &other.c0;
            self.c1 += &other.c1;
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
            self.c0 -= &other.c0;
            self.c1 -= &other.c1;
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
            let (c0, c1) = (&self.c0, &self.c1);
            let (d0, d1) = (&other.c0, &other.c1);
            *self = Self::new(
                c0.clone() * d0 - c1.clone() * d1,
                c0.clone() * d1 + c1.clone() * d0,
            );
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
            let (c0, c1) = (&self.c0, &self.c1);
            let (d0, d1) = (&other.c0, &other.c1);
            let denom = F::ONE / (d0.square() + d1.square());
            *self = Self::new(
                denom.clone() * (c0.clone() * d0 + c1.clone() * d1),
                denom * &(c1.clone() * d0 - c0.clone() * d1),
            );
        }
        #[cfg(target_os = "zkvm")]
        {
            todo!()
        }
    }

    /// Implementation of Add that doesn't cause zkvm to use an additional store.
    #[inline(always)]
    fn add_refs_impl(&self, other: &Self) -> Self {
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
    fn sub_refs_impl(&self, other: &Self) -> Self {
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
    fn mul_refs_impl(&self, other: &Self) -> Self {
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
    fn div_refs_impl(&self, other: &Self) -> Self {
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

impl<'a, F: IntMod> Mul<&'a Complex<F>> for &Complex<F> {
    type Output = Complex<F>;
    #[inline(always)]
    fn mul(self, other: &'a Complex<F>) -> Self::Output {
        self.mul_refs_impl(other)
    }
}

impl<'a, F: IntMod> DivAssign<&'a Complex<F>> for Complex<F> {
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div_assign(&mut self, other: &'a Complex<F>) {
        self.div_assign_impl(other);
    }
}

impl<F: IntMod> DivAssign for Complex<F> {
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div_assign(&mut self, other: Self) {
        self.div_assign_impl(&other);
    }
}

impl<F: IntMod> Div for Complex<F> {
    type Output = Self;
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div(mut self, other: Self) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a, F: IntMod> Div<&'a Complex<F>> for Complex<F> {
    type Output = Self;
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div(mut self, other: &'a Complex<F>) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a, F: IntMod> Div<&'a Complex<F>> for &Complex<F> {
    type Output = Complex<F>;
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div(self, other: &'a Complex<F>) -> Self::Output {
        self.div_refs_impl(other)
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

impl<F: IntMod> Debug for Complex<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:?} + {:?} * u", self.c0, self.c1)
    }
}
