use core::{
    fmt::{Debug, Formatter, Result},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
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
    const fn new(c0: F, c1: F) -> Self {
        Self { c0, c1 }
    }
}

impl<F: Field> Complex<F> {
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

impl<F: Field + DivUnsafe + DivAssignUnsafe> DivUnsafe for Complex<F> {
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

impl<F: Field + DivUnsafe + DivAssignUnsafe> Div for Complex<F> {
    type Output = Self;
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div(mut self, other: Self) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a, F: Field + DivUnsafe + DivAssignUnsafe> Div<&'a Complex<F>> for Complex<F> {
    type Output = Self;
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div(mut self, other: &'a Complex<F>) -> Self::Output {
        self /= other;
        self
    }
}

impl<'a, F: Field + DivUnsafe + DivAssignUnsafe> Div<&'a Complex<F>> for &Complex<F> {
    type Output = Complex<F>;
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div(self, other: &'a Complex<F>) -> Self::Output {
        self.div_refs_impl(other)
    }
}

impl<F: Field + DivUnsafe + DivAssignUnsafe> DivAssignUnsafe for Complex<F> {
    // TODO[jpw]: add where clause when Self: Field
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
}

impl<'a, F: Field + DivUnsafe + DivAssignUnsafe> DivAssign<&'a Complex<F>> for Complex<F> {
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div_assign(&mut self, other: &'a Complex<F>) {
        self.div_assign_impl(other);
    }
}

impl<F: Field + DivUnsafe + DivAssignUnsafe> DivAssign for Complex<F> {
    /// Undefined behaviour when denominator is not coprime to N
    #[inline(always)]
    fn div_assign(&mut self, other: Self) {
        self.div_assign_impl(&other);
    }
}
