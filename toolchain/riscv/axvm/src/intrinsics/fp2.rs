use core::{
    fmt::{Debug, Formatter, Result},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::BigUint;

use super::IntMod;

/// Trait definition for AXVM Fp2s, which take the form c0 + c1 * u where field
/// Fp2 = Fp[u]/(u^2 + 1).
pub trait Fp2<F: IntMod>:
    Sized
    + Eq
    + Clone
    + Debug
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Sum
    + Product
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> DivAssign<&'a Self>
{
    /// Index of IntMod::MODULUS.
    const MOD_IDX: usize = F::MOD_IDX;

    /// Modulus as an F::Repr.
    const MODULUS: F::Repr = F::MODULUS;

    /// The zero element (i.e. the additive identity).
    const ZERO: Self;

    /// The one element (i.e. the multiplicative identity).
    const ONE: Self;

    /// TODO
    fn new(c0: F, c1: F) -> Self;

    /// TODO
    fn from_fp((c0, c1): (F, F)) -> Self {
        Self::new(c0, c1)
    }

    /// Creates a new Fp2 from 2 instances of Repr.
    fn from_repr((c0_repr, c1_repr): (F::Repr, F::Repr)) -> Self {
        Self::new(F::from_repr(c0_repr), F::from_repr(c1_repr))
    }

    /// Creates a new Fp2 from two arrays of bytes.
    fn from_le_bytes((c0_bytes, c1_bytes): (&[u8], &[u8])) -> Self {
        Self::new(F::from_le_bytes(c0_bytes), F::from_le_bytes(c1_bytes))
    }

    /// Creates a new Fp2 from two u8s.
    fn from_u8((c0_val, c1_val): (u8, u8)) -> Self {
        Self::new(F::from_u8(c0_val), F::from_u8(c1_val))
    }

    /// Creates a new Fp2 from two u32s.
    fn from_u32((c0_val, c1_val): (u32, u32)) -> Self {
        Self::new(F::from_u32(c0_val), F::from_u32(c1_val))
    }

    /// Creates a new Fp2 from two u64s.
    fn from_u64((c0_val, c1_val): (u64, u64)) -> Self {
        Self::new(F::from_u64(c0_val), F::from_u64(c1_val))
    }

    /// Value of c0 and c1 as Fps.
    fn as_fp(&self) -> (&F, &F);

    /// Value of c0 and c1 as arrays of bytes.
    fn as_le_bytes(&self) -> (&[u8], &[u8]);

    /// Returns MODULUS (i.e. p) as a BigUint.
    #[cfg(not(target_os = "zkvm"))]
    fn modulus_biguint() -> BigUint {
        F::modulus_biguint()
    }

    /// Creates a new Fp2 from two BigUints.
    #[cfg(not(target_os = "zkvm"))]
    fn from_biguint((c0, c1): (BigUint, BigUint)) -> Self {
        Self::new(F::from_biguint(c0), F::from_biguint(c1))
    }

    /// Value of c0 and c1 as BigUints.
    #[cfg(not(target_os = "zkvm"))]
    fn as_biguint(&self) -> (BigUint, BigUint) {
        let (c0, c1) = self.as_fp();
        (c0.as_biguint(), c1.as_biguint())
    }
}

/// TODO
#[derive(Clone, PartialEq, Eq)]
#[repr(C)]
pub struct Complex<F> {
    c0: F,
    c1: F,
}

impl<F: IntMod> Complex<F> {
    const fn new(c0: F, c1: F) -> Self {
        Self { c0, c1 }
    }
}

impl<F: IntMod> Complex<F> {
    #[inline(always)]
    fn add_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            let (d0, d1) = other.as_fp();
            self.c0 += d0;
            self.c1 += d1;
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
            let (d0, d1) = other.as_fp();
            self.c0 -= d0;
            self.c1 -= d1;
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
            let (c0, c1) = self.as_fp();
            let (d0, d1) = other.as_fp();
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
            let (c0, c1) = self.as_fp();
            let (d0, d1) = other.as_fp();
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

impl<F: IntMod> Fp2<F> for Complex<F> {
    const ZERO: Self = Self::new(F::ZERO, F::ZERO);

    const ONE: Self = Self::new(F::ONE, F::ZERO);

    fn new(c0: F, c1: F) -> Self {
        Self::new(c0, c1)
    }

    /// Value of c0 and c1 as Fps.
    fn as_fp(&self) -> (&F, &F) {
        (&self.c0, &self.c1)
    }

    /// Value of c0 and c1 as arrays of bytes.
    fn as_le_bytes(&self) -> (&[u8], &[u8]) {
        (self.c0.as_le_bytes(), self.c1.as_le_bytes())
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
        write!(f, "{:?}", self.as_le_bytes())
    }
}
