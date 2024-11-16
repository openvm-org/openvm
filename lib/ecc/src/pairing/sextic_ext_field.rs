use core::{
    fmt::{Debug, Formatter, Result},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::{Field, FieldExtension, Xi};

pub trait Fp12Mul {
    type Fp: Field;
    type Fp2: FieldExtension<BaseField = Self::Fp> + Xi;

    fn fp12_mul(&mut self, other: &Self, xi: &Self::Fp2);

    fn fp12_mul_refs(&self, other: &Self, xi: &Self::Fp2) -> Self;
}

/// Sextic extension field of `F` with irreducible polynomial `X^6 + \xi`.
/// Elements are represented as `c0 + c1 * w` where `w^6 = \xi`, where \xi depends on the twist of the curve.
///
/// Memory alignment follows alignment of `F`.
/// Memory layout is concatenation of `c0` and `c1`.
#[derive(Clone, PartialEq, Eq)]
#[repr(C)]
pub struct SexticExtField<F> {
    pub c: [F; 6],
}

impl<F: Field> SexticExtField<F> {
    pub const fn new(c: [F; 6]) -> Self {
        Self { c }
    }
}

impl<F: Field> SexticExtField<F> {
    /// Implementation of AddAssign
    fn add_assign_impl(&mut self, _other: &Self) {
        unimplemented!()
    }

    /// Implementation of SubAssign.
    #[inline(always)]
    fn sub_assign_impl(&mut self, _other: &Self) {
        unimplemented!()
    }

    /// Implementation of MulAssign.
    #[inline(always)]
    fn mul_assign_impl(&mut self, _other: &Self) {
        unimplemented!()
    }

    /// Implementation of Add that doesn't cause zkvm to use an additional store.
    fn add_refs_impl(&self, _other: &Self) -> Self {
        unimplemented!()
    }

    /// Implementation of Sub that doesn't cause zkvm to use an additional store.
    #[inline(always)]
    fn sub_refs_impl(&self, _other: &Self) -> Self {
        unimplemented!()
    }

    /// Implementation of Mul that doesn't cause zkvm to use an additional store.
    #[inline(always)]
    fn mul_refs_impl(&self, _other: &Self) -> Self {
        unimplemented!()
    }
}

impl<F: Field> Field for SexticExtField<F> {
    type SelfRef<'a>
        = &'a Self
    where
        F: 'a;

    fn zero() -> Self {
        Self::new([
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
        ])
    }

    fn one() -> Self {
        Self::new([
            F::one(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
        ])
    }

    fn invert(&self) -> Option<Self> {
        todo!()
    }
}

impl<'a, F: Field> AddAssign<&'a SexticExtField<F>> for SexticExtField<F> {
    #[inline(always)]
    fn add_assign(&mut self, other: &'a SexticExtField<F>) {
        self.add_assign_impl(other);
    }
}

impl<F: Field> AddAssign for SexticExtField<F> {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        self.add_assign_impl(&other);
    }
}

impl<F: Field> Add for SexticExtField<F> {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl<'a, F: Field> Add<&'a SexticExtField<F>> for SexticExtField<F> {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: &'a SexticExtField<F>) -> Self::Output {
        self += other;
        self
    }
}

impl<'a, F: Field> Add<&'a SexticExtField<F>> for &SexticExtField<F> {
    type Output = SexticExtField<F>;
    #[inline(always)]
    fn add(self, other: &'a SexticExtField<F>) -> Self::Output {
        self.add_refs_impl(other)
    }
}

impl<'a, F: Field> SubAssign<&'a SexticExtField<F>> for SexticExtField<F> {
    #[inline(always)]
    fn sub_assign(&mut self, other: &'a SexticExtField<F>) {
        self.sub_assign_impl(other);
    }
}

impl<F: Field> SubAssign for SexticExtField<F> {
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign_impl(&other);
    }
}

impl<F: Field> Sub for SexticExtField<F> {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, other: Self) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a, F: Field> Sub<&'a SexticExtField<F>> for SexticExtField<F> {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, other: &'a SexticExtField<F>) -> Self::Output {
        self -= other;
        self
    }
}

impl<'a, F: Field> Sub<&'a SexticExtField<F>> for &SexticExtField<F> {
    type Output = SexticExtField<F>;
    #[inline(always)]
    fn sub(self, other: &'a SexticExtField<F>) -> Self::Output {
        self.sub_refs_impl(other)
    }
}

impl<'a, F: Field> MulAssign<&'a SexticExtField<F>> for SexticExtField<F> {
    #[inline(always)]
    fn mul_assign(&mut self, other: &'a SexticExtField<F>) {
        self.mul_assign_impl(other);
    }
}

impl<F: Field> MulAssign for SexticExtField<F> {
    #[inline(always)]
    fn mul_assign(&mut self, other: Self) {
        self.mul_assign_impl(&other);
    }
}

impl<F: Field> Mul for SexticExtField<F> {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: Self) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a, F: Field> Mul<&'a SexticExtField<F>> for SexticExtField<F> {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, other: &'a SexticExtField<F>) -> Self::Output {
        self *= other;
        self
    }
}

impl<'a, F: Field> Mul<&'a SexticExtField<F>> for &SexticExtField<F> {
    type Output = SexticExtField<F>;
    #[inline(always)]
    fn mul(self, other: &'a SexticExtField<F>) -> Self::Output {
        self.mul_refs_impl(other)
    }
}

impl<F: Field> Neg for SexticExtField<F> {
    type Output = SexticExtField<F>;
    fn neg(self) -> Self::Output {
        Self::zero() - &self
    }
}

impl<F: Field> Debug for SexticExtField<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{:?}, {:?}, {:?}, {:?}, {:?}, {:?}",
            self.c[0], self.c[1], self.c[2], self.c[3], self.c[4], self.c[5]
        )
    }
}
