use core::{
    fmt::{Debug, Formatter, Result},
    ops::{Add, AddAssign, Sub, SubAssign},
};

use openvm_algebra_guest::field::Field;

/// Sextic extension field of `F` with irreducible polynomial `X^6 - \xi`.
/// Elements are represented as `c0 + c1 * w + ... + c5 * w^5` where `w^6 = \xi`, where `\xi in F`.
///
/// Memory alignment follows alignment of `F`.
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct SexticExtField<F> {
    pub c: [F; 6],
}

impl<F> SexticExtField<F> {
    pub const fn new(c: [F; 6]) -> Self {
        Self { c }
    }
}

impl<'a, F: Field> AddAssign<&'a SexticExtField<F>> for SexticExtField<F> {
    #[inline(always)]
    fn add_assign(&mut self, other: &'a SexticExtField<F>) {
        for i in 0..6 {
            self.c[i] += &other.c[i];
        }
    }
}

impl<'a, F: Field> Add<&'a SexticExtField<F>> for &SexticExtField<F> {
    type Output = SexticExtField<F>;
    #[inline(always)]
    fn add(self, other: &'a SexticExtField<F>) -> Self::Output {
        let mut res = self.clone();
        res += other;
        res
    }
}

impl<'a, F: Field> SubAssign<&'a SexticExtField<F>> for SexticExtField<F> {
    #[inline(always)]
    fn sub_assign(&mut self, other: &'a SexticExtField<F>) {
        for i in 0..6 {
            self.c[i] -= &other.c[i];
        }
    }
}

impl<'a, F: Field> Sub<&'a SexticExtField<F>> for &SexticExtField<F> {
    type Output = SexticExtField<F>;
    #[inline(always)]
    fn sub(self, other: &'a SexticExtField<F>) -> Self::Output {
        let mut res = self.clone();
        res -= other;
        res
    }
}

pub(crate) fn sextic_tower_mul<F: Field>(
    lhs: &SexticExtField<F>,
    rhs: &SexticExtField<F>,
    xi: &F,
) -> SexticExtField<F>
where
    for<'a> &'a F: core::ops::Mul<&'a F, Output = F>,
{
    // The following multiplication is hand-derived with respect to the basis where degree 6
    // extension is composed of degree 3 extension followed by degree 2 extension.

    // c0 = cs0co0 + xi(cs1co2 + cs2co1 + cs3co5 + cs4co4 + cs5co3)
    // c1 = cs0co1 + cs1co0 + cs3co3 + xi(cs2co2 + cs4co5 + cs5co4)
    // c2 = cs0co2 + cs1co1 + cs2co0 + cs3co4 + cs4co3 + xi(cs5co5)
    // c3 = cs0co3 + cs3co0 + xi(cs1co5 + cs2co4 + cs4co2 + cs5co1)
    // c4 = cs0co4 + cs1co3 + cs3co1 + cs4co0 + xi(cs2co5 + cs5co2)
    // c5 = cs0co5 + cs1co4 + cs2co3 + cs3co2 + cs4co1 + cs5co0
    //   where cs*: lhs.c*, co*: rhs.c*

    let (s0, s1, s2, s3, s4, s5) = (
        &lhs.c[0], &lhs.c[2], &lhs.c[4], &lhs.c[1], &lhs.c[3], &lhs.c[5],
    );
    let (o0, o1, o2, o3, o4, o5) = (
        &rhs.c[0], &rhs.c[2], &rhs.c[4], &rhs.c[1], &rhs.c[3], &rhs.c[5],
    );

    let c0 = s0 * o0 + xi * &(s1 * o2 + s2 * o1 + s3 * o5 + s4 * o4 + s5 * o3);
    let c1 = s0 * o1 + s1 * o0 + s3 * o3 + xi * &(s2 * o2 + s4 * o5 + s5 * o4);
    let c2 = s0 * o2 + s1 * o1 + s2 * o0 + s3 * o4 + s4 * o3 + xi * &(s5 * o5);
    let c3 = s0 * o3 + s3 * o0 + xi * &(s1 * o5 + s2 * o4 + s4 * o2 + s5 * o1);
    let c4 = s0 * o4 + s1 * o3 + s3 * o1 + s4 * o0 + xi * &(s2 * o5 + s5 * o2);
    let c5 = s0 * o5 + s1 * o4 + s2 * o3 + s3 * o2 + s4 * o1 + s5 * o0;

    SexticExtField::new([c0, c3, c1, c4, c2, c5])
}

// Auto-derived implementations:

impl<F: Field> AddAssign for SexticExtField<F> {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        self.add_assign(&other);
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

impl<F: Field> SubAssign for SexticExtField<F> {
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign(&other);
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

impl<F: Field> Debug for SexticExtField<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{:?}, {:?}, {:?}, {:?}, {:?}, {:?}",
            self.c[0], self.c[1], self.c[2], self.c[3], self.c[4], self.c[5]
        )
    }
}
