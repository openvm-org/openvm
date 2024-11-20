use core::{
    fmt::{Debug, Formatter, Result},
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

use axvm_algebra::{
    field::{Complex, ComplexConjugate, FieldExtension},
    DivAssignUnsafe, DivUnsafe, Field, IntMod,
};
#[cfg(target_os = "zkvm")]
use {
    super::shifted_funct7,
    axvm_platform::constants::{Custom1Funct3, PairingBaseFunct7, CUSTOM_1},
    axvm_platform::custom_insn_r,
};

/// Sextic extension field of `F` with irreducible polynomial `X^6 - \xi`.
/// Elements are represented as `c0 + c1 * w + ... + c5 * w^5` where `w^6 = \xi`, where `\xi in F`.
///
/// Memory alignment follows alignment of `F`.
#[derive(Clone, PartialEq, Eq)]
#[repr(C)]
pub struct SexticExtField<F> {
    pub c: [F; 6],
}

impl<F> SexticExtField<F> {
    pub const fn new(c: [F; 6]) -> Self {
        Self { c }
    }
}

// impl<F: Field> SexticExtField<F> {
//     pub fn div_assign_unsafe_impl(&mut self, other: &Self);

//     pub fn invert(&self) -> Self {
//         let [c0, c1, c2, c3, c4, c5] = self.c.clone();
//         let mut c0s = [c0.clone(), c2.clone(), c4.clone()];
//         let mut c1s = [c1.clone(), c3.clone(), c5.clone()];

//         fp6_square_assign(&mut c0s);
//         fp6_square_assign(&mut c1s);
//         fp6_mul_by_nonresidue(&mut c1s);
//         fp6_sub_assign(&mut c0s, &c1s);

//         fp6_invert(&mut c0s);
//         let mut t0 = c0s.clone();
//         let mut t1 = c0s.clone();
//         fp6_mul_assign(&mut t0, &[c0, c2, c4]);
//         fp6_mul_assign(&mut t1, &[c1, c3, c5]);
//         fp6_neg_assign(&mut t1);
//         SexticExtField::new([
//             t0[0].clone(),
//             t1[0].clone(),
//             t0[1].clone(),
//             t1[1].clone(),
//             t0[2].clone(),
//             t1[2].clone(),
//         ])
//     }
// }

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

/// SAFETY: `dst` must be a raw pointer to `&mut SexticExtField<F>`. It will be written to only at the very end .
///
/// When `target_os = "zkvm"`, this function calls an intrinsic instruction,
/// which is assumed to be supported by the VM.
#[cfg(target_os = "zkvm")]
#[inline(always)]
pub(crate) fn sextic_tower_mul_intrinsic<P: super::PairingIntrinsics>(
    dst: *mut u8,
    lhs: *const u8,
    rhs: *const u8,
) {
    custom_insn_r!(
        CUSTOM_1,
        Custom1Funct3::Pairing as usize,
        shifted_funct7::<P>(PairingBaseFunct7::Fp12Mul),
        dst,
        lhs,
        rhs
    );
}

#[cfg(not(target_os = "zkvm"))]
pub(crate) fn sextic_tower_mul_host<F: Field>(
    lhs: &SexticExtField<F>,
    rhs: &SexticExtField<F>,
    xi: &F,
) -> SexticExtField<F>
where
    for<'a> &'a F: core::ops::Mul<&'a F, Output = F>,
{
    // The following multiplication is hand-derived with respect to the basis where degree 6 extension
    // is composed of degree 3 extension followed by degree 2 extension.

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

// pub(crate) fn sextic_tower_div_unsafe_host<F: Field>(
//     lhs: &SexticExtField<F>,
//     rhs: &SexticExtField<F>,
//     xi: &F,
// ) -> SexticExtField<F>
// where
//     for<'a> &'a F: core::ops::Mul<&'a F, Output = F>,
// {
//     // Invert rhs
//     // let rhs_inv = sextic_tower_invert(rhs);
//     let rhs_inv = rhs.invert();

//     // Multiply lhs by the inverse of rhs
//     sextic_tower_mul_host(lhs, &rhs_inv, xi)
// }

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

// impl<F: Field> DivAssignUnsafe for SexticExtField<F> {
//     #[inline(always)]
//     fn div_assign_unsafe(&mut self, other: Self) {
//         self.div_assign_unsafe_impl(&other);
//     }
// }

// impl<F: Field> DivUnsafe for SexticExtField<F> {
//     type Output = Self;
//     #[inline(always)]
//     fn div_unsafe(mut self, other: Self) -> Self::Output {
//         self -= other;
//         self
//     }
// }

// impl<'a, F: Field> DivUnsafe<&'a SexticExtField<F>> for SexticExtField<F> {
//     type Output = Self;
//     #[inline(always)]
//     fn div_unsafe(mut self, other: &'a SexticExtField<F>) -> Self::Output {
//         self /= other;
//         self
//     }
// }

// impl<F: Field> Neg for SexticExtField<F> {
//     type Output = SexticExtField<F>;
//     #[inline(always)]
//     fn neg(self) -> Self::Output {
//         Self::ZERO - &self
//     }
// }

impl<F: Field> Debug for SexticExtField<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{:?}, {:?}, {:?}, {:?}, {:?}, {:?}",
            self.c[0], self.c[1], self.c[2], self.c[3], self.c[4], self.c[5]
        )
    }
}

// pub fn sextic_tower_invert<F: Field>(x: &SexticExtField<F>) -> SexticExtField<F> {
//     let mut c0s = [F::ZERO; 3];
//     let mut c1s = [F::ZERO; 3];
//     c0s.clone_from_slice(&x.c[0..3]);
//     c1s.clone_from_slice(&x.c[3..6]);

//     fp6_square_assign(&mut c0s);
//     fp6_square_assign(&mut c1s);

//     fp6_mul_by_nonresidue(&mut c1s);

//     todo!("finish")
// }

pub fn fp12_mul_by_nonresidue<F: Field>(x: &mut SexticExtField<F>) {}
