// use std::{borrow::Cow, fmt::Debug, iter, ops::Deref};

// use halo2curves_axiom::{ff::PrimeField, CurveAffine};
// use itertools::Itertools;
// use snark_verifier_sdk::snark_verifier::util::arithmetic::FieldOps;

use std::{
    fmt::{self, Debug},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use halo2curves_axiom::{ff::PrimeField, CurveAffine};
use snark_verifier_sdk::snark_verifier::{
    loader::{LoadedEcPoint, LoadedScalar},
    util::arithmetic::{fe_to_big, FieldOps},
};

use super::loader::{AxVmCurve, AxVmLoader, LOADER};

#[derive(Clone)]
pub struct AxVmScalar<F: PrimeField>(pub F);

#[derive(Clone)]
pub struct AxVmEcPoint<C: CurveAffine>(pub C);

impl<F: PrimeField> PartialEq for AxVmScalar<F> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<F: PrimeField> LoadedScalar<F> for AxVmScalar<F> {
    type Loader = AxVmLoader;

    fn loader(&self) -> &Self::Loader {
        &LOADER
    }

    fn pow_var(&self, exp: &Self, _: usize) -> Self {
        let exp = fe_to_big(exp.0).to_u64_digits();
        AxVmScalar(self.0.pow_vartime(exp))
    }
}

impl<F: PrimeField> Debug for AxVmScalar<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scalar").field("value", &self.0).finish()
    }
}

impl<F: PrimeField> FieldOps for AxVmScalar<F> {
    fn invert(&self) -> Option<Self> {
        Option::<F>::from(self.0.invert()).map(|f| AxVmScalar(f))
    }
}

impl<F: PrimeField> Add for AxVmScalar<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        AxVmScalar(self.0 + rhs.0)
    }
}

impl<F: PrimeField> Sub for AxVmScalar<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        AxVmScalar(self.0 - rhs.0)
    }
}

impl<F: PrimeField> Mul for AxVmScalar<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        AxVmScalar(self.0 * rhs.0)
    }
}

impl<F: PrimeField> Neg for AxVmScalar<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        AxVmScalar(-self.0)
    }
}

impl<'b, F: PrimeField> Add<&'b Self> for AxVmScalar<F> {
    type Output = Self;

    fn add(self, rhs: &'b Self) -> Self::Output {
        AxVmScalar(self.0 + rhs.0)
    }
}

impl<'b, F: PrimeField> Sub<&'b Self> for AxVmScalar<F> {
    type Output = Self;

    fn sub(self, rhs: &'b Self) -> Self::Output {
        AxVmScalar(self.0 - rhs.0)
    }
}

impl<'b, F: PrimeField> Mul<&'b Self> for AxVmScalar<F> {
    type Output = Self;

    fn mul(self, rhs: &'b Self) -> Self::Output {
        AxVmScalar(self.0 * rhs.0)
    }
}

impl<F: PrimeField> AddAssign for AxVmScalar<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = AxVmScalar(self.0 + rhs.0)
    }
}

impl<F: PrimeField> SubAssign for AxVmScalar<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = AxVmScalar(self.0 - rhs.0)
    }
}

impl<F: PrimeField> MulAssign for AxVmScalar<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = AxVmScalar(self.0 * rhs.0)
    }
}

impl<'b, F: PrimeField> AddAssign<&'b Self> for AxVmScalar<F> {
    fn add_assign(&mut self, rhs: &'b Self) {
        *self = AxVmScalar(self.0 + rhs.0)
    }
}

impl<'b, F: PrimeField> SubAssign<&'b Self> for AxVmScalar<F> {
    fn sub_assign(&mut self, rhs: &'b Self) {
        *self = AxVmScalar(self.0 - rhs.0)
    }
}

impl<'b, F: PrimeField> MulAssign<&'b Self> for AxVmScalar<F> {
    fn mul_assign(&mut self, rhs: &'b Self) {
        *self = AxVmScalar(self.0 * rhs.0)
    }
}

// impl<C: CurveAffine> AxVmEcPoint<C> {
//     fn value(&self) -> Self {
//         self.value.clone()
//     }
// }

impl<C: CurveAffine> PartialEq for AxVmEcPoint<C> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<C: AxVmCurve> LoadedEcPoint<C> for AxVmEcPoint<C> {
    type Loader = AxVmLoader;

    fn loader(&self) -> &Self::Loader {
        &LOADER
    }
}

impl<C: CurveAffine> Debug for AxVmEcPoint<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EcPoint").field("value", &self.0).finish()
    }
}
