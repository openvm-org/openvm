// use std::{borrow::Cow, fmt::Debug, iter, ops::Deref};

// use halo2curves_axiom::{ff::PrimeField, CurveAffine};
// use itertools::Itertools;
// use snark_verifier_sdk::snark_verifier::util::arithmetic::FieldOps;

use std::{
    fmt::{self, Debug},
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use halo2curves_axiom::{
    bn256::{Fq as Halo2Fp, Fr as Halo2Fr, G1Affine},
    ff::PrimeField,
    CurveAffine,
};
use openvm_ecc_guest::algebra::{ExpBytes, Field, IntMod};
use openvm_pairing_guest::bn254::{Bn254G1Affine as EcPoint, Fp, Scalar as Fr};
use snark_verifier::{
    loader::{LoadedEcPoint, LoadedScalar},
    util::arithmetic::FieldOps,
};

use super::loader::{AxVmLoader, LOADER};

#[derive(Clone)]
pub struct AxVmScalar<F: PrimeField, F2: Field>(pub F2, pub PhantomData<F>);

#[derive(Clone)]
pub struct AxVmEcPoint<CA: CurveAffine, C>(pub C, pub PhantomData<CA>);

impl<F: PrimeField, F2: Field> PartialEq for AxVmScalar<F, F2> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl LoadedScalar<Halo2Fp> for AxVmScalar<Halo2Fp, Fp> {
    type Loader = AxVmLoader;

    fn loader(&self) -> &Self::Loader {
        &LOADER
    }

    fn pow_var(&self, exp: &Self, _: usize) -> Self {
        AxVmScalar(self.0.exp_bytes(true, &exp.0.to_be_bytes()), PhantomData)
    }
}

impl LoadedScalar<Halo2Fr> for AxVmScalar<Halo2Fr, Fr> {
    type Loader = AxVmLoader;

    fn loader(&self) -> &Self::Loader {
        &LOADER
    }

    fn pow_var(&self, exp: &Self, _: usize) -> Self {
        AxVmScalar(self.0.exp_bytes(true, &exp.0.to_be_bytes()), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> Debug for AxVmScalar<F, F2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scalar")
            .field("value", &self.0.clone())
            .finish()
    }
}

impl<F: PrimeField, F2: Field> Add for AxVmScalar<F, F2> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        AxVmScalar(self.0.clone() + rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> Sub for AxVmScalar<F, F2> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        AxVmScalar(self.0.clone() - rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> Mul for AxVmScalar<F, F2> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        AxVmScalar(self.0.clone() * rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> Neg for AxVmScalar<F, F2> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        AxVmScalar(-self.0.clone(), PhantomData)
    }
}

impl<'b, F: PrimeField, F2: Field> Add<&'b Self> for AxVmScalar<F, F2> {
    type Output = Self;

    fn add(self, rhs: &'b Self) -> Self::Output {
        AxVmScalar(self.0.clone() + rhs.0.clone(), PhantomData)
    }
}

impl<'b, F: PrimeField, F2: Field> Sub<&'b Self> for AxVmScalar<F, F2> {
    type Output = Self;

    fn sub(self, rhs: &'b Self) -> Self::Output {
        AxVmScalar(self.0.clone() - rhs.0.clone(), PhantomData)
    }
}

impl<'b, F: PrimeField, F2: Field> Mul<&'b Self> for AxVmScalar<F, F2> {
    type Output = Self;

    fn mul(self, rhs: &'b Self) -> Self::Output {
        AxVmScalar(self.0.clone() * rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> AddAssign for AxVmScalar<F, F2> {
    fn add_assign(&mut self, rhs: Self) {
        *self = AxVmScalar(self.0.clone() + rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> SubAssign for AxVmScalar<F, F2> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = AxVmScalar(self.0.clone() - rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> MulAssign for AxVmScalar<F, F2> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = AxVmScalar(self.0.clone() * rhs.0.clone(), PhantomData)
    }
}

impl<'b, F: PrimeField, F2: Field> AddAssign<&'b Self> for AxVmScalar<F, F2> {
    fn add_assign(&mut self, rhs: &'b Self) {
        *self = AxVmScalar(self.0.clone() + rhs.0.clone(), PhantomData)
    }
}

impl<'b, F: PrimeField, F2: Field> SubAssign<&'b Self> for AxVmScalar<F, F2> {
    fn sub_assign(&mut self, rhs: &'b Self) {
        *self = AxVmScalar(self.0.clone() - rhs.0.clone(), PhantomData)
    }
}

impl<'b, F: PrimeField, F2: Field> MulAssign<&'b Self> for AxVmScalar<F, F2> {
    fn mul_assign(&mut self, rhs: &'b Self) {
        *self = AxVmScalar(self.0.clone() * rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> FieldOps for AxVmScalar<F, F2> {
    fn invert(&self) -> Option<Self> {
        self.0.inverse().map(|f| AxVmScalar(f, PhantomData))
    }
}

// impl<C: CurveAffine> AxVmEcPoint<C> {
//     fn value(&self) -> Self {
//         self.value.clone()
//     }
// }

impl<CA: CurveAffine> PartialEq for AxVmEcPoint<CA, EcPoint> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl LoadedEcPoint<G1Affine> for AxVmEcPoint<G1Affine, EcPoint> {
    type Loader = AxVmLoader;

    fn loader(&self) -> &Self::Loader {
        &LOADER
    }
}

impl<CA: CurveAffine> Debug for AxVmEcPoint<CA, EcPoint> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EcPoint")
            .field("value", &self.0.clone())
            .finish()
    }
}
