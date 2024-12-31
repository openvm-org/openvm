use core::{
    fmt::{self, Debug},
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use halo2_proofs::halo2curves::{
    bn256::{Fq as Halo2Fp, Fr as Halo2Fr, G1Affine},
    ff::PrimeField,
    CurveAffine,
};
use openvm_ecc_guest::algebra::{ExpBytes, Field, IntMod};
use openvm_pairing_guest::bn254::{Bn254G1Affine as EcPoint, Fp, Scalar as Fr};
use serde::{Deserialize, Serialize};
use snark_verifier_sdk::snark_verifier::{
    halo2_base::halo2_proofs,
    loader::{LoadedEcPoint, LoadedScalar},
    util::arithmetic::FieldOps,
};

use super::loader::{OpenVmLoader, LOADER};

#[derive(Clone, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(bound(serialize = "F2: Serialize", deserialize = "F2: Deserialize<'de>"))]
pub struct OpenVmScalar<F, F2>(pub F2, PhantomData<F>);

impl<F, F2> OpenVmScalar<F, F2> {
    pub fn new(value: F2) -> Self {
        Self(value, PhantomData)
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(bound(serialize = "C: Serialize", deserialize = "C: Deserialize<'de>"))]
pub struct OpenVmEcPoint<CA, C>(pub C, PhantomData<CA>);

impl<CA, C> OpenVmEcPoint<CA, C> {
    pub fn new(point: C) -> Self {
        Self(point, PhantomData)
    }
}

impl<F: PrimeField, F2: Field> PartialEq for OpenVmScalar<F, F2> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl LoadedScalar<Halo2Fp> for OpenVmScalar<Halo2Fp, Fp> {
    type Loader = OpenVmLoader;

    fn loader(&self) -> &Self::Loader {
        &LOADER
    }

    fn pow_var(&self, exp: &Self, _: usize) -> Self {
        OpenVmScalar(self.0.exp_bytes(true, &exp.0.to_be_bytes()), PhantomData)
    }
}

impl LoadedScalar<Halo2Fr> for OpenVmScalar<Halo2Fr, Fr> {
    type Loader = OpenVmLoader;

    fn loader(&self) -> &Self::Loader {
        &LOADER
    }

    fn pow_var(&self, exp: &Self, _: usize) -> Self {
        OpenVmScalar(self.0.exp_bytes(true, &exp.0.to_be_bytes()), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> Debug for OpenVmScalar<F, F2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scalar")
            .field("value", &self.0.clone())
            .finish()
    }
}

impl<F: PrimeField, F2: Field> Add for OpenVmScalar<F, F2> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        OpenVmScalar(self.0.clone() + rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> Sub for OpenVmScalar<F, F2> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        OpenVmScalar(self.0.clone() - rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> Mul for OpenVmScalar<F, F2> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        OpenVmScalar(self.0.clone() * rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> Neg for OpenVmScalar<F, F2> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        OpenVmScalar(-self.0.clone(), PhantomData)
    }
}

impl<'b, F: PrimeField, F2: Field> Add<&'b Self> for OpenVmScalar<F, F2> {
    type Output = Self;

    fn add(self, rhs: &'b Self) -> Self::Output {
        OpenVmScalar(self.0.clone() + rhs.0.clone(), PhantomData)
    }
}

impl<'b, F: PrimeField, F2: Field> Sub<&'b Self> for OpenVmScalar<F, F2> {
    type Output = Self;

    fn sub(self, rhs: &'b Self) -> Self::Output {
        OpenVmScalar(self.0.clone() - rhs.0.clone(), PhantomData)
    }
}

impl<'b, F: PrimeField, F2: Field> Mul<&'b Self> for OpenVmScalar<F, F2> {
    type Output = Self;

    fn mul(self, rhs: &'b Self) -> Self::Output {
        OpenVmScalar(self.0.clone() * rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> AddAssign for OpenVmScalar<F, F2> {
    fn add_assign(&mut self, rhs: Self) {
        *self = OpenVmScalar(self.0.clone() + rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> SubAssign for OpenVmScalar<F, F2> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = OpenVmScalar(self.0.clone() - rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> MulAssign for OpenVmScalar<F, F2> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = OpenVmScalar(self.0.clone() * rhs.0.clone(), PhantomData)
    }
}

impl<'b, F: PrimeField, F2: Field> AddAssign<&'b Self> for OpenVmScalar<F, F2> {
    fn add_assign(&mut self, rhs: &'b Self) {
        *self = OpenVmScalar(self.0.clone() + rhs.0.clone(), PhantomData)
    }
}

impl<'b, F: PrimeField, F2: Field> SubAssign<&'b Self> for OpenVmScalar<F, F2> {
    fn sub_assign(&mut self, rhs: &'b Self) {
        *self = OpenVmScalar(self.0.clone() - rhs.0.clone(), PhantomData)
    }
}

impl<'b, F: PrimeField, F2: Field> MulAssign<&'b Self> for OpenVmScalar<F, F2> {
    fn mul_assign(&mut self, rhs: &'b Self) {
        *self = OpenVmScalar(self.0.clone() * rhs.0.clone(), PhantomData)
    }
}

impl<F: PrimeField, F2: Field> FieldOps for OpenVmScalar<F, F2> {
    fn invert(&self) -> Option<Self> {
        (self.0 != F2::ZERO).then(|| OpenVmScalar::new(self.0.invert()))
    }
}

impl<CA: CurveAffine> PartialEq for OpenVmEcPoint<CA, EcPoint> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl LoadedEcPoint<G1Affine> for OpenVmEcPoint<G1Affine, EcPoint> {
    type Loader = OpenVmLoader;

    fn loader(&self) -> &Self::Loader {
        &LOADER
    }
}

impl<CA: CurveAffine> Debug for OpenVmEcPoint<CA, EcPoint> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EcPoint")
            .field("value", &self.0.clone())
            .finish()
    }
}
