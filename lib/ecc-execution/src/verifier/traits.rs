use std::{borrow::Cow, fmt::Debug, iter, ops::Deref};

use halo2curves_axiom::{ff::PrimeField, CurveAffine};
use itertools::Itertools;
use snark_verifier_sdk::snark_verifier::util::arithmetic::FieldOps;

/// Loaded elliptic curve point.
pub trait LoadedEcPoint<C: CurveAffine>: Clone + Debug + PartialEq {
    /// [`Loader`].
    type Loader: Loader<C, LoadedEcPoint = Self>;

    /// Returns [`Loader`].
    fn loader(&self) -> &Self::Loader;
}

/// Loaded field element.
pub trait LoadedScalar<F: PrimeField>: Clone + Debug + PartialEq + FieldOps {
    /// [`Loader`].
    type Loader: ScalarLoader<F, LoadedScalar = Self>;

    /// Returns [`Loader`].
    fn loader(&self) -> &Self::Loader;

    /// Returns square.
    fn square(&self) -> Self {
        self.clone() * self
    }

    /// Returns inverse if any.
    fn invert(&self) -> Option<Self> {
        FieldOps::invert(self)
    }

    /// Returns power to exponent.
    fn pow_const(&self, mut exp: u64) -> Self {
        assert!(exp > 0);

        let mut base = self.clone();

        while exp & 1 == 0 {
            base = base.square();
            exp >>= 1;
        }

        let mut acc = base.clone();
        while exp > 1 {
            exp >>= 1;
            base = base.square();
            if exp & 1 == 1 {
                acc *= &base;
            }
        }
        acc
    }

    /// Returns power to exponent, where exponent is also [`LoadedScalar`].
    /// If `Loader` is for Halo2, then `exp` must have at most `exp_max_bits` bits (otherwise constraints will fail).
    ///
    /// Currently **unimplemented** for EvmLoader
    fn pow_var(&self, exp: &Self, exp_max_bits: usize) -> Self;

    /// Returns powers up to exponent `n-1`.
    fn powers(&self, n: usize) -> Vec<Self> {
        iter::once(self.loader().load_one())
            .chain(
                iter::successors(Some(self.clone()), |power| Some(power.clone() * self))
                    .take(n - 1),
            )
            .collect_vec()
    }
}

/// Elliptic curve point loader.
pub trait EcPointLoader<C: CurveAffine> {
    /// [`LoadedEcPoint`].
    type LoadedEcPoint: LoadedEcPoint<C, Loader = Self>;

    /// Load a constant elliptic curve point.
    fn ec_point_load_const(&self, value: &C) -> Self::LoadedEcPoint;

    /// Load `identity` as constant.
    fn ec_point_load_zero(&self) -> Self::LoadedEcPoint {
        self.ec_point_load_const(&C::identity())
    }

    /// Load `generator` as constant.
    fn ec_point_load_one(&self) -> Self::LoadedEcPoint {
        self.ec_point_load_const(&C::generator())
    }

    /// Assert lhs and rhs elliptic curve points are equal.
    fn ec_point_assert_eq(
        &self,
        annotation: &str,
        lhs: &Self::LoadedEcPoint,
        rhs: &Self::LoadedEcPoint,
    );

    /// Perform multi-scalar multiplication.
    fn multi_scalar_multiplication(
        pairs: &[(&Self::LoadedScalar, &Self::LoadedEcPoint)],
    ) -> Self::LoadedEcPoint
    where
        Self: ScalarLoader<C::ScalarExt>;
}

/// Field element loader.
pub trait ScalarLoader<F: PrimeField> {
    /// [`LoadedScalar`].
    type LoadedScalar: LoadedScalar<F, Loader = Self>;

    /// Load a constant field element.
    fn load_const(&self, value: &F) -> Self::LoadedScalar;

    /// Load `zero` as constant.
    fn load_zero(&self) -> Self::LoadedScalar {
        self.load_const(&F::ZERO)
    }

    /// Load `one` as constant.
    fn load_one(&self) -> Self::LoadedScalar {
        self.load_const(&F::ONE)
    }

    /// Assert lhs and rhs field elements are equal.
    fn assert_eq(&self, annotation: &str, lhs: &Self::LoadedScalar, rhs: &Self::LoadedScalar);

    /// Sum field elements with coefficients and constant.
    fn sum_with_coeff_and_const(
        &self,
        values: &[(F, &Self::LoadedScalar)],
        constant: F,
    ) -> Self::LoadedScalar {
        if values.is_empty() {
            return self.load_const(&constant);
        }

        let loader = values.first().unwrap().1.loader();
        iter::empty()
            .chain(if constant == F::ZERO {
                None
            } else {
                Some(Cow::Owned(loader.load_const(&constant)))
            })
            .chain(values.iter().map(|&(coeff, value)| {
                if coeff == F::ONE {
                    Cow::Borrowed(value)
                } else {
                    Cow::Owned(loader.load_const(&coeff) * value)
                }
            }))
            .reduce(|acc, term| Cow::Owned(acc.into_owned() + term.deref()))
            .unwrap()
            .into_owned()
    }

    /// Sum product of field elements with coefficients and constant.
    fn sum_products_with_coeff_and_const(
        &self,
        values: &[(F, &Self::LoadedScalar, &Self::LoadedScalar)],
        constant: F,
    ) -> Self::LoadedScalar {
        if values.is_empty() {
            return self.load_const(&constant);
        }

        let loader = values.first().unwrap().1.loader();
        iter::empty()
            .chain(if constant == F::ZERO {
                None
            } else {
                Some(loader.load_const(&constant))
            })
            .chain(values.iter().map(|&(coeff, lhs, rhs)| {
                if coeff == F::ONE {
                    lhs.clone() * rhs
                } else {
                    loader.load_const(&coeff) * lhs * rhs
                }
            }))
            .reduce(|acc, term| acc + term)
            .unwrap()
    }

    /// Sum field elements with coefficients.
    fn sum_with_coeff(&self, values: &[(F, &Self::LoadedScalar)]) -> Self::LoadedScalar {
        self.sum_with_coeff_and_const(values, F::ZERO)
    }

    /// Sum field elements and constant.
    fn sum_with_const(&self, values: &[&Self::LoadedScalar], constant: F) -> Self::LoadedScalar {
        self.sum_with_coeff_and_const(
            &values.iter().map(|&value| (F::ONE, value)).collect_vec(),
            constant,
        )
    }

    /// Sum field elements.
    fn sum(&self, values: &[&Self::LoadedScalar]) -> Self::LoadedScalar {
        self.sum_with_const(values, F::ZERO)
    }

    /// Sum product of field elements with coefficients.
    fn sum_products_with_coeff(
        &self,
        values: &[(F, &Self::LoadedScalar, &Self::LoadedScalar)],
    ) -> Self::LoadedScalar {
        self.sum_products_with_coeff_and_const(values, F::ZERO)
    }

    /// Sum product of field elements and constant.
    fn sum_products_with_const(
        &self,
        values: &[(&Self::LoadedScalar, &Self::LoadedScalar)],
        constant: F,
    ) -> Self::LoadedScalar {
        self.sum_products_with_coeff_and_const(
            &values
                .iter()
                .map(|&(lhs, rhs)| (F::ONE, lhs, rhs))
                .collect_vec(),
            constant,
        )
    }

    /// Sum product of field elements.
    fn sum_products(
        &self,
        values: &[(&Self::LoadedScalar, &Self::LoadedScalar)],
    ) -> Self::LoadedScalar {
        self.sum_products_with_const(values, F::ZERO)
    }

    /// Product of field elements.
    fn product(&self, values: &[&Self::LoadedScalar]) -> Self::LoadedScalar {
        values
            .iter()
            .fold(self.load_one(), |acc, value| acc * *value)
    }

    /// Batch invert field elements.
    fn batch_invert<'a>(values: impl IntoIterator<Item = &'a mut Self::LoadedScalar>)
    where
        Self::LoadedScalar: 'a,
    {
        values
            .into_iter()
            .for_each(|value| *value = LoadedScalar::invert(value).unwrap_or_else(|| value.clone()))
    }
}

/// [`EcPointLoader`] and [`ScalarLoader`] with some helper methods.
pub trait Loader<C: CurveAffine>:
    EcPointLoader<C> + ScalarLoader<C::ScalarExt> + Clone + Debug
{
    /// Start cost metering with an `identifier`.
    fn start_cost_metering(&self, _identifier: &str) {}

    /// End latest started cost metering.
    fn end_cost_metering(&self) {}
}
