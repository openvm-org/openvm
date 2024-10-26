//! `Loader` implementation in native rust.

use std::fmt::Debug;

use ax_ecc_lib::ec_msm::msm_axvm_C;
use halo2curves_axiom::{ff::PrimeField, CurveAffine};
use lazy_static::lazy_static;
use snark_verifier_sdk::snark_verifier::{util::arithmetic::fe_to_big, Error};

use super::traits::{EcPointLoader, LoadedEcPoint, LoadedScalar, Loader, ScalarLoader};

lazy_static! {
    /// NativeLoader instance for [`LoadedEcPoint::loader`] and
    /// [`LoadedScalar::loader`] referencing.
    pub static ref LOADER: AxVmLoader = AxVmLoader;
}

/// `Loader` implementation in native rust.
#[derive(Clone, Debug)]
pub struct AxVmLoader;

impl<C: CurveAffine> LoadedEcPoint<C> for C {
    type Loader = AxVmLoader;

    fn loader(&self) -> &AxVmLoader {
        &LOADER
    }
}

// impl<F: PrimeField> FieldOps for F {
//     fn invert(&self) -> Option<F> {
//         self.invert().into()
//     }
// }

impl<F: PrimeField> LoadedScalar<F> for F {
    type Loader = AxVmLoader;

    fn loader(&self) -> &AxVmLoader {
        &LOADER
    }

    fn pow_var(&self, exp: &Self, _: usize) -> Self {
        let exp = fe_to_big(*exp).to_u64_digits();
        self.pow_vartime(exp)
    }
}

impl<C: CurveAffine> EcPointLoader<C> for AxVmLoader {
    type LoadedEcPoint = C;

    fn ec_point_load_const(&self, value: &C) -> Self::LoadedEcPoint {
        *value
    }

    fn ec_point_assert_eq(
        &self,
        annotation: &str,
        lhs: &Self::LoadedEcPoint,
        rhs: &Self::LoadedEcPoint,
    ) {
        lhs.eq(rhs)
            .then_some(())
            .unwrap_or_else(|| panic!("{:?}", Error::AssertionFailure(annotation.to_string())))
    }

    fn multi_scalar_multiplication(
        pairs: &[(&<Self as ScalarLoader<C::Scalar>>::LoadedScalar, &C)],
    ) -> C {
        let mut scalars = Vec::with_capacity(pairs.len());
        let mut base = Vec::with_capacity(pairs.len());
        for (scalar, point) in pairs {
            scalars.push(**scalar);
            base.push(**point);
        }
        msm_axvm_C(base, scalars)
    }
}

impl<F: PrimeField> ScalarLoader<F> for AxVmLoader {
    type LoadedScalar = F;

    fn load_const(&self, value: &F) -> Self::LoadedScalar {
        *value
    }

    fn assert_eq(&self, annotation: &str, lhs: &Self::LoadedScalar, rhs: &Self::LoadedScalar) {
        lhs.eq(rhs)
            .then_some(())
            .unwrap_or_else(|| panic!("{:?}", Error::AssertionFailure(annotation.to_string())))
    }
}

impl<C: CurveAffine> Loader<C> for AxVmLoader {}
