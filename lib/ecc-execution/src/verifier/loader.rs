//! `Loader` implementation in native rust.

use std::{fmt::Debug, marker::PhantomData};

use ax_ecc_lib::ec_msm::msm_axvm_C;
use halo2curves_axiom::{
    ff::{Field, PrimeField},
    group::ScalarMul,
    CurveAffine,
};
use lazy_static::lazy_static;
use snark_verifier_sdk::snark_verifier::{
    loader::{EcPointLoader, Loader, ScalarLoader},
    pcs::AccumulationDecider,
    Error,
};

use super::traits::{AxVmEcPoint, AxVmScalar};
use crate::common::{AffineCoords, FieldExtension, MultiMillerLoop};

lazy_static! {
    /// NativeLoader instance for [`LoadedEcPoint::loader`] and
    /// [`LoadedScalar::loader`] referencing.
    pub static ref LOADER: AxVmLoader = AxVmLoader;
}

pub trait AxVmCurve: CurveAffine + AffineCoords<Self::Base> + ScalarMul<Self> {}

/// `Loader` implementation in native rust.
#[derive(Clone, Debug)]
pub struct AxVmLoader;

// impl<C: CurveAffine> LoadedEcPoint<C> for AxVmCurve<C> {
//     type Loader = AxVmLoader;

//     fn loader(&self) -> &AxVmLoader<C> {
//         &LOADER
//     }
// }

// impl<F: PrimeField> FieldOps for F {
//     fn invert(&self) -> Option<F> {
//         self.invert().into()
//     }
// }

impl<C: AxVmCurve> EcPointLoader<C> for AxVmLoader {
    type LoadedEcPoint = AxVmEcPoint<C>;

    fn ec_point_load_const(&self, value: &C) -> Self::LoadedEcPoint {
        AxVmEcPoint(value.clone())
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
        pairs: &[(
            &<Self as ScalarLoader<C::Scalar>>::LoadedScalar,
            &Self::LoadedEcPoint,
        )],
    ) -> Self::LoadedEcPoint {
        let mut scalars = Vec::with_capacity(pairs.len());
        let mut base = Vec::with_capacity(pairs.len());
        for (scalar, point) in pairs {
            scalars.push(scalar.0);
            base.push(point.0);
        }
        AxVmEcPoint(msm_axvm_C(base, scalars))
    }
}

impl<F: PrimeField> ScalarLoader<F> for AxVmLoader {
    type LoadedScalar = AxVmScalar<F>;

    fn load_const(&self, value: &F) -> Self::LoadedScalar {
        AxVmScalar(*value)
    }

    fn assert_eq(&self, annotation: &str, lhs: &Self::LoadedScalar, rhs: &Self::LoadedScalar) {
        lhs.eq(rhs)
            .then_some(())
            .unwrap_or_else(|| panic!("{:?}", Error::AssertionFailure(annotation.to_string())))
    }
}

impl<C: AxVmCurve> Loader<C> for AxVmLoader {}

/// KZG accumulation scheme. The second generic `MOS` stands for different kind
/// of multi-open scheme.
#[derive(Clone, Debug)]
pub struct AxVmKzgAs<M, MOS>(PhantomData<(M, MOS)>);

impl<M, MOS, C: AxVmCurve, Fp, Fp2, Fp12, const BITS: usize> AccumulationDecider<C, AxVmLoader>
    for AxVmKzgAs<M, MOS>
where
    M: MultiMillerLoop<Fp, Fp2, Fp12, BITS>,
    C: CurveAffine<ScalarExt = Fp>,
    MOS: Clone + Debug,
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    type DecidingKey = KzgDecidingKey<M>;

    fn decide(
        dk: &Self::DecidingKey,
        KzgAccumulator { lhs, rhs }: KzgAccumulator<M::G1Affine, NativeLoader>,
    ) -> Result<(), Error> {
        let terms = [(&lhs, &dk.g2.into()), (&rhs, &(-dk.s_g2).into())];
        bool::from(
            M::multi_miller_loop(&terms)
                .final_exponentiation()
                .is_identity(),
        )
        .then_some(())
        .ok_or_else(|| Error::AssertionFailure("e(lhs, g2)·e(rhs, -s_g2) == O".to_string()))
    }

    fn decide_all(
        dk: &Self::DecidingKey,
        accumulators: Vec<KzgAccumulator<M::G1Affine, NativeLoader>>,
    ) -> Result<(), Error> {
        assert!(!accumulators.is_empty());
        accumulators
            .into_iter()
            .map(|accumulator| Self::decide(dk, accumulator))
            .try_collect::<_, Vec<_>, _>()?;
        Ok(())
    }
}
