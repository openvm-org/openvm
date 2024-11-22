//! `Loader` implementation in native rust.

use std::{fmt::Debug, marker::PhantomData};

use axvm_ecc::{pairing::MultiMillerLoop, EccBinOps};
use halo2curves_axiom::{
    ff::PrimeField,
    group::{Group, ScalarMul},
    CurveAffine,
};
use lazy_static::lazy_static;
use snark_verifier::{
    loader::{EcPointLoader, Loader, ScalarLoader},
    pcs::{
        kzg::{KzgAccumulator, KzgAs, KzgDecidingKey},
        AccumulationDecider,
    },
    Error,
};

use super::traits::{AxVmEcPoint, AxVmScalar};

lazy_static! {
    /// NativeLoader instance for [`LoadedEcPoint::loader`] and
    /// [`LoadedScalar::loader`] referencing.
    pub static ref LOADER: AxVmLoader = AxVmLoader;
}

pub trait AxVmCurve: CurveAffine + ScalarMul<Self> + EccBinOps<Self::Base> {}

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
        AxVmEcPoint(*value)
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

impl<M: MultiMillerLoop, MOS> AccumulationDecider<M::G1Affine, AxVmLoader> for KzgAs<M, MOS>
where
    M::G1Affine: AxVmCurve + CurveAffine<ScalarExt = M::Fr, CurveExt = M::G1>,
    MOS: Clone + Debug,
{
    type DecidingKey = KzgDecidingKey<M>;

    fn decide(
        dk: &Self::DecidingKey,
        KzgAccumulator { lhs, rhs }: KzgAccumulator<M::G1Affine, AxVmLoader>,
    ) -> Result<(), Error> {
        let terms = [(&lhs.0, &dk.g2().into()), (&rhs.0, &(-dk.s_g2()).into())];
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
        accumulators: Vec<KzgAccumulator<M::G1Affine, AxVmLoader>>,
    ) -> Result<(), Error> {
        assert!(!accumulators.is_empty());
        accumulators
            .into_iter()
            .map(|accumulator| Self::decide(dk, accumulator))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }
}
