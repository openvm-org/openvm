//! `Loader` implementation in native rust.

use std::{fmt::Debug, marker::PhantomData};

use axvm_ecc::{
    algebra::{Field, IntMod},
    bn254::{self, Bn254, EcPoint, Fp, Fr},
    msm,
    pairing::MultiMillerLoop,
    sw::SwPoint,
    AffineCoords, AffinePoint, EccBinOps,
};
use halo2curves_axiom::{
    bn256::{Bn256, Fq as Halo2Fp, Fr as Halo2Fr, G1Affine},
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

// pub trait AxVmCurve: CurveAffine + ScalarMul<Self> + EccBinOps<Self::Base> {}

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

impl EcPointLoader<G1Affine> for AxVmLoader {
    type LoadedEcPoint = AxVmEcPoint<G1Affine, EcPoint>;

    fn ec_point_load_const(&self, value: &G1Affine) -> Self::LoadedEcPoint {
        let point = EcPoint {
            x: Fp::from_be_bytes(&value.x().to_bytes()),
            y: Fp::from_be_bytes(&value.y().to_bytes()),
        };
        // new(value.x(), value.y());
        AxVmEcPoint(point, PhantomData)
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
        pairs: &[(&AxVmScalar<Halo2Fr, Fr>, &AxVmEcPoint<G1Affine, EcPoint>)],
    ) -> Self::LoadedEcPoint {
        let mut scalars: Vec<Fr> = Vec::with_capacity(pairs.len());
        let mut base: Vec<EcPoint> = Vec::with_capacity(pairs.len());
        for (scalar, point) in pairs {
            scalars.push(scalar.0);
            base.push(point.0);
        }
        AxVmEcPoint(msm(&scalars, &base), PhantomData)
    }
}

impl ScalarLoader<Halo2Fr> for AxVmLoader {
    type LoadedScalar = AxVmScalar<Halo2Fr, Fr>;

    fn load_const(&self, value: &Halo2Fr) -> Self::LoadedScalar {
        let value = Fr::from_be_bytes(&value.to_bytes());
        AxVmScalar(value, PhantomData)
    }

    fn assert_eq(&self, annotation: &str, lhs: &Self::LoadedScalar, rhs: &Self::LoadedScalar) {
        lhs.eq(rhs)
            .then_some(())
            .unwrap_or_else(|| panic!("{:?}", Error::AssertionFailure(annotation.to_string())))
    }
}

impl ScalarLoader<Halo2Fp> for AxVmLoader {
    type LoadedScalar = AxVmScalar<Halo2Fp, Fp>;

    fn load_const(&self, value: &Halo2Fp) -> Self::LoadedScalar {
        let value = Fp::from_be_bytes(&value.to_bytes());
        AxVmScalar(value, PhantomData)
    }

    fn assert_eq(&self, annotation: &str, lhs: &Self::LoadedScalar, rhs: &Self::LoadedScalar) {
        lhs.eq(rhs)
            .then_some(())
            .unwrap_or_else(|| panic!("{:?}", Error::AssertionFailure(annotation.to_string())))
    }
}

impl Loader<G1Affine> for AxVmLoader {}

/// KZG accumulation scheme. The second generic `MOS` stands for different kind
/// of multi-open scheme.
#[derive(Clone, Debug)]
pub struct AxVmKzgAs<M, MOS>(PhantomData<(M, MOS)>);

impl<MOS> AccumulationDecider<G1Affine, AxVmLoader> for KzgAs<Bn256, MOS>
where
    MOS: Clone + Debug,
{
    type DecidingKey = KzgDecidingKey<Bn256>;

    fn decide(
        dk: &Self::DecidingKey,
        KzgAccumulator { lhs, rhs }: KzgAccumulator<G1Affine, AxVmLoader>,
    ) -> Result<(), Error> {
        let terms = [(&lhs.0, &dk.g2().into()), (&rhs.0, &(-dk.s_g2()).into())];
        bool::from(
            bn254::multi_miller_loop(&terms)
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
