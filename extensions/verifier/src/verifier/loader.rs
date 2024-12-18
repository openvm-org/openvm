//! `Loader` implementation in native rust.

use std::{fmt::Debug, marker::PhantomData};

use halo2curves_axiom::bn256::{Bn256, Fq as Halo2Fp, Fr as Halo2Fr, G1Affine, G2Affine};
use lazy_static::lazy_static;
use openvm_ecc_guest::{
    algebra::{field::FieldExtension, IntMod},
    msm, AffinePoint,
};
use openvm_pairing_guest::{
    affine_point::AffineCoords,
    bn254::{Bn254, Bn254Fp as Fp, Bn254G1Affine as EcPoint, Fp2, Scalar as Fr},
    pairing::PairingCheck,
};
use snark_verifier::{
    loader::{EcPointLoader, Loader, ScalarLoader},
    pcs::{
        kzg::{KzgAccumulator, KzgAs, KzgDecidingKey},
        AccumulationDecider,
    },
    Error,
};

use super::traits::{OpenVmEcPoint, OpenVmScalar};

lazy_static! {
    pub static ref LOADER: OpenVmLoader = OpenVmLoader;
}

#[derive(Clone, Debug)]
pub struct OpenVmLoader;

impl EcPointLoader<G1Affine> for OpenVmLoader {
    type LoadedEcPoint = OpenVmEcPoint<G1Affine, EcPoint>;

    fn ec_point_load_const(&self, value: &G1Affine) -> Self::LoadedEcPoint {
        let point = EcPoint {
            x: Fp::from_be_bytes(&value.x().to_bytes()),
            y: Fp::from_be_bytes(&value.y().to_bytes()),
        };
        // new(value.x(), value.y());
        OpenVmEcPoint(point, PhantomData)
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
            &OpenVmScalar<Halo2Fr, Fr>,
            &OpenVmEcPoint<G1Affine, EcPoint>,
        )],
    ) -> Self::LoadedEcPoint {
        let mut scalars = Vec::with_capacity(pairs.len());
        let mut base = Vec::with_capacity(pairs.len());
        for (scalar, point) in pairs {
            scalars.push(scalar.0.clone());
            base.push(point.0.clone());
        }
        OpenVmEcPoint(msm::<EcPoint, Fr>(&scalars, &base), PhantomData)
    }
}

impl ScalarLoader<Halo2Fr> for OpenVmLoader {
    type LoadedScalar = OpenVmScalar<Halo2Fr, Fr>;

    fn load_const(&self, value: &Halo2Fr) -> Self::LoadedScalar {
        let value = Fr::from_be_bytes(&value.to_bytes());
        OpenVmScalar(value, PhantomData)
    }

    fn assert_eq(&self, annotation: &str, lhs: &Self::LoadedScalar, rhs: &Self::LoadedScalar) {
        lhs.eq(rhs)
            .then_some(())
            .unwrap_or_else(|| panic!("{:?}", Error::AssertionFailure(annotation.to_string())))
    }
}

impl ScalarLoader<Halo2Fp> for OpenVmLoader {
    type LoadedScalar = OpenVmScalar<Halo2Fp, Fp>;

    fn load_const(&self, value: &Halo2Fp) -> Self::LoadedScalar {
        let value = Fp::from_be_bytes(&value.to_bytes());
        OpenVmScalar(value, PhantomData)
    }

    fn assert_eq(&self, annotation: &str, lhs: &Self::LoadedScalar, rhs: &Self::LoadedScalar) {
        lhs.eq(rhs)
            .then_some(())
            .unwrap_or_else(|| panic!("{:?}", Error::AssertionFailure(annotation.to_string())))
    }
}

impl Loader<G1Affine> for OpenVmLoader {}

/// KZG accumulation scheme. The second generic `MOS` stands for different kind
/// of multi-open scheme.
#[derive(Clone, Debug)]
pub struct OpenVmKzgAs<M, MOS>(PhantomData<(M, MOS)>);

impl<MOS> AccumulationDecider<G1Affine, OpenVmLoader> for KzgAs<Bn256, MOS>
where
    MOS: Clone + Debug,
{
    type DecidingKey = KzgDecidingKey<Bn256>;

    #[allow(non_snake_case)]
    fn decide(
        dk: &Self::DecidingKey,
        KzgAccumulator { lhs, rhs }: KzgAccumulator<G1Affine, OpenVmLoader>,
    ) -> Result<(), Error> {
        let terms: [(EcPoint, G2Affine); 2] = [(lhs.0, dk.g2()), (rhs.0, (-dk.s_g2()))];
        let mut P = Vec::with_capacity(2);
        let mut Q = Vec::with_capacity(2);
        for t in terms {
            let x = t.1.x().to_bytes();
            let y = t.1.y().to_bytes();
            let point = AffinePoint { x: t.0.x, y: t.0.y };
            P.push(point);
            let point = AffinePoint {
                x: Fp2::from_coeffs([Fp::from_le_bytes(&x[0..32]), Fp::from_le_bytes(&x[32..64])]),
                y: Fp2::from_coeffs([Fp::from_le_bytes(&y[0..32]), Fp::from_le_bytes(&y[32..64])]),
            };
            Q.push(point);
        }
        Bn254::pairing_check(&P, &Q).unwrap();
        Ok(())
    }

    fn decide_all(
        dk: &Self::DecidingKey,
        accumulators: Vec<KzgAccumulator<G1Affine, OpenVmLoader>>,
    ) -> Result<(), Error> {
        assert!(!accumulators.is_empty());
        accumulators
            .into_iter()
            .map(|accumulator| Self::decide(dk, accumulator))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }
}
