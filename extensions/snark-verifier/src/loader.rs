//! `Loader` implementation in native rust.

use core::fmt::Debug;

use halo2_proofs::halo2curves::bn256::{Fq as Halo2Fp, Fr as Halo2Fr, G1Affine};
use itertools::Itertools;
use lazy_static::lazy_static;
use openvm_ecc_guest::{
    algebra::{field::FieldExtension, IntMod},
    msm,
    weierstrass::WeierstrassPoint,
    AffinePoint,
};
use openvm_pairing_guest::{
    bn254::{Bn254, Bn254Fp as Fp, Bn254G1Affine as EcPoint, Fp2, Scalar as Fr},
    pairing::PairingCheck,
};
use serde::{Deserialize, Serialize};
use snark_verifier_sdk::snark_verifier::{
    halo2_base::halo2_proofs::{
        self,
        halo2curves::bn256::{Bn256, G2Affine},
    },
    loader::{EcPointLoader, Loader, ScalarLoader},
    pcs::{
        kzg::{KzgAccumulator, KzgAs, KzgDecidingKey, LimbsEncoding},
        AccumulationDecider, AccumulatorEncoding,
    },
    util::arithmetic::fe_from_limbs,
    Error,
};

use super::traits::{OpenVmEcPoint, OpenVmScalar};

lazy_static! {
    pub static ref LOADER: OpenVmLoader = OpenVmLoader;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenVmLoader;

impl<const LIMBS: usize, const BITS: usize> AccumulatorEncoding<G1Affine, OpenVmLoader>
    for LimbsEncoding<LIMBS, BITS>
{
    type Accumulator = KzgAccumulator<G1Affine, OpenVmLoader>;

    fn from_repr(limbs: &[&OpenVmScalar<Halo2Fr, Fr>]) -> Result<Self::Accumulator, Error> {
        assert_eq!(limbs.len(), 4 * LIMBS);

        let [lhs_x, lhs_y, rhs_x, rhs_y]: [_; 4] = limbs
            .chunks(LIMBS)
            .map(|limbs| {
                let v: [Halo2Fr; LIMBS] = limbs
                    .iter()
                    .map(|limb| {
                        let mut buf = limb.0.to_be_bytes();
                        buf.reverse();
                        Halo2Fr::from_bytes(&buf).expect("Halo2Fr::from_bytes")
                    })
                    .collect_vec()
                    .try_into()
                    .unwrap();
                fe_from_limbs::<_, Halo2Fp, LIMBS, BITS>(v)
            })
            .collect_vec()
            .try_into()
            .unwrap();
        let accumulator = KzgAccumulator::new(
            OpenVmEcPoint::new(
                EcPoint::from_xy(
                    Fp::from_le_bytes(&lhs_x.to_bytes()),
                    Fp::from_le_bytes(&lhs_y.to_bytes()),
                )
                .unwrap(),
            ),
            OpenVmEcPoint::new(
                EcPoint::from_xy(
                    Fp::from_le_bytes(&rhs_x.to_bytes()),
                    Fp::from_le_bytes(&rhs_y.to_bytes()),
                )
                .unwrap(),
            ),
        );
        Ok(accumulator)
    }
}

impl EcPointLoader<G1Affine> for OpenVmLoader {
    type LoadedEcPoint = OpenVmEcPoint<G1Affine, EcPoint>;

    fn ec_point_load_const(&self, value: &G1Affine) -> Self::LoadedEcPoint {
        // unchecked because this is a constant point
        let point = EcPoint::from_xy_unchecked(
            Fp::from_le_bytes(&value.x.to_bytes()),
            Fp::from_le_bytes(&value.y.to_bytes()),
        );
        // new(value.x(), value.y());
        OpenVmEcPoint::new(point)
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
        OpenVmEcPoint::new(msm::<EcPoint, Fr>(&scalars, &base))
    }
}

impl ScalarLoader<Halo2Fr> for OpenVmLoader {
    type LoadedScalar = OpenVmScalar<Halo2Fr, Fr>;

    fn load_const(&self, value: &Halo2Fr) -> Self::LoadedScalar {
        let value = Fr::from_le_bytes(&value.to_bytes());
        OpenVmScalar::new(value)
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
        let value = Fp::from_le_bytes(&value.to_bytes());
        OpenVmScalar::new(value)
    }

    fn assert_eq(&self, annotation: &str, lhs: &Self::LoadedScalar, rhs: &Self::LoadedScalar) {
        lhs.eq(rhs)
            .then_some(())
            .unwrap_or_else(|| panic!("{:?}", Error::AssertionFailure(annotation.to_string())))
    }
}

impl Loader<G1Affine> for OpenVmLoader {}

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
            let x = t.1.x.to_bytes();
            let y = t.1.y.to_bytes();
            let (x0, y0) = t.0.into_coords();
            let point = AffinePoint { x: x0, y: y0 };
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
