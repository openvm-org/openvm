use std::{
    io::{self, Read},
    marker::PhantomData,
};

use axvm_ecc_guest::{algebra::IntMod, sw::SwPoint};
use axvm_keccak256_guest::keccak256;
use axvm_pairing_guest::{
    affine_point::AffineCoords,
    bn254::{Bn254Point, Fp, Fr},
};
use halo2curves_axiom::{
    bn256::{Fq, Fr as Halo2Fr, G1Affine},
    Coordinates, CurveAffine,
};
use itertools::Itertools;
use snark_verifier::{
    util::transcript::{Transcript, TranscriptRead},
    Error,
};

use super::{
    loader::{AxVmLoader, LOADER},
    traits::{AxVmEcPoint, AxVmScalar},
};

#[derive(Debug)]
pub struct AxVmTranscript<C: CurveAffine, S, B> {
    stream: S,
    buf: B,
    _marker: PhantomData<C>,
}

impl<S> Transcript<G1Affine, AxVmLoader> for AxVmTranscript<G1Affine, S, Vec<u8>> {
    fn loader(&self) -> &AxVmLoader {
        &LOADER
    }

    fn squeeze_challenge(
        &mut self,
    ) -> <super::loader::AxVmLoader as snark_verifier::loader::ScalarLoader<Halo2Fr>>::LoadedScalar
    {
        let data = self
            .buf
            .iter()
            .cloned()
            .chain(if self.buf.len() == 0x20 {
                Some(1)
            } else {
                None
            })
            .collect_vec();
        let hash = keccak256(&data);
        self.buf = hash.to_vec();
        AxVmScalar(Fr::from_be_bytes(&hash), PhantomData)
    }

    // is this sus?
    fn common_ec_point(
        &mut self,
        ec_point: &AxVmEcPoint<G1Affine, Bn254Point>,
    ) -> Result<(), Error> {
        let mut x = [0; 32];
        let mut y = [0; 32];
        x.copy_from_slice(ec_point.0.x().as_le_bytes());
        y.copy_from_slice(ec_point.0.y().as_le_bytes());
        let coordinates = Option::<Coordinates<G1Affine>>::from(
            G1Affine::new(Fq::from_bytes(&x).unwrap(), Fq::from_bytes(&y).unwrap()).coordinates(),
        )
        .ok_or_else(|| {
            Error::Transcript(
                io::ErrorKind::Other,
                "Invalid elliptic curve point".to_string(),
            )
        })?;

        [coordinates.x(), coordinates.y()].map(|coordinate| {
            self.buf.extend(coordinate.to_bytes().iter().rev());
        });

        Ok(())
    }

    fn common_scalar(&mut self, scalar: &AxVmScalar<Halo2Fr, Fr>) -> Result<(), Error> {
        self.buf.extend(scalar.0.as_be_bytes());

        Ok(())
    }
}

impl<S> TranscriptRead<G1Affine, AxVmLoader> for AxVmTranscript<G1Affine, S, Vec<u8>>
where
    S: Read,
{
    fn read_scalar(&mut self) -> Result<AxVmScalar<Halo2Fr, Fr>, Error> {
        let mut data = [0; 32];
        self.stream
            .read_exact(data.as_mut())
            .map_err(|err| Error::Transcript(err.kind(), err.to_string()))?;
        let scalar = Fr::from_be_bytes(&data);
        let scalar = AxVmScalar(scalar, PhantomData);
        // let scalar = AxVmScalar::<Halo2Fr, Fr>::from_repr_vartime(data).ok_or_else(|| {
        //     Error::Transcript(
        //         io::ErrorKind::Other,
        //         "Invalid scalar encoding in proof".to_string(),
        //     )
        // })?;
        self.common_scalar(&scalar)?;
        Ok(scalar)
    }

    fn read_ec_point(&mut self) -> Result<AxVmEcPoint<G1Affine, Bn254Point>, Error> {
        let [mut x, mut y] = [[0; 32]; 2];
        for repr in [&mut x, &mut y] {
            self.stream
                .read_exact(repr.as_mut())
                .map_err(|err| Error::Transcript(err.kind(), err.to_string()))?;
            repr.as_mut().reverse();
        }
        let x = Fp::from_be_bytes(&x);
        let y = Fp::from_be_bytes(&y);
        let ec_point = AxVmEcPoint(Bn254Point { x, y }, PhantomData);
        self.common_ec_point(&ec_point)?;
        Ok(ec_point)
    }
}
