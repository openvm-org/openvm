use std::{
    io::{self, Read},
    marker::PhantomData,
};

use halo2curves_axiom::{
    bn256::{Fq, Fr as Halo2Fr, G1Affine},
    Coordinates, CurveAffine,
};
use itertools::Itertools;
use openvm_ecc_guest::algebra::IntMod;
use openvm_keccak256_guest::keccak256;
use openvm_pairing_guest::{
    affine_point::AffineCoords,
    bn254::{Bn254G1Affine as EcPoint, Fp, Scalar as Fr},
};
use snark_verifier::{
    loader::evm::{u256_to_fe, U256},
    util::transcript::{Transcript, TranscriptRead},
    Error,
};

use super::{
    loader::{OpenVmLoader, LOADER},
    traits::{OpenVmEcPoint, OpenVmScalar},
};

#[derive(Debug)]
pub struct OpenVmTranscript<C: CurveAffine, S, B> {
    stream: S,
    buf: B,
    _marker: PhantomData<C>,
}

impl<S> OpenVmTranscript<G1Affine, S, Vec<u8>> {
    /// Initialize [`OpenVmTranscript`] given readable or writeable stream for
    /// verifying or proving with [`OpenVmLoader`].
    pub fn new(stream: S) -> Self {
        Self {
            stream,
            buf: Vec::new(),
            _marker: PhantomData,
        }
    }
}
impl<S> Transcript<G1Affine, OpenVmLoader> for OpenVmTranscript<G1Affine, S, Vec<u8>> {
    fn loader(&self) -> &OpenVmLoader {
        &LOADER
    }

    fn squeeze_challenge(
        &mut self,
    ) -> <super::loader::OpenVmLoader as snark_verifier::loader::ScalarLoader<Halo2Fr>>::LoadedScalar
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
        let fr: Halo2Fr = u256_to_fe(U256::from_be_bytes(hash));
        use halo2curves_axiom::ff::PrimeField;
        let bytes: [u8; 32] = fr.to_repr();
        OpenVmScalar(Fr::from_le_bytes(&bytes), PhantomData)
    }

    fn common_ec_point(
        &mut self,
        ec_point: &OpenVmEcPoint<G1Affine, EcPoint>,
    ) -> Result<(), Error> {
        let mut x = [0; 32];
        let mut y = [0; 32];
        x.copy_from_slice(ec_point.0.x.as_le_bytes());
        y.copy_from_slice(ec_point.0.y.as_le_bytes());
        let coordinates = Option::<Coordinates<G1Affine>>::from(
            G1Affine {
                x: Fq::from_bytes(&x).unwrap(),
                y: Fq::from_bytes(&y).unwrap(),
            }
            .coordinates(),
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

    fn common_scalar(&mut self, scalar: &OpenVmScalar<Halo2Fr, Fr>) -> Result<(), Error> {
        self.buf.extend(scalar.0.to_be_bytes());

        Ok(())
    }
}

impl<S> TranscriptRead<G1Affine, OpenVmLoader> for OpenVmTranscript<G1Affine, S, Vec<u8>>
where
    S: Read,
{
    fn read_scalar(&mut self) -> Result<OpenVmScalar<Halo2Fr, Fr>, Error> {
        let mut data = [0; 32];
        self.stream
            .read_exact(data.as_mut())
            .map_err(|err| Error::Transcript(err.kind(), err.to_string()))?;
        let scalar = Fr::from_be_bytes(&data);
        let scalar = OpenVmScalar(scalar, PhantomData);
        self.common_scalar(&scalar)?;
        Ok(scalar)
    }

    fn read_ec_point(&mut self) -> Result<OpenVmEcPoint<G1Affine, EcPoint>, Error> {
        let [mut x, mut y] = [[0; 32]; 2];
        for repr in [&mut x, &mut y] {
            self.stream
                .read_exact(repr.as_mut())
                .map_err(|err| Error::Transcript(err.kind(), err.to_string()))?;
        }
        let x = Fp::from_be_bytes(&x);
        let y = Fp::from_be_bytes(&y);
        let ec_point = OpenVmEcPoint(EcPoint { x, y }, PhantomData);
        self.common_ec_point(&ec_point)?;
        Ok(ec_point)
    }
}
