use std::io::{self, Read, Write};

use halo2_base::{
    gates::circuit::builder::BaseCircuitBuilder,
    halo2_proofs::{
        halo2curves::bn256::{Fr, G1Affine},
        plonk::ProvingKey,
        SerdeFormat,
    },
};
use openvm_stark_sdk::openvm_stark_backend::codec::{Decode, Encode};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    keygen::StaticVerifierProvingKey,
    prover::{Halo2ProvingMetadata, Halo2ProvingPinning},
    wrapper::Halo2WrapperProvingKey,
};

const MAX_JSON_SECTION_LEN: usize = 64 * 1024 * 1024;

impl Encode for StaticVerifierProvingKey {
    fn encode<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_json_section(writer, &(&self.circuit, &self.shape))?;
        self.pinning.encode(writer)
    }
}

impl Decode for StaticVerifierProvingKey {
    fn decode<R: Read>(reader: &mut R) -> io::Result<Self> {
        let (circuit, shape) = read_json_section(reader)?;
        let pinning = Halo2ProvingPinning::decode(reader)?;
        Ok(Self {
            circuit,
            pinning,
            shape,
            #[cfg(feature = "halo2-gpu")]
            graph_prover: Default::default(),
        })
    }
}

impl Encode for Halo2WrapperProvingKey {
    fn encode<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        self.pinning.encode(writer)
    }
}

impl Decode for Halo2WrapperProvingKey {
    fn decode<R: Read>(reader: &mut R) -> io::Result<Self> {
        Ok(Self {
            pinning: Halo2ProvingPinning::decode(reader)?,
        })
    }
}

impl Encode for Halo2ProvingPinning {
    fn encode<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_json_section(writer, &self.metadata)?;
        self.pk.write(writer, SerdeFormat::RawBytes)
    }
}

impl Decode for Halo2ProvingPinning {
    fn decode<R: Read>(reader: &mut R) -> io::Result<Self> {
        let metadata: Halo2ProvingMetadata = read_json_section(reader)?;
        let pk = ProvingKey::<G1Affine>::read::<_, BaseCircuitBuilder<Fr>>(
            reader,
            SerdeFormat::RawBytes,
            metadata.config_params.clone(),
        )?;
        Ok(Self { pk, metadata })
    }
}

// Each JSON section is length-prefixed because it is followed by raw Halo2 proving-key bytes.
fn write_json_section<W: Write, T: Serialize>(writer: &mut W, value: &T) -> io::Result<()> {
    let bytes = serde_json::to_vec(value).map_err(io::Error::other)?;
    writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
    writer.write_all(&bytes)
}

fn read_json_section<R: Read, T: DeserializeOwned>(reader: &mut R) -> io::Result<T> {
    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes)?;
    let len = usize::try_from(u64::from_le_bytes(len_bytes)).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "JSON section length overflows usize",
        )
    })?;
    if len > MAX_JSON_SECTION_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "JSON section is too large",
        ));
    }
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    serde_json::from_slice(&bytes).map_err(io::Error::other)
}
