use std::sync::Arc;

use derive_more::derive::From;
use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
use openvm_circuit::{
    arch::instructions::exe::VmExe, system::memory::merkle::public_values::UserPublicValuesProof,
};
use openvm_stark_backend::{
    codec::{Decode, Encode},
    proof::Proof,
};
use openvm_transpiler::elf::Elf;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use verify_stark::NonRootStarkProof;

use crate::OPENVM_VERSION;

#[derive(From)]
pub enum ExecutableFormat {
    Elf(Elf),
    VmExe(VmExe<crate::F>),
    SharedVmExe(Arc<VmExe<crate::F>>),
}

impl<'a> From<&'a [u8]> for ExecutableFormat {
    fn from(bytes: &'a [u8]) -> Self {
        let elf = Elf::decode(bytes, MEM_SIZE.try_into().unwrap()).expect("Invalid ELF bytes");
        ExecutableFormat::Elf(elf)
    }
}
impl From<Vec<u8>> for ExecutableFormat {
    fn from(bytes: Vec<u8>) -> Self {
        ExecutableFormat::from(&bytes[..])
    }
}

/// Struct purely for encoding and decoding of [NonRootStarkProof].
#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VersionedNonRootStarkProof {
    /// The openvm major and minor version v{}.{}. The proof format will not change on patch
    /// versions.
    pub version: String,
    #[serde_as(as = "serde_with::hex::Hex")]
    pub proof: Vec<u8>,
    #[serde_as(as = "serde_with::hex::Hex")]
    pub user_pvs_proof: Vec<u8>,
}

impl VersionedNonRootStarkProof {
    pub fn new(proof: NonRootStarkProof) -> Result<Self> {
        Ok(Self {
            version: format!("v{}", OPENVM_VERSION),
            proof: proof.inner.encode_to_vec()?,
            user_pvs_proof: {
                let mut buf = Vec::new();
                proof.user_pvs_proof.encode::<crate::SC, _>(&mut buf)?;
                buf
            },
        })
    }
}

impl TryFrom<VersionedNonRootStarkProof> for NonRootStarkProof {
    type Error = std::io::Error;
    fn try_from(proof: VersionedNonRootStarkProof) -> Result<Self, std::io::Error> {
        let VersionedNonRootStarkProof {
            proof,
            user_pvs_proof,
            ..
        } = proof;
        Ok(Self {
            inner: Proof::<crate::SC>::decode_from_bytes(&proof)?,
            user_pvs_proof: UserPublicValuesProof::decode::<crate::SC, _>(
                &mut std::io::Cursor::new(&user_pvs_proof),
            )?,
        })
    }
}
