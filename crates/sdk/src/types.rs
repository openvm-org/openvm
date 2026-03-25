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
use openvm_verify_stark_host::{deferral::DeferralMerkleProofs, NonRootStarkProof};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

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

/// Number of bytes in a Bn254.
#[allow(dead_code)]
pub(crate) const BN254_BYTES: usize = 32;
/// Number of Bn254 in `accumulator` field (KZG accumulator).
pub const NUM_BN254_ACCUMULATOR: usize = 12;
/// Number of Bn254 in `proof` field for a circuit with only 1 advice column.
#[cfg(feature = "evm-prove")]
#[allow(dead_code)]
pub(crate) const NUM_BN254_PROOF: usize = 43;

#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProofData {
    #[serde_as(as = "serde_with::hex::Hex")]
    /// KZG accumulator.
    pub accumulator: Vec<u8>,
    #[serde_as(as = "serde_with::hex::Hex")]
    /// Bn254 proof in little-endian bytes. The circuit only has 1 advice column, so the proof is
    /// of length `NUM_BN254_PROOF * BN254_BYTES`.
    pub proof: Vec<u8>,
}

// =================== EVM types (evm-prove feature) ===================

#[cfg(feature = "evm-prove")]
pub use openvm_static_verifier::wrapper::EvmVerifierByteCode;

#[cfg(feature = "evm-prove")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvmHalo2Verifier {
    pub halo2_verifier_code: String,
    pub openvm_verifier_code: String,
    pub openvm_verifier_interface: String,
    pub artifact: EvmVerifierByteCode,
}

/// Application execution commitment pair (big-endian 32-byte values).
#[cfg(feature = "evm-prove")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppExecutionCommit {
    #[serde(with = "hex_bytes32")]
    pub app_exe_commit: openvm_continuations::CommitBytes,
    #[serde(with = "hex_bytes32")]
    pub app_vm_commit: openvm_continuations::CommitBytes,
}

/// Custom serde for CommitBytes as hex-encoded [u8; 32].
#[cfg(feature = "evm-prove")]
mod hex_bytes32 {
    use openvm_continuations::CommitBytes;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(val: &CommitBytes, s: S) -> Result<S::Ok, S::Error> {
        hex::encode(val.as_slice()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<CommitBytes, D::Error> {
        let hex_str = String::deserialize(d)?;
        let bytes: [u8; 32] = hex::decode(hex_str)
            .map_err(serde::de::Error::custom)?
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected 32 bytes"))?;
        Ok(CommitBytes::new(bytes))
    }
}

#[cfg(feature = "evm-prove")]
#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvmProof {
    /// The openvm major and minor version v{}.{}. The proof format will not change on patch
    /// versions.
    pub version: String,
    #[serde(flatten)]
    /// Bn254 public value app commits.
    pub app_commit: AppExecutionCommit,
    #[serde_as(as = "serde_with::hex::Hex")]
    /// User public values packed into bytes.
    pub user_public_values: Vec<u8>,
    /// Byte encoding of the `proof`.
    pub proof_data: ProofData,
}

#[cfg(feature = "evm-prove")]
#[derive(Debug, thiserror::Error)]
pub enum EvmProofConversionError {
    #[error("Invalid length of instances: expected at least 3, got {0}")]
    InvalidLengthInstances(usize),
    #[error("Invalid length of user public values")]
    InvalidUserPublicValuesLength,
}

#[cfg(feature = "evm-prove")]
impl EvmProof {
    #[cfg(feature = "evm-verify")]
    /// Return bytes calldata to be passed to the verifier contract.
    pub fn verifier_calldata(self) -> Vec<u8> {
        use alloy_sol_types::SolCall;

        use crate::IOpenVmHalo2Verifier;

        let EvmProof {
            user_public_values,
            app_commit,
            proof_data,
            version: _,
        } = self;

        let ProofData { accumulator, proof } = proof_data;

        let mut proof_data_bytes = accumulator;
        proof_data_bytes.extend(proof);

        IOpenVmHalo2Verifier::verifyCall {
            publicValues: user_public_values.into(),
            proofData: proof_data_bytes.into(),
            appExeCommit: (*app_commit.app_exe_commit.as_slice()).into(),
            appVmCommit: (*app_commit.app_vm_commit.as_slice()).into(),
        }
        .abi_encode()
    }

    #[cfg(feature = "evm-verify")]
    pub fn fallback_calldata(&self) -> Vec<u8> {
        let raw: openvm_static_verifier::keygen::RawEvmProof = self.clone().into();
        encode_raw_evm_proof_calldata(&raw)
    }
}

/// Encode a [`RawEvmProof`](openvm_static_verifier::keygen::RawEvmProof) as calldata for the
/// fallback (raw) verifier.
///
/// Format: each instance as 32-byte big-endian, followed by raw proof bytes.
#[cfg(feature = "evm-verify")]
pub fn encode_raw_evm_proof_calldata(
    proof: &openvm_static_verifier::keygen::RawEvmProof,
) -> Vec<u8> {
    let mut calldata = Vec::new();
    for instance in &proof.instances {
        // Fr::to_bytes() is little-endian; EVM expects big-endian
        let mut bytes = instance.to_bytes();
        bytes.reverse();
        calldata.extend_from_slice(&bytes);
    }
    calldata.extend_from_slice(&proof.proof);
    calldata
}

/// Convert `RawEvmProof` → `EvmProof`.
///
/// Instance layout (with KZG accumulator from wrapper circuit):
/// - `instances[0..12]`: KZG accumulator (12 Fr values)
/// - `instances[12]`: app_exe_commit (Fr)
/// - `instances[13]`: app_vk_commit (Fr)
/// - `instances[14..]`: user public values (each byte as Fr)
#[cfg(feature = "evm-prove")]
impl From<openvm_static_verifier::keygen::RawEvmProof> for EvmProof {
    fn from(raw: openvm_static_verifier::keygen::RawEvmProof) -> Self {
        use openvm_continuations::CommitBytes;

        let openvm_static_verifier::keygen::RawEvmProof { instances, proof } = raw;
        assert!(
            instances.len() > NUM_BN254_ACCUMULATOR + 2,
            "RawEvmProof instances must have at least {} elements (accumulator + exe commit + vk commit)",
            NUM_BN254_ACCUMULATOR + 2
        );

        // instances[0..12] are the KZG accumulator
        let accumulator = instances[0..NUM_BN254_ACCUMULATOR]
            .iter()
            .flat_map(|f| f.to_bytes())
            .collect::<Vec<_>>();

        // Reverse each 32-byte chunk for big-endian EVM format
        let mut evm_accumulator = Vec::with_capacity(accumulator.len());
        accumulator
            .chunks(BN254_BYTES)
            .for_each(|chunk| evm_accumulator.extend(chunk.iter().rev().copied()));

        // instances[12] and [13] are Fr values encoding commits.
        // Fr::to_bytes() returns 32 bytes in little-endian; CommitBytes expects big-endian.
        let mut app_exe_bytes = instances[NUM_BN254_ACCUMULATOR].to_bytes();
        app_exe_bytes.reverse();
        let mut app_vm_bytes = instances[NUM_BN254_ACCUMULATOR + 1].to_bytes();
        app_vm_bytes.reverse();

        let user_public_values = instances[NUM_BN254_ACCUMULATOR + 2..]
            .iter()
            .map(|f| {
                // Each user public value is a single byte stored in the least significant position
                f.to_bytes()[0]
            })
            .collect::<Vec<u8>>();

        let app_commit = AppExecutionCommit {
            app_exe_commit: CommitBytes::new(app_exe_bytes),
            app_vm_commit: CommitBytes::new(app_vm_bytes),
        };

        Self {
            version: format!("v{OPENVM_VERSION}"),
            app_commit,
            user_public_values,
            proof_data: ProofData {
                accumulator: evm_accumulator,
                proof,
            },
        }
    }
}

/// Convert `EvmProof` → `RawEvmProof`.
#[cfg(feature = "evm-prove")]
impl From<EvmProof> for openvm_static_verifier::keygen::RawEvmProof {
    fn from(evm_proof: EvmProof) -> Self {
        use openvm_static_verifier::Fr;

        let EvmProof {
            app_commit,
            user_public_values,
            proof_data,
            version: _,
        } = evm_proof;

        let ProofData { accumulator, proof } = proof_data;

        // Reverse each 32-byte chunk from big-endian (EVM) to little-endian (Fr)
        let mut reversed_accumulator = Vec::with_capacity(accumulator.len());
        accumulator
            .chunks(BN254_BYTES)
            .for_each(|chunk| reversed_accumulator.extend(chunk.iter().rev().copied()));

        // CommitBytes is big-endian; Fr::from_bytes expects little-endian
        let mut app_exe_bytes = *app_commit.app_exe_commit.as_slice();
        app_exe_bytes.reverse();
        let app_exe_fr = Fr::from_bytes(&app_exe_bytes).unwrap();

        let mut app_vm_bytes = *app_commit.app_vm_commit.as_slice();
        app_vm_bytes.reverse();
        let app_vm_fr = Fr::from_bytes(&app_vm_bytes).unwrap();

        let user_pvs_frs: Vec<Fr> = user_public_values
            .into_iter()
            .map(|byte| {
                let mut bytes = [0u8; 32];
                bytes[0] = byte;
                Fr::from_bytes(&bytes).unwrap()
            })
            .collect();

        // Reconstruct instances: accumulator + commits + user PVs
        let mut instances = Vec::new();
        for chunk in reversed_accumulator.chunks(BN254_BYTES) {
            let c: [u8; 32] = chunk.try_into().unwrap();
            instances.push(Fr::from_bytes(&c).unwrap());
        }
        instances.push(app_exe_fr);
        instances.push(app_vm_fr);
        instances.extend(user_pvs_frs);

        openvm_static_verifier::keygen::RawEvmProof { instances, proof }
    }
}

// =================== Non-EVM types ===================

/// Struct purely for encoding and decoding of [NonRootStarkProof].
#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize, Encode, Decode)]
pub struct VersionedNonRootStarkProof {
    /// The openvm major and minor version v{}.{}. The proof format will not change on patch
    /// versions.
    pub version: String,
    #[serde_as(as = "serde_with::hex::Hex")]
    pub proof: Vec<u8>,
    #[serde_as(as = "serde_with::hex::Hex")]
    pub user_pvs_proof: Vec<u8>,
    #[serde(default)]
    #[serde_as(as = "Option<serde_with::hex::Hex>")]
    pub deferral_merkle_proofs: Option<Vec<u8>>,
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
            deferral_merkle_proofs: proof
                .deferral_merkle_proofs
                .map(|ref dmp| {
                    let mut buf = Vec::new();
                    dmp.encode(&mut buf)?;
                    Ok::<_, std::io::Error>(buf)
                })
                .transpose()?,
        })
    }
}

impl TryFrom<VersionedNonRootStarkProof> for NonRootStarkProof {
    type Error = std::io::Error;
    fn try_from(proof: VersionedNonRootStarkProof) -> Result<Self, std::io::Error> {
        let VersionedNonRootStarkProof {
            proof,
            user_pvs_proof,
            deferral_merkle_proofs,
            ..
        } = proof;
        Ok(Self {
            inner: Proof::<crate::SC>::decode_from_bytes(&proof)?,
            user_pvs_proof: UserPublicValuesProof::decode::<crate::SC, _>(
                &mut std::io::Cursor::new(&user_pvs_proof),
            )?,
            deferral_merkle_proofs: deferral_merkle_proofs
                .map(|bytes| DeferralMerkleProofs::decode(&mut std::io::Cursor::new(&bytes)))
                .transpose()?,
        })
    }
}
