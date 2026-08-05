#[cfg(feature = "evm-prove")]
use std::mem::size_of;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(feature = "evm-verify")]
use alloy_sol_types::SolCall;
use derive_more::derive::From;
use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
#[cfg(feature = "evm-prove")]
use openvm_circuit::arch::{U16_CELLS_PER_PUBLIC_VALUE, U16_CELL_SIZE};
use openvm_circuit::{
    arch::instructions::exe::VmExe,
    system::{memory::dimensions::MemoryDimensions, public_values::proof::PublicValuesOpening},
};
use openvm_continuations::CommitBytes;
use openvm_stark_backend::{
    codec::{Decode, Encode},
    proof::Proof,
};
#[cfg(feature = "evm-prove")]
use openvm_static_verifier::Fr;
use openvm_transpiler::elf::Elf;
use openvm_verify_stark_host::{
    deferral::DeferralMerkleProofs, pvs::VkCommit, vk::VerificationBaseline, VmStarkProof,
};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

#[cfg(feature = "evm-verify")]
use crate::solidity::IOpenVmHalo2Verifier;
use crate::OPENVM_VERSION;

#[cfg(feature = "evm-prove")]
const USER_PUBLIC_VALUES_COUNT_OFFSET: usize = NUM_BN254_ACCUMULATOR + 2;
#[cfg(feature = "evm-prove")]
const USER_PUBLIC_VALUES_OFFSET: usize = USER_PUBLIC_VALUES_COUNT_OFFSET + 1;

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

/// Input accepted by SDK compile methods.
pub enum ExecutableInput {
    /// An in-memory executable.
    Format(ExecutableFormat),
    /// An ELF file path. Path provenance is preserved for source maps.
    ElfFile(PathBuf),
    /// An in-memory executable with the ELF file it was built from.
    #[cfg(feature = "rvr")]
    WithElfPath {
        executable: ExecutableFormat,
        elf_path: PathBuf,
    },
}

impl ExecutableInput {
    #[cfg(feature = "rvr")]
    pub fn with_elf_path(
        executable: impl Into<ExecutableFormat>,
        elf_path: impl Into<PathBuf>,
    ) -> Self {
        Self::WithElfPath {
            executable: executable.into(),
            elf_path: elf_path.into(),
        }
    }
}

impl From<&Path> for ExecutableInput {
    fn from(path: &Path) -> Self {
        Self::ElfFile(path.to_path_buf())
    }
}

impl From<PathBuf> for ExecutableInput {
    fn from(path: PathBuf) -> Self {
        Self::ElfFile(path)
    }
}

impl<T> From<T> for ExecutableInput
where
    ExecutableFormat: From<T>,
{
    fn from(value: T) -> Self {
        Self::Format(value.into())
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProofData {
    #[serde(with = "prefixed_hex")]
    /// KZG accumulator.
    pub accumulator: Vec<u8>,
    #[serde(with = "prefixed_hex")]
    /// Bn254 proof in little-endian bytes. The circuit only has 1 advice column, so the proof is
    /// of length `NUM_BN254_PROOF * BN254_BYTES`.
    pub proof: Vec<u8>,
}

mod prefixed_hex {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("0x{}", hex::encode(bytes)))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        let hex_str = String::deserialize(deserializer)?;
        let hex_str = hex_str.strip_prefix("0x").unwrap_or(&hex_str);
        hex::decode(hex_str).map_err(serde::de::Error::custom)
    }
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AppExecutionCommit {
    pub app_exe_commit: openvm_continuations::CommitBytes,
    pub app_vm_commit: openvm_continuations::CommitBytes,
}

#[cfg(feature = "evm-prove")]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvmProof {
    /// The openvm major and minor version v{}.{}. The proof format will not change on patch
    /// versions.
    pub version: String,
    #[serde(flatten)]
    /// Bn254 public value app commits.
    pub app_commit: AppExecutionCommit,
    /// Number of `u64` values in the revealed public-output prefix.
    pub num_public_values: u32,
    #[serde(with = "prefixed_hex")]
    /// Fixed-capacity `u16` cells, each encoded as two little-endian bytes.
    pub user_public_values: Vec<u8>,
    /// Byte encoding of the `proof`.
    pub proof_data: ProofData,
}

#[cfg(feature = "evm-prove")]
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EvmProofConversionError {
    #[error("invalid instance count: expected at least {expected}, got {actual}")]
    InvalidInstanceCount { expected: usize, actual: usize },
    #[error("invalid accumulator length: expected {expected} bytes, got {actual}")]
    InvalidAccumulatorLength { expected: usize, actual: usize },
    #[error("invalid proof length: expected {expected} bytes, got {actual}")]
    InvalidProofLength { expected: usize, actual: usize },
    #[error(
        "invalid user public-values byte length {0}: expected complete little-endian u16 cells"
    )]
    InvalidUserPublicValuesLength(usize),
    #[error("invalid user public-value cell count {0}: expected four cells times a nonzero power of two")]
    InvalidUserPublicValueCellCount(usize),
    #[error("public-values count is not a canonical u32")]
    NonCanonicalPublicValuesCount,
    #[error("public-values count {count} exceeds configured capacity {capacity}")]
    PublicValuesCountOutOfRange { count: u32, capacity: usize },
    #[error("public-value cell {0} is not a canonical u16")]
    NonCanonicalPublicValueCell(usize),
    #[error("nonzero public-value padding at cell {0}")]
    NonzeroPublicValuePadding(usize),
    #[error("invalid BN254 scalar encoding for {0}")]
    InvalidBn254Scalar(&'static str),
}

#[cfg(feature = "evm-prove")]
impl EvmProof {
    pub fn validate(&self) -> Result<(), EvmProofConversionError> {
        let expected_accumulator_len = NUM_BN254_ACCUMULATOR * BN254_BYTES;
        if self.proof_data.accumulator.len() != expected_accumulator_len {
            return Err(EvmProofConversionError::InvalidAccumulatorLength {
                expected: expected_accumulator_len,
                actual: self.proof_data.accumulator.len(),
            });
        }
        for chunk in self.proof_data.accumulator.chunks_exact(BN254_BYTES) {
            decode_bn254_be(chunk, "KZG accumulator")?;
        }
        let expected_proof_len = NUM_BN254_PROOF * BN254_BYTES;
        if self.proof_data.proof.len() != expected_proof_len {
            return Err(EvmProofConversionError::InvalidProofLength {
                expected: expected_proof_len,
                actual: self.proof_data.proof.len(),
            });
        }
        decode_bn254_be(
            self.app_commit.app_exe_commit.as_slice(),
            "app executable commitment",
        )?;
        decode_bn254_be(
            self.app_commit.app_vm_commit.as_slice(),
            "app VM commitment",
        )?;

        if !self.user_public_values.len().is_multiple_of(U16_CELL_SIZE) {
            return Err(EvmProofConversionError::InvalidUserPublicValuesLength(
                self.user_public_values.len(),
            ));
        }
        let num_cells = self.user_public_values.len() / U16_CELL_SIZE;
        validate_public_value_cell_count(num_cells)?;
        let capacity = num_cells / U16_CELLS_PER_PUBLIC_VALUE;
        if self.num_public_values as usize > capacity {
            return Err(EvmProofConversionError::PublicValuesCountOutOfRange {
                count: self.num_public_values,
                capacity,
            });
        }
        let published_cells = self.num_public_values as usize * U16_CELLS_PER_PUBLIC_VALUE;
        for (offset, cell) in self.user_public_values[published_cells * U16_CELL_SIZE..]
            .chunks_exact(U16_CELL_SIZE)
            .enumerate()
        {
            if cell != [0, 0] {
                return Err(EvmProofConversionError::NonzeroPublicValuePadding(
                    published_cells + offset,
                ));
            }
        }
        Ok(())
    }

    #[cfg(feature = "evm-verify")]
    /// Return bytes calldata to be passed to the verifier contract.
    pub fn verifier_calldata(self) -> Result<Vec<u8>, EvmProofConversionError> {
        self.validate()?;
        let EvmProof {
            num_public_values,
            user_public_values,
            app_commit,
            proof_data,
            version: _,
        } = self;

        let ProofData { accumulator, proof } = proof_data;

        let mut proof_data_bytes = accumulator;
        proof_data_bytes.extend(proof);

        Ok(IOpenVmHalo2Verifier::verifyCall {
            publicValuesCount: num_public_values,
            publicValues: user_public_values.into(),
            proofData: proof_data_bytes.into(),
            appExeCommit: (*app_commit.app_exe_commit.as_slice()).into(),
            appVmCommit: (*app_commit.app_vm_commit.as_slice()).into(),
        }
        .abi_encode())
    }

    #[cfg(feature = "evm-verify")]
    pub fn fallback_calldata(&self) -> Result<Vec<u8>, EvmProofConversionError> {
        let raw = openvm_static_verifier::keygen::RawEvmProof::try_from(self.clone())?;
        Ok(encode_raw_evm_proof_calldata(&raw))
    }
}

#[cfg(feature = "evm-prove")]
fn validate_public_value_cell_count(num_cells: usize) -> Result<(), EvmProofConversionError> {
    if !num_cells.is_multiple_of(U16_CELLS_PER_PUBLIC_VALUE)
        || !(num_cells / U16_CELLS_PER_PUBLIC_VALUE).is_power_of_two()
    {
        return Err(EvmProofConversionError::InvalidUserPublicValueCellCount(
            num_cells,
        ));
    }
    Ok(())
}

#[cfg(feature = "evm-prove")]
fn decode_bn254_be(
    bytes: &[u8],
    component: &'static str,
) -> Result<openvm_static_verifier::Fr, EvmProofConversionError> {
    let mut bytes: [u8; BN254_BYTES] = bytes
        .try_into()
        .map_err(|_| EvmProofConversionError::InvalidBn254Scalar(component))?;
    bytes.reverse();
    Option::from(openvm_static_verifier::Fr::from_bytes(&bytes))
        .ok_or(EvmProofConversionError::InvalidBn254Scalar(component))
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
/// - `instances[13]`: app_vm_commit (Fr)
/// - `instances[14]`: number of revealed `u64` values
/// - `instances[15..]`: fixed-capacity user public values (each u16 cell as Fr)
#[cfg(feature = "evm-prove")]
impl TryFrom<openvm_static_verifier::keygen::RawEvmProof> for EvmProof {
    type Error = EvmProofConversionError;

    fn try_from(raw: openvm_static_verifier::keygen::RawEvmProof) -> Result<Self, Self::Error> {
        let openvm_static_verifier::keygen::RawEvmProof { instances, proof } = raw;
        let minimum_instances = USER_PUBLIC_VALUES_OFFSET + U16_CELLS_PER_PUBLIC_VALUE;
        if instances.len() < minimum_instances {
            return Err(EvmProofConversionError::InvalidInstanceCount {
                expected: minimum_instances,
                actual: instances.len(),
            });
        }
        let num_cells = instances.len() - USER_PUBLIC_VALUES_OFFSET;
        validate_public_value_cell_count(num_cells)?;

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

        let count_bytes = instances[USER_PUBLIC_VALUES_COUNT_OFFSET].to_bytes();
        if count_bytes[size_of::<u32>()..]
            .iter()
            .any(|&byte| byte != 0)
        {
            return Err(EvmProofConversionError::NonCanonicalPublicValuesCount);
        }
        let num_public_values =
            u32::from_le_bytes(count_bytes[..size_of::<u32>()].try_into().unwrap());

        let user_public_values = instances[USER_PUBLIC_VALUES_OFFSET..]
            .iter()
            .enumerate()
            .map(|(index, f)| {
                let bytes = f.to_bytes();
                if bytes[U16_CELL_SIZE..].iter().any(|&byte| byte != 0) {
                    return Err(EvmProofConversionError::NonCanonicalPublicValueCell(index));
                }
                Ok([bytes[0], bytes[1]])
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let app_commit = AppExecutionCommit {
            app_exe_commit: CommitBytes::new(app_exe_bytes),
            app_vm_commit: CommitBytes::new(app_vm_bytes),
        };

        let proof = Self {
            version: format!("v{OPENVM_VERSION}"),
            app_commit,
            num_public_values,
            user_public_values,
            proof_data: ProofData {
                accumulator: evm_accumulator,
                proof,
            },
        };
        proof.validate()?;
        Ok(proof)
    }
}

/// Convert `EvmProof` → `RawEvmProof`.
#[cfg(feature = "evm-prove")]
impl TryFrom<EvmProof> for openvm_static_verifier::keygen::RawEvmProof {
    type Error = EvmProofConversionError;

    fn try_from(evm_proof: EvmProof) -> Result<Self, Self::Error> {
        evm_proof.validate()?;
        let EvmProof {
            app_commit,
            num_public_values,
            user_public_values,
            proof_data,
            version: _,
        } = evm_proof;

        let ProofData { accumulator, proof } = proof_data;

        let app_exe_fr = decode_bn254_be(
            app_commit.app_exe_commit.as_slice(),
            "app executable commitment",
        )?;
        let app_vm_fr = decode_bn254_be(app_commit.app_vm_commit.as_slice(), "app VM commitment")?;
        let user_pvs_frs: Vec<Fr> = user_public_values
            .chunks_exact(U16_CELL_SIZE)
            .map(|limb| {
                let mut bytes = [0u8; 32];
                bytes[..U16_CELL_SIZE].copy_from_slice(limb);
                Fr::from_bytes(&bytes).unwrap()
            })
            .collect();

        // Reconstruct instances: accumulator + commits + count + u16 cells.
        let mut instances = Vec::with_capacity(USER_PUBLIC_VALUES_OFFSET + user_pvs_frs.len());
        for chunk in accumulator.chunks_exact(BN254_BYTES) {
            instances.push(decode_bn254_be(chunk, "KZG accumulator")?);
        }
        instances.push(app_exe_fr);
        instances.push(app_vm_fr);
        instances.push(Fr::from(u64::from(num_public_values)));
        instances.extend(user_pvs_frs);

        Ok(openvm_static_verifier::keygen::RawEvmProof { instances, proof })
    }
}

// =================== Non-EVM types ===================

/// Struct purely for encoding and decoding of [VmStarkProof].
#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize, Encode, Decode)]
pub struct VersionedVmStarkProof {
    /// The openvm major and minor version v{}.{}. The proof format will not change on patch
    /// versions.
    pub version: String,
    #[serde_as(as = "serde_with::hex::Hex")]
    pub proof: Vec<u8>,
    #[serde_as(as = "serde_with::hex::Hex")]
    pub public_values_opening: Vec<u8>,
    #[serde(default)]
    #[serde_as(as = "Option<serde_with::hex::Hex>")]
    pub deferral_merkle_proofs: Option<Vec<u8>>,
}

impl VersionedVmStarkProof {
    pub fn new(proof: VmStarkProof) -> Result<Self> {
        Ok(Self {
            version: format!("v{}", OPENVM_VERSION),
            proof: proof.inner.encode_to_vec()?,
            public_values_opening: {
                let mut buf = Vec::new();
                proof
                    .public_values_opening
                    .encode::<crate::SC, _>(&mut buf)?;
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

#[cfg(all(test, feature = "evm-prove"))]
mod tests {
    use halo2_base::halo2_proofs::arithmetic::Field;
    use openvm_static_verifier::{keygen::RawEvmProof, Fr};

    use super::{
        EvmProof, EvmProofConversionError, BN254_BYTES, NUM_BN254_ACCUMULATOR, NUM_BN254_PROOF,
        U16_CELLS_PER_PUBLIC_VALUE, U16_CELL_SIZE,
    };

    fn fr_from_u16(value: u16) -> Fr {
        let mut bytes = [0u8; 32];
        bytes[..U16_CELL_SIZE].copy_from_slice(&value.to_le_bytes());
        Fr::from_bytes(&bytes).unwrap()
    }

    fn raw_evm_proof(count: u32, cells: impl IntoIterator<Item = u16>) -> RawEvmProof {
        let mut instances = vec![Fr::ZERO; NUM_BN254_ACCUMULATOR + 2];
        instances.push(Fr::from(u64::from(count)));
        instances.extend(cells.into_iter().map(fr_from_u16));
        RawEvmProof {
            instances,
            proof: vec![1; NUM_BN254_PROOF * BN254_BYTES],
        }
    }

    #[test]
    fn evm_proof_roundtrips_u16_public_values() {
        let raw = raw_evm_proof(1, [0x1234, 0xabcd, 0x5678, 0x9abc, 0, 0, 0, 0]);

        let proof = EvmProof::try_from(raw.clone()).unwrap();
        assert_eq!(proof.num_public_values, 1);
        assert_eq!(
            proof.user_public_values,
            [0x34, 0x12, 0xcd, 0xab, 0x78, 0x56, 0xbc, 0x9a, 0, 0, 0, 0, 0, 0, 0, 0,]
        );

        let roundtrip = RawEvmProof::try_from(proof).unwrap();
        assert_eq!(roundtrip.instances, raw.instances);
        assert_eq!(roundtrip.proof, raw.proof);
    }

    #[test]
    fn evm_proof_rejects_out_of_range_count() {
        let raw = raw_evm_proof(3, [0; 2 * U16_CELLS_PER_PUBLIC_VALUE]);
        assert_eq!(
            EvmProof::try_from(raw).unwrap_err(),
            EvmProofConversionError::PublicValuesCountOutOfRange {
                count: 3,
                capacity: 2,
            }
        );
    }

    #[test]
    fn evm_proof_rejects_noncanonical_count() {
        let mut raw = raw_evm_proof(0, [0; U16_CELLS_PER_PUBLIC_VALUE]);
        raw.instances[NUM_BN254_ACCUMULATOR + 2] = Fr::from(1u64 << 32);
        assert_eq!(
            EvmProof::try_from(raw).unwrap_err(),
            EvmProofConversionError::NonCanonicalPublicValuesCount
        );
    }

    #[test]
    fn evm_proof_rejects_noncanonical_cell() {
        let mut raw = raw_evm_proof(1, [0; U16_CELLS_PER_PUBLIC_VALUE]);
        raw.instances[NUM_BN254_ACCUMULATOR + 3] = Fr::from(u64::from(u16::MAX) + 1);
        assert_eq!(
            EvmProof::try_from(raw).unwrap_err(),
            EvmProofConversionError::NonCanonicalPublicValueCell(0)
        );
    }

    #[test]
    fn evm_proof_rejects_nonzero_padding() {
        let raw = raw_evm_proof(1, [0, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(
            EvmProof::try_from(raw).unwrap_err(),
            EvmProofConversionError::NonzeroPublicValuePadding(4)
        );
    }

    #[test]
    fn evm_proof_rejects_invalid_proof_length() {
        let mut raw = raw_evm_proof(0, [0; U16_CELLS_PER_PUBLIC_VALUE]);
        raw.proof.pop();
        assert_eq!(
            EvmProof::try_from(raw).unwrap_err(),
            EvmProofConversionError::InvalidProofLength {
                expected: NUM_BN254_PROOF * BN254_BYTES,
                actual: NUM_BN254_PROOF * BN254_BYTES - 1,
            }
        );
    }
}

impl TryFrom<VersionedVmStarkProof> for VmStarkProof {
    type Error = std::io::Error;
    fn try_from(proof: VersionedVmStarkProof) -> Result<Self, std::io::Error> {
        let VersionedVmStarkProof {
            proof,
            public_values_opening,
            deferral_merkle_proofs,
            ..
        } = proof;
        Ok(Self {
            inner: Proof::<crate::SC>::decode_from_bytes(&proof)?,
            public_values_opening: PublicValuesOpening::decode::<crate::SC, _>(
                &mut std::io::Cursor::new(&public_values_opening),
            )?,
            deferral_merkle_proofs: deferral_merkle_proofs
                .map(|bytes| DeferralMerkleProofs::decode(&mut std::io::Cursor::new(&bytes)))
                .transpose()?,
        })
    }
}

// =================== Verification baseline JSON types ===================

/// Hex-formatted [`VkCommit`] for JSON serialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VkCommitJson {
    pub cached_commit: CommitBytes,
    pub vk_pre_hash: CommitBytes,
}

/// Hex-formatted [`VerificationBaseline`] for JSON serialization.
///
/// Mirrors [`VerificationBaseline`] but serializes all commit fields as `0x`-prefixed hex strings,
/// consistent with [`AppExecutionCommit`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationBaselineJson {
    pub app_exe_commit: CommitBytes,
    pub memory_dimensions: MemoryDimensions,
    pub num_user_pvs: usize,
    pub app_vk_commit: VkCommitJson,
    pub leaf_vk_commit: VkCommitJson,
    pub internal_for_leaf_vk_commit: VkCommitJson,
    pub internal_recursive_vk_commit: VkCommitJson,
    pub expected_def_hook_commit: Option<CommitBytes>,
}

impl From<VerificationBaseline> for VerificationBaselineJson {
    fn from(b: VerificationBaseline) -> Self {
        let vk = |d: VkCommit<crate::F>| VkCommitJson {
            cached_commit: CommitBytes::from(d.cached_commit),
            vk_pre_hash: CommitBytes::from(d.vk_pre_hash),
        };
        Self {
            app_exe_commit: CommitBytes::from(b.app_exe_commit),
            memory_dimensions: b.memory_dimensions,
            num_user_pvs: b.num_user_pvs,
            app_vk_commit: vk(b.app_vk_commit),
            leaf_vk_commit: vk(b.leaf_vk_commit),
            internal_for_leaf_vk_commit: vk(b.internal_for_leaf_vk_commit),
            internal_recursive_vk_commit: vk(b.internal_recursive_vk_commit),
            expected_def_hook_commit: b.expected_def_hook_commit.map(CommitBytes::from),
        }
    }
}

impl From<VerificationBaselineJson> for VerificationBaseline {
    fn from(b: VerificationBaselineJson) -> Self {
        use openvm_verify_stark_host::pvs::VkCommit;
        let vk = |d: VkCommitJson| VkCommit {
            cached_commit: d.cached_commit.into(),
            vk_pre_hash: d.vk_pre_hash.into(),
        };
        Self {
            app_exe_commit: b.app_exe_commit.into(),
            memory_dimensions: b.memory_dimensions,
            num_user_pvs: b.num_user_pvs,
            app_vk_commit: vk(b.app_vk_commit),
            leaf_vk_commit: vk(b.leaf_vk_commit),
            internal_for_leaf_vk_commit: vk(b.internal_for_leaf_vk_commit),
            internal_recursive_vk_commit: vk(b.internal_recursive_vk_commit),
            expected_def_hook_commit: b.expected_def_hook_commit.map(|c| c.into()),
        }
    }
}
