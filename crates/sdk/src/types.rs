use itertools::Itertools;
use openvm_native_recursion::halo2::{Fr, RawEvmProof};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

/// Number of bytes in a Bn254Fr.
const BN254_BYTES: usize = 32;
/// Number of Bn254Fr in `accumulators` field.
const NUM_BN254_ACCUMULATORS: usize = 12;
/// Number of Bn254Fr in `proof` field for a circuit with only 1 advice column.
const NUM_BN254_PROOF: usize = 43;

#[serde_as]
#[derive(Clone, Deserialize, Serialize)]
pub struct EvmProof {
    #[serde_as(as = "serde_with::hex::Hex")]
    /// Bn254Fr public values for accumulators in flatten little-endian bytes. Length is `NUM_BN254_ACCUMULATORS * BN254_BYTES`.
    pub accumulators: Vec<u8>,
    #[serde_as(as = "serde_with::hex::Hex")]
    /// 1 Bn254Fr public value for exe commit in little-endian bytes.
    pub exe_commit: [u8; BN254_BYTES],
    #[serde_as(as = "serde_with::hex::Hex")]
    /// 1 Bn254Fr public value for leaf commit in little-endian bytes.
    pub leaf_commit: [u8; BN254_BYTES],
    #[serde_as(as = "serde_with::hex::Hex")]
    /// Bn254Fr user public values in little-endian bytes.
    pub user_public_values: Vec<u8>,
    #[serde_as(as = "serde_with::hex::Hex")]
    /// Bn254Fr proof in little-endian bytes. The circuit only has 1 advice column, so the proof is of length `NUM_BN254_PROOF * BN254_BYTES`.
    pub proof: Vec<u8>,
}

impl EvmProof {
    /// Return bytes calldata to be passed to the verifier contract.
    pub fn verifier_calldata(&self) -> Vec<u8> {
        let evm_proof: RawEvmProof = self.clone().into();
        evm_proof.verifier_calldata()
    }
}

impl From<RawEvmProof> for EvmProof {
    fn from(evm_proof: RawEvmProof) -> Self {
        let RawEvmProof { instances, proof } = evm_proof;
        assert!(NUM_BN254_ACCUMULATORS + 2 < instances.len());
        assert_eq!(proof.len(), NUM_BN254_PROOF * BN254_BYTES);
        let accumulators = instances[0..NUM_BN254_ACCUMULATORS]
            .iter()
            .flat_map(|f| f.to_bytes())
            .collect::<Vec<_>>();
        let exe_commit = instances[NUM_BN254_ACCUMULATORS].to_bytes();
        let leaf_commit = instances[NUM_BN254_ACCUMULATORS + 1].to_bytes();
        let user_public_values = instances[NUM_BN254_ACCUMULATORS + 2..]
            .iter()
            .flat_map(|f| f.to_bytes())
            .collect::<Vec<_>>();
        Self {
            accumulators,
            exe_commit,
            leaf_commit,
            user_public_values,
            proof,
        }
    }
}

impl From<EvmProof> for RawEvmProof {
    fn from(evm_openvm_proof: EvmProof) -> Self {
        let EvmProof {
            accumulators,
            exe_commit,
            leaf_commit,
            user_public_values,
            proof,
        } = evm_openvm_proof;
        assert_eq!(proof.len(), NUM_BN254_PROOF * BN254_BYTES);
        let instances = {
            assert_eq!(accumulators.len(), NUM_BN254_ACCUMULATORS * BN254_BYTES);
            assert_eq!(user_public_values.len() % BN254_BYTES, 0);
            assert!(!user_public_values.is_empty());
            let mut ret = Vec::new();
            for chunk in &accumulators
                .iter()
                .chain(&exe_commit)
                .chain(&leaf_commit)
                .chain(&user_public_values)
                .chunks(BN254_BYTES)
            {
                let c = chunk.copied().collect::<Vec<_>>().try_into().unwrap();
                ret.push(Fr::from_bytes(&c).unwrap());
            }
            ret
        };
        RawEvmProof { instances, proof }
    }
}
