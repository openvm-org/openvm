use itertools::Itertools;
use openvm_native_recursion::halo2::{EvmProof, Fr};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

const BN254_BYTES: usize = 32;
const NUM_BN254_ACCUMULATORS: usize = 12;
const NUM_BN254_PROOF: usize = 43;

#[derive(Clone, Deserialize, Serialize)]
pub struct EvmOpenvmProof {
    #[serde(with = "BigArray")]
    pub accumulators: [u8; BN254_BYTES * NUM_BN254_ACCUMULATORS],
    pub exe_commit: [u8; BN254_BYTES],
    pub leaf_commit: [u8; BN254_BYTES],
    pub user_public_values: Vec<u8>,
    #[serde(with = "BigArray")]
    pub proof: [u8; BN254_BYTES * NUM_BN254_PROOF],
}

impl EvmOpenvmProof {
    /// Return bytes calldata to be passed to the verifier contract.
    pub fn verifier_calldata(&self) -> Vec<u8> {
        let evm_proof: EvmProof = self.clone().into();
        evm_proof.verifier_calldata()
    }
}

impl From<EvmProof> for EvmOpenvmProof {
    fn from(evm_proof: EvmProof) -> Self {
        let EvmProof { instances, proof } = evm_proof;
        assert_eq!(instances.len(), 1);
        assert!(NUM_BN254_ACCUMULATORS + 2 < instances[0].len());
        let accumulators = instances[0][0..NUM_BN254_ACCUMULATORS]
            .iter()
            .flat_map(|f| f.to_bytes())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let exe_commit = instances[0][NUM_BN254_ACCUMULATORS].to_bytes();
        let leaf_commit = instances[0][NUM_BN254_ACCUMULATORS + 1].to_bytes();
        let user_public_values = instances[0][NUM_BN254_ACCUMULATORS + 2..]
            .iter()
            .flat_map(|f| f.to_bytes())
            .collect::<Vec<_>>();
        Self {
            accumulators,
            exe_commit,
            leaf_commit,
            user_public_values,
            proof: proof.try_into().unwrap(),
        }
    }
}

impl From<EvmOpenvmProof> for EvmProof {
    fn from(evm_openvm_proof: EvmOpenvmProof) -> Self {
        let EvmOpenvmProof {
            accumulators,
            exe_commit,
            leaf_commit,
            user_public_values,
            proof,
        } = evm_openvm_proof;
        let instances = {
            assert_eq!(user_public_values.len() % BN254_BYTES, 0);
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
            vec![ret]
        };
        EvmProof {
            instances,
            proof: proof.to_vec(),
        }
    }
}
