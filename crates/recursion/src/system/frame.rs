use itertools::Itertools;
use openvm_stark_backend::{
    keygen::types::{
        MultiStarkVerifyingKey, StarkVerifyingKey, StarkVerifyingParams,
        VerifierSinglePreprocessedData,
    },
    SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, Digest, F};

/*
 * Modified versions of the STARK and multi-STARK verifying keys for AirModule
 * implementations. AirModules should use MultiStarkVerifyingKeyFrame instead
 * of MultiStarkVerifyingKey<BabyBearPoseidon2Config> in their AIRs, as use of some fields in the
 * latter will compromise internal vk stability. For more information on what
 * can be used in AIRs and how, see crates/recursion/README.md.
 */

#[derive(Clone)]
pub struct StarkVkeyFrame {
    pub preprocessed_data: Option<VerifierSinglePreprocessedData<Digest>>,
    pub params: StarkVerifyingParams,
    pub num_interactions: usize,
    pub max_constraint_degree: u8,
    pub is_required: bool,
}

#[derive(Clone)]
pub struct MultiStarkVkeyFrame {
    pub params: SystemParams,
    pub per_air: Vec<StarkVkeyFrame>,
    pub max_constraint_degree: usize,
}

impl From<&StarkVerifyingKey<F, Digest>> for StarkVkeyFrame {
    fn from(vk: &StarkVerifyingKey<F, Digest>) -> Self {
        Self {
            preprocessed_data: vk.preprocessed_data.clone(),
            params: vk.params.clone(),
            num_interactions: vk.num_interactions(),
            max_constraint_degree: vk.max_constraint_degree,
            is_required: vk.is_required,
        }
    }
}

impl From<&MultiStarkVerifyingKey<BabyBearPoseidon2Config>> for MultiStarkVkeyFrame {
    fn from(mvk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>) -> Self {
        Self {
            params: mvk.inner.params.clone(),
            per_air: mvk.inner.per_air.iter().map(Into::into).collect_vec(),
            max_constraint_degree: mvk.max_constraint_degree(),
        }
    }
}
