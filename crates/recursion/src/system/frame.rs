use itertools::Itertools;
use stark_backend_v2::{
    Digest, F, SystemParams,
    keygen::types::{
        MultiStarkVerifyingKeyV2, StarkVerifyingKeyV2, StarkVerifyingParamsV2,
        VerifierSinglePreprocessedData,
    },
};

/*
 * Modified versions of the STARK and multi-STARK verifying keys for AirModule
 * implementations. AirModules should use MultiStarkVerifyingKeyFrame instead
 * of MultiStarkVerifyingKeyV2 in their AIRs, as use of some fields in the
 * latter will compromise internal vk stability. For more information on what
 * can be used in AIRs and how, see crates/recursion/README.md.
 */

#[derive(Clone)]
pub struct StarkVkeyFrame {
    pub preprocessed_data: Option<VerifierSinglePreprocessedData<Digest>>,
    pub params: StarkVerifyingParamsV2,
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

impl From<&StarkVerifyingKeyV2<F, Digest>> for StarkVkeyFrame {
    fn from(vk: &StarkVerifyingKeyV2<F, Digest>) -> Self {
        Self {
            preprocessed_data: vk.preprocessed_data.clone(),
            params: vk.params.clone(),
            num_interactions: vk.num_interactions(),
            max_constraint_degree: vk.max_constraint_degree,
            is_required: vk.is_required,
        }
    }
}

impl From<&MultiStarkVerifyingKeyV2> for MultiStarkVkeyFrame {
    fn from(mvk: &MultiStarkVerifyingKeyV2) -> Self {
        Self {
            params: mvk.inner.params,
            per_air: mvk.inner.per_air.iter().map(Into::into).collect_vec(),
            max_constraint_degree: mvk.max_constraint_degree(),
        }
    }
}
