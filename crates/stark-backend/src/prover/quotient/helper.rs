use crate::{
    config::StarkGenericConfig,
    interaction::HasInteractionChunkSize,
    keygen::types::{MultiStarkProvingKey, StarkProvingKey},
    prover::quotient::QuotientVkData,
};

pub(crate) trait QuotientVkDataHelper<SC: StarkGenericConfig> {
    fn get_quotient_vk_data(&self) -> QuotientVkData<SC>;
}

impl<SC: StarkGenericConfig> QuotientVkDataHelper<SC> for StarkProvingKey<SC> {
    fn get_quotient_vk_data(&self) -> QuotientVkData<SC> {
        QuotientVkData {
            quotient_degree: self.vk.quotient_degree,
            rap_phase_seq_kind: self.vk.rap_phase_seq_kind,
            interaction_chunk_size: self.rap_phase_seq_pk.interaction_chunk_size(),
            symbolic_constraints: &self.vk.symbolic_constraints,
        }
    }
}

impl<SC: StarkGenericConfig> MultiStarkProvingKey<SC> {
    pub fn get_quotient_vk_data_per_air(&self) -> Vec<QuotientVkData<SC>> {
        self.per_air
            .iter()
            .map(|pk| pk.get_quotient_vk_data())
            .collect()
    }
}
