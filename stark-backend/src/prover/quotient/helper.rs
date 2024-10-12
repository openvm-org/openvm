use p3_uni_stark::StarkGenericConfig;

use crate::{
    keygen::types::{MultiStarkProvingKey, StarkProvingKey},
    prover::quotient::QuotientVKData,
};

pub(crate) trait QuotientVkDataHelper<SC: StarkGenericConfig> {
    fn get_quotient_vk_data(&self) -> QuotientVKData<SC>;
}

impl<SC: StarkGenericConfig> QuotientVkDataHelper<SC> for StarkProvingKey<SC> {
    fn get_quotient_vk_data(&self) -> QuotientVKData<SC> {
        QuotientVKData {
            quotient_degree: self.vk.quotient_degree,
            interaction_chunk_size: self.interaction_chunk_size,
            symbolic_constraints: &self.vk.symbolic_constraints,
        }
    }
}

impl<SC: StarkGenericConfig> MultiStarkProvingKey<SC> {
    pub fn get_quotient_vk_data_per_air(&self) -> Vec<QuotientVKData<SC>> {
        self.per_air
            .iter()
            .map(|pk| pk.get_quotient_vk_data())
            .collect()
    }
}
