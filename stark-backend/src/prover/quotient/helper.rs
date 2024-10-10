use p3_uni_stark::StarkGenericConfig;

use crate::{
    keygen::{
        types::{MultiStarkProvingKey, StarkProvingKey},
        v2::types::StarkProvingKeyV2,
    },
    prover::quotient::QuotientVKData,
};

pub(crate) trait QuotientVKDataHelper<SC: StarkGenericConfig> {
    fn get_quotient_vk_data(&self) -> QuotientVKData<SC>;
}

impl<SC: StarkGenericConfig> QuotientVKDataHelper<SC> for StarkProvingKeyV2<SC> {
    fn get_quotient_vk_data(&self) -> QuotientVKData<SC> {
        QuotientVKData {
            quotient_degree: self.vk.quotient_degree,
            interaction_chunk_size: self.interaction_chunk_size,
            symbolic_constraints: &self.vk.symbolic_constraints,
        }
    }
}

impl<SC: StarkGenericConfig> QuotientVKDataHelper<SC> for StarkProvingKey<SC> {
    fn get_quotient_vk_data(&self) -> QuotientVKData<SC> {
        QuotientVKData {
            quotient_degree: self.vk.quotient_degree,
            interaction_chunk_size: self.vk.interaction_chunk_size,
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
