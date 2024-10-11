use std::cmp::Reverse;

use afs_compiler::ir::{Array, Builder, Config, Usize};
use afs_stark_backend::prover::{
    opener::OpeningProof,
    types::Commitments,
    v2::types::{AirProofData, ProofV2},
};
use ax_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use itertools::Itertools;
use p3_util::log2_strict_usize;

use crate::{
    hints::{Hintable, InnerChallenge, InnerVal, VecAutoHintable},
    types::InnerConfig,
    v2::vars::{AirProofDataVariable, StarkProofV2Variable},
};

impl VecAutoHintable for AirProofData<BabyBearPoseidon2Config> {}

impl Hintable<InnerConfig> for ProofV2<BabyBearPoseidon2Config> {
    type HintVariable = StarkProofV2Variable<InnerConfig>;

    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let commitments = Commitments::<BabyBearPoseidon2Config>::read(builder);
        let opening = OpeningProof::<BabyBearPoseidon2Config>::read(builder);
        let per_air = Vec::<AirProofData<BabyBearPoseidon2Config>>::read(builder);
        let raw_air_perm_by_height = Vec::<usize>::read(builder);
        // A hacky way to transmute from Array of Var to Array of Usize.
        let air_perm_by_height = if let Array::Dyn(ptr, len) = raw_air_perm_by_height {
            Array::Dyn(ptr, len)
        } else {
            unreachable!();
        };

        StarkProofV2Variable {
            commitments,
            opening,
            per_air,
            air_perm_by_height,
        }
    }

    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        let mut stream = Vec::new();

        stream.extend(self.commitments.write());
        stream.extend(self.opening.write());
        stream.extend(<Vec<AirProofData<_>> as Hintable<_>>::write(&self.per_air));
        let air_perm_by_height: Vec<_> = (0..self.per_air.len())
            .sorted_by_key(|i| Reverse(self.per_air[*i].degree))
            .collect();
        stream.extend(air_perm_by_height.write());

        stream
    }
}

impl Hintable<InnerConfig> for AirProofData<BabyBearPoseidon2Config> {
    type HintVariable = AirProofDataVariable<InnerConfig>;
    fn read(builder: &mut Builder<InnerConfig>) -> Self::HintVariable {
        let air_id = Usize::Var(usize::read(builder));
        let log_degree = Usize::Var(usize::read(builder));
        let exposed_values_after_challenge = Vec::<Vec<InnerChallenge>>::read(builder);
        let public_values = Vec::<InnerVal>::read(builder);
        Self::HintVariable {
            air_id,
            log_degree,
            exposed_values_after_challenge,
            public_values,
        }
    }
    fn write(&self) -> Vec<Vec<<InnerConfig as Config>::N>> {
        let mut stream = Vec::new();

        stream.extend(<usize as Hintable<InnerConfig>>::write(&self.air_id));
        stream.extend(<usize as Hintable<InnerConfig>>::write(&log2_strict_usize(
            self.degree,
        )));
        stream.extend(self.exposed_values_after_challenge.write());
        stream.extend(self.public_values.write());

        stream
    }
}
