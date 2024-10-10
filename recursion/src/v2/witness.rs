use afs_compiler::ir::{Builder, Usize, Witness};
use afs_stark_backend::prover::v2::types::{AirProofData, ProofV2};
use ax_sdk::config::baby_bear_poseidon2_outer::BabyBearPoseidon2OuterConfig;
use p3_util::log2_strict_usize;

use crate::{
    config::outer::OuterConfig,
    v2::vars::{AirProofDataVariable, StarkProofV2Variable},
    witness::{VectorWitnessable, Witnessable},
};

type C = OuterConfig;

impl VectorWitnessable<C> for AirProofData<BabyBearPoseidon2OuterConfig> {}

impl Witnessable<C> for ProofV2<BabyBearPoseidon2OuterConfig> {
    type WitnessVariable = StarkProofV2Variable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let commitments = self.commitments.read(builder);
        let opening = self.opening.read(builder);
        let per_air = self.per_air.read(builder);
        // This reads nothing because air_perm_by_height is a constant.
        let air_perm_by_height = builder.array(0);

        StarkProofV2Variable {
            commitments,
            opening,
            per_air,
            air_perm_by_height,
        }
    }

    fn write(&self, witness: &mut Witness<C>) {
        self.commitments.write(witness);
        self.opening.write(witness);
        self.per_air.write(witness);
        // air_perm_by_height is a constant so we write nothing.
    }
}

impl Witnessable<C> for AirProofData<BabyBearPoseidon2OuterConfig> {
    type WitnessVariable = AirProofDataVariable<C>;
    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        // air_id is constant, skip
        let air_id = Usize::from(0);
        // log_degree is constant, skip
        let log_degree = Usize::from(log2_strict_usize(self.degree));
        let exposed_values_after_challenge = self.exposed_values_after_challenge.read(builder);
        let public_values = self.public_values.read(builder);
        Self::WitnessVariable {
            air_id,
            log_degree,
            exposed_values_after_challenge,
            public_values,
        }
    }
    fn write(&self, witness: &mut Witness<C>) {
        // air_id is constant, skip
        // log_degree is constant, skip
        <_ as Witnessable<_>>::write(&self.exposed_values_after_challenge, witness);
        <_ as Witnessable<_>>::write(&self.public_values, witness);
    }
}
