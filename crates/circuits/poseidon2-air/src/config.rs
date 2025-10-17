use openvm_stark_backend::p3_field::{Field, PrimeField32};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use p3_poseidon2::ExternalLayerConstants;
use p3_poseidon2_air::RoundConstants;

use super::{
    BABYBEAR_BEGIN_EXT_CONSTS, BABYBEAR_END_EXT_CONSTS, BABYBEAR_PARTIAL_CONSTS,
    BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS, BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS, POSEIDON2_WIDTH,
};

// Currently only contains round constants, but this struct may contain other configuration
// parameters in the future.
#[derive(Clone, Copy, Debug)]
pub struct Poseidon2Config<F> {
    pub constants: Poseidon2Constants<F>,
}

impl<F: Field> Default for Poseidon2Config<F> {
    fn default() -> Self {
        Self {
            constants: default_baby_bear_rc(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Poseidon2Constants<F> {
    pub beginning_full_round_constants:
        [[F; POSEIDON2_WIDTH]; BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS],
    pub partial_round_constants: [F; BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS],
    pub ending_full_round_constants: [[F; POSEIDON2_WIDTH]; BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS],
}

impl<F: Field> From<Poseidon2Constants<F>> for Plonky3RoundConstants<F> {
    fn from(constants: Poseidon2Constants<F>) -> Self {
        Plonky3RoundConstants::new(
            constants.beginning_full_round_constants,
            constants.partial_round_constants,
            constants.ending_full_round_constants,
        )
    }
}

impl<F: Field> Poseidon2Constants<F> {
    pub fn to_external_internal_constants(
        &self,
    ) -> (ExternalLayerConstants<F, POSEIDON2_WIDTH>, Vec<F>) {
        (
            ExternalLayerConstants::new(
                self.beginning_full_round_constants.to_vec(),
                self.ending_full_round_constants.to_vec(),
            ),
            self.partial_round_constants.to_vec(),
        )
    }
}

// Round constants for only BabyBear, but we convert to `F` due to some annoyances with generics.
// This should only be used concretely when `F = BabyBear`.
fn default_baby_bear_rc<F: Field>() -> Poseidon2Constants<F> {
    let convert_field = |f: BabyBear| F::from_canonical_u32(f.as_canonical_u32());
    Poseidon2Constants {
        beginning_full_round_constants: BABYBEAR_BEGIN_EXT_CONSTS.map(|x| x.map(convert_field)),
        partial_round_constants: BABYBEAR_PARTIAL_CONSTS.map(convert_field),
        ending_full_round_constants: BABYBEAR_END_EXT_CONSTS.map(|x| x.map(convert_field)),
    }
}

pub type Plonky3RoundConstants<F> = RoundConstants<
    F,
    POSEIDON2_WIDTH,
    BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS,
    BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
>;
