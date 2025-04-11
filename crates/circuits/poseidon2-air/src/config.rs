use openvm_stark_backend::p3_field::Field;
use p3_poseidon2::ExternalLayerConstants;
use p3_poseidon2_air::RoundConstants;

use super::{POSEIDON2_HALF_FULL_ROUNDS, POSEIDON2_WIDTH};

// Currently only contains round constants, but this struct may contain other configuration parameters in the future.
#[derive(Clone, Debug)]
pub struct Poseidon2Config<F: Field> {
    pub constants: Poseidon2Constants<F>,
}

impl<F: Field> Poseidon2Config<F> {
    pub fn new(constants: Poseidon2Constants<F>) -> Self {
        Self { constants }
    }
}

#[derive(Clone, Debug)]
pub struct Poseidon2Constants<F: Field> {
    pub beginning_full_round_constants: [[F; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS],
    pub partial_round_constants: Vec<F>,
    pub ending_full_round_constants: [[F; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS],
}

impl<F: Field, const PARTIAL_ROUNDS: usize> From<Poseidon2Constants<F>>
    for Plonky3RoundConstants<F, PARTIAL_ROUNDS>
{
    fn from(constants: Poseidon2Constants<F>) -> Self {
        let partial_round_constants = constants.partial_round_constants.try_into().unwrap();
        Plonky3RoundConstants::new(
            constants.beginning_full_round_constants,
            partial_round_constants,
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
            self.partial_round_constants.clone(),
        )
    }
}

pub type Plonky3RoundConstants<F, const PARTIAL_ROUNDS: usize> =
    RoundConstants<F, POSEIDON2_WIDTH, POSEIDON2_HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;
