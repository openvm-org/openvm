use openvm_stark_backend::p3_field::{Field, PrimeField32};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use p3_poseidon2::ExternalLayerConstants;
use p3_poseidon2_air::RoundConstants;

use super::{
    BABYBEAR_BEGIN_EXT_CONSTS, BABYBEAR_END_EXT_CONSTS, BABYBEAR_PARTIAL_CONSTS,
    POSEIDON2_HALF_FULL_ROUNDS, POSEIDON2_PARTIAL_ROUNDS, POSEIDON2_WIDTH,
};

#[derive(Clone, Copy, Debug)]
pub struct Poseidon2Config<F> {
    pub matrix: Poseidon2Matrix,
    pub constants: Poseidon2Constants<F>,
}

impl<F: PrimeField32> Default for Poseidon2Config<F> {
    fn default() -> Self {
        Self {
            matrix: Poseidon2Matrix::MdsMatrix,
            constants: Default::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Poseidon2Matrix {
    MdsMatrix,
    HlMdsMatrix,
}

#[derive(Clone, Copy, Debug)]
pub struct Poseidon2Constants<F> {
    pub beginning_full_round_constants: [[F; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS],
    pub partial_round_constants: [F; POSEIDON2_PARTIAL_ROUNDS],
    pub ending_full_round_constants: [[F; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS],
}

impl<F: Field> Poseidon2Constants<F> {
    // FIXME[stephenh]: Right now RoundConstants is not very easy to use. I've put up a Plonky3 PR to
    // add a public constructor to it and derive Clone. Once this goes through, we should not do this
    // unsafely.
    pub fn to_round_constants(&self) -> Plonky3RoundConstants<F> {
        unsafe { std::mem::transmute_copy(self) }
    }

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

impl<F: PrimeField32> Default for Poseidon2Constants<F> {
    fn default() -> Self {
        let convert_field = |f: BabyBear| F::from_canonical_u32(f.as_canonical_u32());
        Self {
            beginning_full_round_constants: BABYBEAR_BEGIN_EXT_CONSTS.map(|x| x.map(convert_field)),
            partial_round_constants: BABYBEAR_PARTIAL_CONSTS.map(convert_field),
            ending_full_round_constants: BABYBEAR_END_EXT_CONSTS.map(|x| x.map(convert_field)),
        }
    }
}

pub type Plonky3RoundConstants<F> =
    RoundConstants<F, POSEIDON2_WIDTH, POSEIDON2_HALF_FULL_ROUNDS, POSEIDON2_PARTIAL_ROUNDS>;
