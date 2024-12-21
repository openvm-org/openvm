mod air;
mod babybear;
mod config;
mod permute;

#[cfg(test)]
mod tests;

use std::sync::Arc;

pub use air::*;
pub use babybear::*;
pub use config::*;
use openvm_stark_backend::{
    p3_field::{Field, PrimeField},
    p3_matrix::dense::RowMajorMatrix,
};
use p3_poseidon2::Poseidon2;
use p3_poseidon2_air::generate_trace_rows;
pub use p3_poseidon2_air::Poseidon2Air;
pub use p3_symmetric::Permutation;
pub use permute::*;

pub const POSEIDON2_WIDTH: usize = 16;
pub const POSEIDON2_HALF_FULL_ROUNDS: usize = 4;
pub const POSEIDON2_FULL_ROUNDS: usize = 8;
pub const POSEIDON2_PARTIAL_ROUNDS: usize = 13;

// Currently we only support SBOX_DEGREE = 7
pub const POSEIDON2_SBOX_DEGREE: u64 = 7;

#[derive(Debug)]
pub struct Poseidon2SubChip<F: Field, const SBOX_REGISTERS: usize> {
    pub air: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
    pub(crate) executor: Poseidon2Executor<F>,
    pub(crate) constants: Plonky3RoundConstants<F>,
}

impl<F: PrimeField, const SBOX_REGISTERS: usize> Poseidon2SubChip<F, SBOX_REGISTERS> {
    pub fn new(config: Poseidon2Config<F>) -> Self {
        Self {
            air: Arc::new(Poseidon2SubAir::new(
                config.matrix,
                config.constants.to_round_constants(),
            )),
            executor: Poseidon2Executor::new(config.matrix, config.constants),
            constants: config.constants.to_round_constants(),
        }
    }

    pub fn permute(&self, input_state: [F; POSEIDON2_WIDTH]) -> [F; POSEIDON2_WIDTH] {
        match &self.executor {
            Poseidon2Executor::BabyBearMds(permuter) => permuter.permute(input_state),
            Poseidon2Executor::BabyBearHlMds(permuter) => permuter.permute(input_state),
        }
    }

    pub fn permute_mut(&self, input_state: &mut [F; POSEIDON2_WIDTH]) {
        match &self.executor {
            Poseidon2Executor::BabyBearMds(permuter) => permuter.permute_mut(input_state),
            Poseidon2Executor::BabyBearHlMds(permuter) => permuter.permute_mut(input_state),
        };
    }

    pub fn generate_trace(&self, inputs: Vec<[F; POSEIDON2_WIDTH]>) -> RowMajorMatrix<F>
    where
        F: PrimeField,
    {
        match self.air.as_ref() {
            Poseidon2SubAir::BabyBearMds(_) => generate_trace_rows::<
                F,
                Poseidon2LinearLayers<MdsMatrixConfig>,
                POSEIDON2_WIDTH,
                POSEIDON2_SBOX_DEGREE,
                SBOX_REGISTERS,
                POSEIDON2_HALF_FULL_ROUNDS,
                POSEIDON2_PARTIAL_ROUNDS,
            >(inputs, &self.constants),
            Poseidon2SubAir::BabyBearHlMds(_) => generate_trace_rows::<
                F,
                Poseidon2LinearLayers<HlMdsMatrixConfig>,
                POSEIDON2_WIDTH,
                POSEIDON2_SBOX_DEGREE,
                SBOX_REGISTERS,
                POSEIDON2_HALF_FULL_ROUNDS,
                POSEIDON2_PARTIAL_ROUNDS,
            >(inputs, &self.constants),
        }
    }
}

pub fn from_config<F: PrimeField, const SBOX_REGISTERS: usize>(
    config: Poseidon2Config<F>,
) -> (Poseidon2SubAir<F, SBOX_REGISTERS>, Poseidon2Executor<F>) {
    (
        Poseidon2SubAir::new(config.matrix, config.constants.to_round_constants()),
        Poseidon2Executor::new(config.matrix, config.constants),
    )
}

#[derive(Debug)]
pub enum Poseidon2Executor<F: Field> {
    BabyBearMds(Plonky3Poseidon2Executor<F, MdsMatrixConfig>),
    BabyBearHlMds(Plonky3Poseidon2Executor<F, HlMdsMatrixConfig>),
}

impl<F: PrimeField> Poseidon2Executor<F> {
    pub fn new(matrix: Poseidon2Matrix, constants: Poseidon2Constants<F>) -> Self {
        let (external_constants, internal_constants) = constants.to_external_internal_constants();
        match matrix {
            Poseidon2Matrix::MdsMatrix => Self::BabyBearMds(Plonky3Poseidon2Executor::new(
                external_constants,
                internal_constants,
            )),
            Poseidon2Matrix::HlMdsMatrix => Self::BabyBearHlMds(Plonky3Poseidon2Executor::new(
                external_constants,
                internal_constants,
            )),
        }
    }
}

pub type Plonky3Poseidon2Executor<F, Config> = Poseidon2<
    <F as Field>::Packing,
    Poseidon2ExternalLayer<F, Config, POSEIDON2_WIDTH>,
    Poseidon2InternalLayer<F, Config>,
    POSEIDON2_WIDTH,
    POSEIDON2_SBOX_DEGREE,
>;
