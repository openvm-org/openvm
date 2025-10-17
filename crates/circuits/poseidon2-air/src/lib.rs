//! This is a wrapper around the Plonky3 [p3_poseidon2_air] used only for integration convenience to
//! get around some complications with field-specific generics associated with Poseidon2.
//! Currently it is only intended for use in OpenVM with BabyBear.
//!
//! We do not recommend external use of this crate, and suggest using the [p3_poseidon2_air] crate
//! directly.

use std::sync::Arc;

use openvm_stark_backend::{
    p3_field::{Field, PrimeField},
    p3_matrix::dense::RowMajorMatrix,
};
pub use openvm_stark_sdk::p3_baby_bear;
pub use p3_poseidon2;
use p3_poseidon2::{ExternalLayerConstants, Poseidon2};
use p3_poseidon2_air::generate_trace_rows;
pub use p3_poseidon2_air::{self, Poseidon2Air};
pub use p3_symmetric::{self, Permutation};

mod air;
mod babybear;
mod config;
mod permute;

pub use air::*;
pub use babybear::*;
pub use config::*;
pub use permute::*;

#[cfg(all(feature = "cuda", test))]
mod cuda_abi;

#[cfg(test)]
mod tests;

pub const POSEIDON2_WIDTH: usize = 16;
// NOTE: these constants are for BabyBear only.
pub const BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS: usize = 4;
pub const BABY_BEAR_POSEIDON2_FULL_ROUNDS: usize = 8;
pub const BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS: usize = 13;

// Currently we only support SBOX_DEGREE = 7
pub const BABY_BEAR_POSEIDON2_SBOX_DEGREE: u64 = 7;

/// `SBOX_REGISTERS` affects the max constraint degree of the AIR. See [p3_poseidon2_air] for more
/// details.
#[derive(Debug, Clone)]
pub struct Poseidon2SubChip<F: Field, const SBOX_REGISTERS: usize> {
    // This is Arc purely because Poseidon2Air cannot derive Clone
    pub air: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
    pub(crate) executor: Poseidon2Executor<F>,
    pub(crate) constants: Plonky3RoundConstants<F>,
}

impl<F: PrimeField, const SBOX_REGISTERS: usize> Poseidon2SubChip<F, SBOX_REGISTERS> {
    pub fn new(constants: Poseidon2Constants<F>) -> Self {
        let (external_constants, internal_constants) = constants.to_external_internal_constants();
        Self {
            air: Arc::new(Poseidon2SubAir::new(constants.into())),
            executor: Poseidon2Executor::new(external_constants, internal_constants),
            constants: constants.into(),
        }
    }

    pub fn permute(&self, input_state: [F; POSEIDON2_WIDTH]) -> [F; POSEIDON2_WIDTH] {
        match &self.executor {
            Poseidon2Executor::BabyBearMds(permuter) => permuter.permute(input_state),
        }
    }

    pub fn permute_mut(&self, input_state: &mut [F; POSEIDON2_WIDTH]) {
        match &self.executor {
            Poseidon2Executor::BabyBearMds(permuter) => permuter.permute_mut(input_state),
        };
    }

    pub fn generate_trace(&self, inputs: Vec<[F; POSEIDON2_WIDTH]>) -> RowMajorMatrix<F>
    where
        F: PrimeField,
    {
        match self.air.as_ref() {
            Poseidon2SubAir::BabyBearMds(_) => generate_trace_rows::<
                F,
                BabyBearPoseidon2LinearLayers,
                POSEIDON2_WIDTH,
                BABY_BEAR_POSEIDON2_SBOX_DEGREE,
                SBOX_REGISTERS,
                BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS,
                BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
            >(inputs, &self.constants),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Poseidon2Executor<F: Field> {
    BabyBearMds(Plonky3Poseidon2Executor<F, BabyBearPoseidon2LinearLayers>),
}

impl<F: PrimeField> Poseidon2Executor<F> {
    pub fn new(
        external_constants: ExternalLayerConstants<F, POSEIDON2_WIDTH>,
        internal_constants: Vec<F>,
    ) -> Self {
        Self::BabyBearMds(Plonky3Poseidon2Executor::new(
            external_constants,
            internal_constants,
        ))
    }
}

pub type Plonky3Poseidon2Executor<F, LinearLayers> = Poseidon2<
    <F as Field>::Packing,
    Poseidon2ExternalLayer<F, LinearLayers, POSEIDON2_WIDTH>,
    Poseidon2InternalLayer<F, LinearLayers>,
    POSEIDON2_WIDTH,
    BABY_BEAR_POSEIDON2_SBOX_DEGREE,
>;
