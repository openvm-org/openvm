use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_poseidon2_air::{Poseidon2Air, Poseidon2Cols};

use super::{
    BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS, BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
    BABY_BEAR_POSEIDON2_SBOX_DEGREE, POSEIDON2_WIDTH,
};
use crate::{BabyBearPoseidon2LinearLayers, Plonky3RoundConstants};

pub type Poseidon2SubCols<F, const SBOX_REGISTERS: usize> = Poseidon2Cols<
    F,
    POSEIDON2_WIDTH,
    BABY_BEAR_POSEIDON2_SBOX_DEGREE,
    SBOX_REGISTERS,
    BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS,
    BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
>;

pub type Plonky3Poseidon2Air<F, LinearLayers, const SBOX_REGISTERS: usize> = Poseidon2Air<
    F,
    LinearLayers,
    POSEIDON2_WIDTH,
    BABY_BEAR_POSEIDON2_SBOX_DEGREE,
    SBOX_REGISTERS,
    BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS,
    BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
>;

#[derive(Debug)]
pub enum Poseidon2SubAir<F: Field, const SBOX_REGISTERS: usize> {
    BabyBearMds(Plonky3Poseidon2Air<F, BabyBearPoseidon2LinearLayers, SBOX_REGISTERS>),
}

impl<F: Field, const SBOX_REGISTERS: usize> Poseidon2SubAir<F, SBOX_REGISTERS> {
    pub fn new(constants: Plonky3RoundConstants<F>) -> Self {
        Self::BabyBearMds(Plonky3Poseidon2Air::new(constants))
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAir<F> for Poseidon2SubAir<F, SBOX_REGISTERS> {
    fn width(&self) -> usize {
        match self {
            Self::BabyBearMds(air) => air.width(),
        }
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAirWithPublicValues<F>
    for Poseidon2SubAir<F, SBOX_REGISTERS>
{
}
impl<F: Field, const SBOX_REGISTERS: usize> PartitionedBaseAir<F>
    for Poseidon2SubAir<F, SBOX_REGISTERS>
{
}

impl<AB: AirBuilder, const SBOX_REGISTERS: usize> Air<AB>
    for Poseidon2SubAir<AB::F, SBOX_REGISTERS>
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::BabyBearMds(air) => air.eval(builder),
        }
    }
}
