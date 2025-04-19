use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_poseidon2_air::{Poseidon2Air, Poseidon2Cols};

use super::{POSEIDON2_HALF_FULL_ROUNDS, POSEIDON2_WIDTH};
use crate::{BabyBearPoseidon2LinearLayers, KoalaBearPoseidon2LinearLayers, Plonky3RoundConstants};

pub type Poseidon2SubCols<
    F,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const PARTIAL_ROUNDS: usize,
> = Poseidon2Cols<
    F,
    POSEIDON2_WIDTH,
    SBOX_DEGREE,
    SBOX_REGISTERS,
    POSEIDON2_HALF_FULL_ROUNDS,
    PARTIAL_ROUNDS,
>;

pub type Plonky3Poseidon2Air<
    F,
    LinearLayers,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const PARTIAL_ROUNDS: usize,
> = Poseidon2Air<
    F,
    LinearLayers,
    POSEIDON2_WIDTH,
    SBOX_DEGREE,
    SBOX_REGISTERS,
    POSEIDON2_HALF_FULL_ROUNDS,
    PARTIAL_ROUNDS,
>;

#[derive(Debug)]
pub enum Poseidon2SubAir<
    F: Field,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    BabyBearMds(
        Plonky3Poseidon2Air<
            F,
            BabyBearPoseidon2LinearLayers,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            PARTIAL_ROUNDS,
        >,
    ),
    KoalaBearMds(
        Plonky3Poseidon2Air<
            F,
            KoalaBearPoseidon2LinearLayers,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            PARTIAL_ROUNDS,
        >,
    ),
}

impl<
        F: Field,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const PARTIAL_ROUNDS: usize,
    > Poseidon2SubAir<F, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>
{
    pub fn new(constants: Plonky3RoundConstants<F, PARTIAL_ROUNDS>) -> Self {
        Self::KoalaBearMds(Plonky3Poseidon2Air::new(constants))
    }

    pub fn new_babybear(constants: Plonky3RoundConstants<F, PARTIAL_ROUNDS>) -> Self {
        Self::BabyBearMds(Plonky3Poseidon2Air::new(constants))
    }
}

impl<
        F: Field,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const PARTIAL_ROUNDS: usize,
    > BaseAir<F> for Poseidon2SubAir<F, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>
{
    fn width(&self) -> usize {
        match self {
            Self::BabyBearMds(air) => air.width(),
            Self::KoalaBearMds(air) => air.width(),
        }
    }
}

impl<
        F: Field,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const PARTIAL_ROUNDS: usize,
    > BaseAirWithPublicValues<F>
    for Poseidon2SubAir<F, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>
{
}
impl<
        F: Field,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const PARTIAL_ROUNDS: usize,
    > PartitionedBaseAir<F> for Poseidon2SubAir<F, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>
{
}

impl<
        AB: AirBuilder,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const PARTIAL_ROUNDS: usize,
    > Air<AB> for Poseidon2SubAir<AB::F, SBOX_DEGREE, SBOX_REGISTERS, PARTIAL_ROUNDS>
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::BabyBearMds(air) => air.eval(builder),
            Self::KoalaBearMds(air) => air.eval(builder),
        }
    }
}
