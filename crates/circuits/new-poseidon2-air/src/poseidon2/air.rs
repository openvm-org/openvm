use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_poseidon2_air::{Poseidon2Air, Poseidon2Cols};

use super::{
    HlMdsMatrixConfig, MdsMatrixConfig, Plonky3RoundConstants, Poseidon2LinearLayers,
    Poseidon2Matrix, POSEIDON2_HALF_FULL_ROUNDS, POSEIDON2_PARTIAL_ROUNDS, POSEIDON2_SBOX_DEGREE,
    POSEIDON2_WIDTH,
};

pub type Poseidon2SubCols<F, const SBOX_REGISTERS: usize> = Poseidon2Cols<
    F,
    POSEIDON2_WIDTH,
    POSEIDON2_SBOX_DEGREE,
    SBOX_REGISTERS,
    POSEIDON2_HALF_FULL_ROUNDS,
    POSEIDON2_PARTIAL_ROUNDS,
>;

pub type Plonky3Poseidon2Air<F, LinearLayers, const SBOX_REGISTERS: usize> = Poseidon2Air<
    F,
    LinearLayers,
    POSEIDON2_WIDTH,
    POSEIDON2_SBOX_DEGREE,
    SBOX_REGISTERS,
    POSEIDON2_HALF_FULL_ROUNDS,
    POSEIDON2_PARTIAL_ROUNDS,
>;

pub const fn sbox_registers(max_constraint_degree: usize) -> usize {
    if max_constraint_degree < 7 {
        1
    } else {
        0
    }
}

#[derive(Debug)]
pub enum Poseidon2SubAir<F: Field, const SBOX_REGISTERS: usize> {
    BabyBearMds(Plonky3Poseidon2Air<F, Poseidon2LinearLayers<MdsMatrixConfig>, SBOX_REGISTERS>),
    BabyBearHlMds(Plonky3Poseidon2Air<F, Poseidon2LinearLayers<HlMdsMatrixConfig>, SBOX_REGISTERS>),
}

impl<F: Field, const SBOX_REGISTERS: usize> Poseidon2SubAir<F, SBOX_REGISTERS> {
    pub fn new(matrix: Poseidon2Matrix, constants: Plonky3RoundConstants<F>) -> Self {
        match matrix {
            Poseidon2Matrix::MdsMatrix => Self::BabyBearMds(Plonky3Poseidon2Air::new(constants)),
            Poseidon2Matrix::HlMdsMatrix => {
                Self::BabyBearHlMds(Plonky3Poseidon2Air::new(constants))
            }
        }
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAir<F> for Poseidon2SubAir<F, SBOX_REGISTERS> {
    fn width(&self) -> usize {
        match self {
            Self::BabyBearMds(air) => air.width(),
            Self::BabyBearHlMds(air) => air.width(),
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
        println!("eval");
        match self {
            Self::BabyBearMds(air) => air.eval(builder),
            Self::BabyBearHlMds(air) => air.eval(builder),
        }
    }
}
