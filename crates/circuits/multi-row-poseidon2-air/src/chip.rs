use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use super::{
    BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS, BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
    BABY_BEAR_POSEIDON2_SBOX_DEGREE, POSEIDON2_WIDTH,
};
use crate::{
    air::MultiRowPoseidon2Air as BasePoseidon2Air, columns::MultiRowPoseidon2Cols,
    config::Plonky3RoundConstants, permute::BabyBearPoseidon2LinearLayers,
};

pub type MultiRowPoseidon2Air<F, LinearLayers, const SBOX_REGISTERS: usize> = BasePoseidon2Air<
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
    BabyBearMds(MultiRowPoseidon2Air<F, BabyBearPoseidon2LinearLayers, SBOX_REGISTERS>),
}

impl<F: Field, const SBOX_REGISTERS: usize> Poseidon2SubAir<F, SBOX_REGISTERS> {
    pub fn new(constants: Plonky3RoundConstants<F>) -> Self {
        Self::BabyBearMds(MultiRowPoseidon2Air::new(constants))
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAir<F> for Poseidon2SubAir<F, SBOX_REGISTERS> {
    fn width(&self) -> usize {
        match self {
            Self::BabyBearMds(air) => air.width(),
        }
    }

    // fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
    //     let rows: Vec<_> = (0..(1 << M) * (1 << M))
    //         .flat_map(|i| {
    //             let x = i / (1 << M);
    //             let y = i % (1 << M);
    //             let z = x ^ y;
    //             [x, y, z].map(F::from_canonical_u32)
    //         })
    //         .collect();

    //     Some(RowMajorMatrix::new(rows, NUM_XOR_LOOKUP_PREPROCESSED_COLS))
    // }
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
