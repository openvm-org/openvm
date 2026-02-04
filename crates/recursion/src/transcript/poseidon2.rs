use core::borrow::Borrow;
use std::{array::from_fn, sync::Arc};

use openvm_poseidon2_air::{
    BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS, Poseidon2SubAir, Poseidon2SubCols,
};
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder,
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::bus::{
    Poseidon2CompressBus, Poseidon2CompressMessage, Poseidon2PermuteBus, Poseidon2PermuteMessage,
};

pub const CHUNK: usize = 8;
pub use openvm_poseidon2_air::POSEIDON2_WIDTH;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Poseidon2Cols<T, const SBOX_REGISTERS: usize> {
    pub inner: Poseidon2SubCols<T, SBOX_REGISTERS>,
    pub permute_mult: T,
    pub compress_mult: T,
}

pub struct Poseidon2Air<F: Field, const SBOX_REGISTERS: usize> {
    pub subair: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
    pub poseidon2_permute_bus: Poseidon2PermuteBus,
    pub poseidon2_compress_bus: Poseidon2CompressBus,
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAir<F> for Poseidon2Air<F, SBOX_REGISTERS> {
    fn width(&self) -> usize {
        Poseidon2Cols::<F, SBOX_REGISTERS>::width()
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAirWithPublicValues<F>
    for Poseidon2Air<F, SBOX_REGISTERS>
{
}
impl<F: Field, const SBOX_REGISTERS: usize> PartitionedBaseAir<F>
    for Poseidon2Air<F, SBOX_REGISTERS>
{
}

impl<AB: AirBuilder + InteractionBuilder, const SBOX_REGISTERS: usize> Air<AB>
    for Poseidon2Air<AB::F, SBOX_REGISTERS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, _next) = (main.row_slice(0), main.row_slice(1));
        let local: &Poseidon2Cols<AB::Var, SBOX_REGISTERS> = (*local).borrow();

        let mut sub_builder =
            SubAirBuilder::<AB, Poseidon2SubAir<AB::F, SBOX_REGISTERS>, AB::F>::new(
                builder,
                0..self.subair.width(),
            );
        self.subair.eval(&mut sub_builder);

        self.poseidon2_permute_bus.add_key_with_lookups(
            builder,
            Poseidon2PermuteMessage {
                input: local.inner.inputs,
                output: local.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1]
                    .post,
            },
            local.permute_mult,
        );

        self.poseidon2_compress_bus.add_key_with_lookups(
            builder,
            Poseidon2CompressMessage {
                input: local.inner.inputs,
                output: from_fn(|i| {
                    local.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1].post[i]
                }),
            },
            local.compress_mult,
        );
    }
}
