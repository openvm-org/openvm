use std::{borrow::Borrow, sync::Arc};

use openvm_circuit_primitives::{AlignedBorrow, ColumnsAir};
use openvm_poseidon2_air::{
    Poseidon2Config, Poseidon2SubAir, Poseidon2SubCols, BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS,
};
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder,
    interaction::{InteractionBuilder, LookupBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use super::SBOX_REGISTERS;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Poseidon2PeripheryCols<T> {
    pub inner: Poseidon2SubCols<T, SBOX_REGISTERS>,
    pub mult: T,
}

pub struct Poseidon2PeripheryAir<F: Field> {
    pub subair: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
    pub bus: LookupBus,
}

// No columns provided: `Poseidon2PeripheryCols` embeds external `Poseidon2SubCols` which doesn't
// derive `StructReflection`.
impl<F: Field> ColumnsAir for Poseidon2PeripheryAir<F> {}

impl<F: Field> Poseidon2PeripheryAir<F> {
    pub fn new(config: Poseidon2Config<F>, bus: LookupBus) -> Self {
        Self {
            subair: Arc::new(Poseidon2SubAir::new(config.constants.into())),
            bus,
        }
    }
}

impl<F: Field> BaseAir<F> for Poseidon2PeripheryAir<F> {
    fn width(&self) -> usize {
        Poseidon2PeripheryCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for Poseidon2PeripheryAir<F> {}
impl<F: Field> PartitionedBaseAir<F> for Poseidon2PeripheryAir<F> {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for Poseidon2PeripheryAir<AB::F> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main
            .row_slice(0)
            .expect("window should have at least one row");
        let local: &Poseidon2PeripheryCols<AB::Var> = (*local).borrow();

        let mut sub_builder =
            SubAirBuilder::<AB, Poseidon2SubAir<AB::F, SBOX_REGISTERS>, AB::F>::new(
                builder,
                0..self.subair.width(),
            );
        self.subair.eval(&mut sub_builder);

        let inputs = local.inner.inputs;
        let outputs =
            &local.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1].post;

        // Send the `[input || output]` state on the direct bus so the adapter can look it up.
        self.bus.add_key_with_lookups(
            builder,
            inputs
                .into_iter()
                .map(Into::into)
                .chain(outputs.iter().copied().map(Into::into)),
            local.mult,
        );
    }
}
