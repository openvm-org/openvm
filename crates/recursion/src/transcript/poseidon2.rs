use core::borrow::Borrow;

use crate::bus::Poseidon2Bus;
use crate::bus::Poseidon2BusMessage;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

pub const CHUNK: usize = 8;
pub use openvm_poseidon2_air::POSEIDON2_WIDTH;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Poseidon2Cols<T, const SBOX_REGISTERS: usize> {
    // TODO: use inner to constraint permutation
    // pub inner: Poseidon2SubCols<T, SBOX_REGISTERS>,
    pub input: [T; POSEIDON2_WIDTH],
    pub output: [T; POSEIDON2_WIDTH],
    pub mult: T,
}

pub struct Poseidon2Air {
    pub poseidon2_bus: Poseidon2Bus,
}

impl<F: Field> BaseAir<F> for Poseidon2Air {
    fn width(&self) -> usize {
        // TODO: what's sbox_registers?
        Poseidon2Cols::<F, 0>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Poseidon2Air {}
impl<F: Field> PartitionedBaseAir<F> for Poseidon2Air {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for Poseidon2Air {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, _next) = (main.row_slice(0), main.row_slice(1));
        let local: &Poseidon2Cols<AB::Var, 0> = (*local).borrow();

        self.poseidon2_bus.add_key_with_lookups(
            builder,
            Poseidon2BusMessage {
                input: local.input,
                output: local.output,
            },
            local.mult,
        )
    }
}
