use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::Field;

use crate::poseidon2::Poseidon2Air;

use super::columns::Poseidon2IoCols;

// Receives input and output columns in one interaction
impl<const WIDTH: usize, F: Field> Poseidon2Air<WIDTH, F> {
    pub fn eval_interactions<AB: InteractionBuilder<F = F>>(
        &self,
        builder: &mut AB,
        io: Poseidon2IoCols<WIDTH, AB::Var>,
    ) {
        let fields = io.input.into_iter().chain(io.output);
        builder.push_receive(self.bus_index, fields, F::one());
    }
}
