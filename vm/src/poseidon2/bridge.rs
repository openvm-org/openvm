use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::Field;

use super::{
    air::Poseidon2VmAir,
    columns::{Poseidon2VmAuxCols, Poseidon2VmIoCols},
};
use crate::cpu::{POSEIDON2_BUS, POSEIDON2_DIRECT_BUS};

impl<const WIDTH: usize, const WORD_SIZE: usize, F: Field> Poseidon2VmAir<WIDTH, WORD_SIZE, F> {
    /// Receives instructions from the CPU on the designated `POSEIDON2_BUS` (opcodes) or `POSEIDON2_DIRECT_BUS` (direct), and sends both read and write requests to the memory chip.
    ///
    /// Receives (clk, a, b, c, d, e, cmp) for opcodes, width exposed in `opcode_interaction_width()`
    ///
    /// Receives (hash_in.0, hash_in.1, hash_out) for direct, width exposed in `direct_interaction_width()`
    pub fn eval_interactions<AB: InteractionBuilder<F = F>>(
        &self,
        builder: &mut AB,
        io: Poseidon2VmIoCols<AB::Var>,
        aux: &Poseidon2VmAuxCols<WIDTH, WORD_SIZE, AB::Var>,
    ) {
        let fields = io.flatten().into_iter().skip(2);
        builder.push_receive(POSEIDON2_BUS, fields, io.is_opcode);

        // DIRECT
        if self.direct {
            let expand_fields = aux
                .internal
                .io
                .flatten()
                .into_iter()
                .take(WIDTH + WIDTH / 2)
                .collect::<Vec<AB::Var>>();

            builder.push_receive(POSEIDON2_DIRECT_BUS, expand_fields, io.is_direct);
        }
    }
}
