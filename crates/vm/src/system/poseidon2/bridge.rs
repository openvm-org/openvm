use ax_poseidon2_air::poseidon2::columns::Poseidon2IoCols;
use ax_stark_backend::{
    interaction::InteractionBuilder,
    p3_field::{AbstractField, Field},
};

use super::{air::Poseidon2VmAir, columns::Poseidon2VmIoCols, WIDTH};
use crate::arch::{instructions::Poseidon2Opcode::PERM_POS2, ExecutionState};

impl<F: Field> Poseidon2VmAir<F> {
    /// Receives instructions from the Core on the designated `POSEIDON2_BUS` (opcodes) or `POSEIDON2_DIRECT_BUS` (direct), and sends both read and write requests to the memory chip.
    ///
    /// Receives (clk, a, b, c, d, e, cmp) for opcodes, width exposed in `opcode_interaction_width()`
    ///
    /// Receives (hash_in.0, hash_in.1, hash_out) for direct, width exposed in `direct_interaction_width()`
    pub fn eval_interactions<AB: InteractionBuilder<F = F>>(
        &self,
        builder: &mut AB,
        io: Poseidon2VmIoCols<AB::Var>,
        internal_io: Poseidon2IoCols<WIDTH, AB::Var>,
        timestamp_delta: AB::Expr,
    ) {
        let opcode = AB::Expr::from_canonical_usize(PERM_POS2 as usize) + io.is_compress_opcode;

        self.execution_bridge
            .execute_and_increment_pc(
                opcode + AB::Expr::from_canonical_usize(self.offset),
                [io.a, io.b, io.c, io.d, io.e],
                ExecutionState::new(io.pc, io.timestamp),
                timestamp_delta,
            )
            .eval(builder, io.is_opcode);

        // DIRECT
        if let Some(direct_bus) = self.direct_bus {
            let fields = internal_io
                .flatten()
                .into_iter()
                .take(WIDTH + WIDTH / 2)
                .collect::<Vec<AB::Var>>();

            builder.push_receive(direct_bus, fields, io.is_compress_direct);
        }
    }
}
