use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::{AbstractField, Field};
use poseidon2_air::poseidon2::columns::Poseidon2IoCols;

use super::{air::Poseidon2VmAir, columns::Poseidon2VmIoCols, WIDTH};
use crate::{
    arch::{
        columns::{ExecutionState, InstructionCols},
        instructions::Opcode::PERM_POS2,
    },
    cpu::POSEIDON2_DIRECT_BUS,
};

impl<F: Field> Poseidon2VmAir<F> {
    /// Receives instructions from the CPU on the designated `POSEIDON2_BUS` (opcodes) or `POSEIDON2_DIRECT_BUS` (direct), and sends both read and write requests to the memory chip.
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
        let opcode = AB::Expr::from_canonical_usize(PERM_POS2 as usize) + io.cmp;
        self.execution_bus.execute_increment_pc(
            builder,
            io.is_opcode,
            ExecutionState::new(io.pc, io.timestamp),
            timestamp_delta,
            InstructionCols::new(opcode, [io.a, io.b, io.c, io.d, io.e]),
        );

        // DIRECT
        if self.direct {
            let expand_fields = internal_io
                .flatten()
                .into_iter()
                .take(WIDTH + WIDTH / 2)
                .collect::<Vec<AB::Var>>();

            builder.push_receive(POSEIDON2_DIRECT_BUS, expand_fields, io.is_direct);
        }
    }
}
