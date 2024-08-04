use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;
use p3_field::{AbstractField, Field};

use super::{
    columns::{Poseidon2VmAuxCols, Poseidon2VmIoCols},
    Poseidon2VmAir,
};
use crate::cpu::{MEMORY_BUS, POSEIDON2_BUS, POSEIDON2_DIRECT_BUS};

impl<const WIDTH: usize, F: Field> Poseidon2VmAir<WIDTH, F> {
    /// Receives instructions from the CPU on the designated `POSEIDON2_BUS` (opcodes) or `POSEIDON2_DIRECT_BUS` (direct), and sends both read and write requests to the memory chip.
    ///
    /// Receives (clk, a, b, c, d, e, cmp) for opcodes, width exposed in `opcode_interaction_width()`
    ///
    /// Receives (hash_in.0, hash_in.1, hash_out) for direct, width exposed in `direct_interaction_width()`
    pub fn eval_interactions<AB: InteractionBuilder<F = F>>(
        &self,
        builder: &mut AB,
        io: Poseidon2VmIoCols<AB::Var>,
        aux: &Poseidon2VmAuxCols<WIDTH, AB::Var>,
    ) {
        let d_is_zero = aux.d_is_zero;

        let fields = io.flatten().into_iter().skip(2);
        builder.push_receive(POSEIDON2_BUS, fields, io.is_opcode);

        let chunks: usize = WIDTH / 2;

        let mut timestamp_offset = 0;
        // read addresses
        for (io_addr, aux_addr) in [io.a, io.b, io.c]
            .into_iter()
            .zip_eq([aux.dst, aux.lhs, aux.rhs])
        {
            let timestamp = io.clk + AB::F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let fields = [
                timestamp,
                AB::Expr::from_bool(false),
                io.d.into(),
                io_addr.into(),
                aux_addr.into(),
            ];
            builder.push_send(MEMORY_BUS, fields, io.is_opcode - d_is_zero);
        }

        // READ
        for i in 0..WIDTH {
            let timestamp = io.clk + AB::F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let address = if i < chunks { aux.lhs } else { aux.rhs }
                + F::from_canonical_usize(if i < chunks { i } else { i - chunks });

            let fields = [
                timestamp,
                AB::Expr::from_bool(false),
                io.e.into(),
                address,
                aux.internal.io.input[i].into(),
            ];

            builder.push_send(MEMORY_BUS, fields, io.is_opcode);
        }

        // WRITE
        for i in 0..WIDTH {
            let timestamp = io.clk + AB::F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let address = aux.dst + AB::F::from_canonical_usize(i);

            let fields = [
                timestamp,
                AB::Expr::from_bool(true),
                io.e.into(),
                address,
                aux.internal.io.output[i].into(),
            ];

            let count = if i < chunks {
                io.is_opcode.into()
            } else {
                io.is_opcode - io.cmp
            };

            builder.push_send(MEMORY_BUS, fields, count);
        }

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
