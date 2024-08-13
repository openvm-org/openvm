use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_field::Field;

use super::{
    columns::{Poseidon2VmAuxCols, Poseidon2VmIoCols},
    Poseidon2VmAir,
};
use crate::cpu::{MEMORY_BUS, POSEIDON2_BUS, POSEIDON2_DIRECT_BUS};
use crate::memory::{MemoryAccess, OpType};

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
        let fields = io.flatten().into_iter().skip(2);
        builder.push_receive(POSEIDON2_BUS, fields, io.is_opcode);

        let chunks: usize = WIDTH / 2;

        let mut timestamp_offset = 0;
        // read addresses when is_opcode:
        // dst <- [a]_d, lhs <- [b]_d
        // Only when opcode is COMPRESS is rhs <- [c]_d read
        for (io_addr, aux_addr, count) in izip!(
            [io.a, io.b, io.c],
            [aux.dst, aux.lhs, aux.rhs],
            [io.is_opcode, io.is_opcode, io.cmp]
        ) {
            let timestamp = io.clk + F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let access = MemoryAccess {
                timestamp,
                op_type: OpType::Read,
                address_space: io.d.into(),
                address: io_addr.into(),
                data: [aux_addr.into()],
            };

            MEMORY_BUS.send_interaction(builder, access, count);
        }

        // READ
        for i in 0..WIDTH {
            let timestamp = io.clk + F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let address = if i < chunks { aux.lhs } else { aux.rhs }
                + F::from_canonical_usize(if i < chunks { i } else { i - chunks });

            let access = MemoryAccess {
                timestamp,
                op_type: OpType::Read,
                address_space: io.e.into(),
                address,
                data: [aux.internal.io.input[i].into()],
            };

            let count = io.is_opcode;
            MEMORY_BUS.send_interaction(builder, access, count);
        }

        // WRITE
        for i in 0..WIDTH {
            let timestamp = io.clk + F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let address = aux.dst + F::from_canonical_usize(i);

            let access = MemoryAccess {
                timestamp,
                op_type: OpType::Write,
                address_space: io.e.into(),
                address,
                data: [aux.internal.io.output[i].into()],
            };

            let count = if i < chunks {
                io.is_opcode.into()
            } else {
                io.is_opcode - io.cmp
            };

            MEMORY_BUS.send_interaction(builder, access, count);
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
