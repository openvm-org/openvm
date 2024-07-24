use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::{AbstractField, Field};

use crate::cpu::{MEMORY_BUS, POSEIDON2_BUS};

use super::columns::{Poseidon2VmAuxCols, Poseidon2VmIoCols};
use super::Poseidon2VmAir;

impl<const WIDTH: usize, F: Field> Poseidon2VmAir<WIDTH, F> {
    /// Receives instructions from the CPU on the designated `POSEIDON2_BUS`, and sends both read and write requests to the memory chip.
    /// Receives (clk, a, b, c, d, e, cmp)
    pub fn eval_interactions<AB: InteractionBuilder<F = F>>(
        &self,
        builder: &mut AB,
        io: Poseidon2VmIoCols<AB::Var>,
        aux: &Poseidon2VmAuxCols<WIDTH, AB::Var>,
    ) {
        let addresses = aux.addresses;
        let d_is_zero = aux.d_is_zero;

        let fields = io.flatten().into_iter().skip(1);
        builder.push_receive(POSEIDON2_BUS, fields, io.is_alloc);

        let chunks: usize = WIDTH / 2;

        let mut timestamp_offset = 0;
        // read addresses
        for (i, addr) in [io.a, io.b, io.c].into_iter().enumerate() {
            let timestamp = io.clk + AB::F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let fields = [
                timestamp,
                AB::Expr::from_bool(false),
                io.d.into(),
                addr.into(),
                addresses[i].into(),
            ];
            builder.push_send(MEMORY_BUS, fields, io.is_alloc - d_is_zero);
        }

        // READ
        for i in 0..WIDTH {
            let timestamp = io.clk + AB::F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let address = if i < chunks {
                addresses[0]
            } else {
                addresses[1]
            } + AB::F::from_canonical_usize(if i < chunks { i } else { i - chunks });

            let fields = [
                timestamp,
                AB::Expr::from_bool(false),
                io.e.into(),
                address,
                aux.internal.io.input[i].into(),
            ];

            builder.push_send(MEMORY_BUS, fields, io.is_alloc);
        }

        // WRITE
        for i in 0..WIDTH {
            let timestamp = io.clk + AB::F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let address = aux.addresses[2] + AB::F::from_canonical_usize(i);

            let fields = [
                timestamp,
                AB::Expr::from_bool(true),
                io.e.into(),
                address,
                aux.internal.io.output[i].into(),
            ];

            let count = if i < chunks {
                io.is_alloc.into()
            } else {
                io.is_alloc - io.cmp
            };

            builder.push_send(MEMORY_BUS, fields, count);
        }
    }
}
