use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;
use p3_field::AbstractField;
use p3_keccak_air::NUM_ROUNDS;

use crate::cpu::{KECCAK_PERMUTE_BUS, MEMORY_BUS};

use super::{columns::KeccakPermuteCols, KeccakPermuteAir, NUM_U64_HASH_ELEMS};

impl KeccakPermuteAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakPermuteCols<AB::Var>,
    ) {
        let input = local.keccak.preimage.into_iter().flatten().flatten();
        let output = (0..5).flat_map(move |x| {
            (0..5).flat_map(move |y| {
                (0..NUM_U64_HASH_ELEMS).map(move |limb| {
                    // TODO: after switching to latest p3 commit, this should be y, x
                    local.keccak.a_prime_prime_prime(x, y, limb)
                })
            })
        });
        let is_input = local.is_direct * local.keccak.step_flags[0];
        builder.push_receive(self.input_bus, input, is_input);

        let is_output = local.is_direct * local.keccak.step_flags[NUM_ROUNDS - 1];
        builder.push_send(self.output_bus, output, is_output);

        builder.push_receive(
            KECCAK_PERMUTE_BUS,
            [
                local.clk.into(),
                local.a.into(),
                AB::Expr::zero(),
                local.c.into(),
                local.d.into(),
                local.e.into(),
            ],
            local.is_opcode,
        );

        let mut timestamp_offset = 0;
        // read addresses
        for (io_addr, aux_addr) in [local.a, local.c]
            .into_iter()
            .zip_eq([local.dst, local.src])
        {
            let timestamp = local.clk + AB::F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let fields = [
                timestamp,
                AB::Expr::from_bool(false), // read
                local.d.into(),
                io_addr.into(),
                aux_addr.into(),
            ];
            builder.push_send(MEMORY_BUS, fields, local.is_opcode - local.d_is_zero);
        }

        // READ
        for i in 0..WIDTH {
            let timestamp = local.clk + AB::F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let address = local.src + AB::F::from_canonical_usize(i);

            let fields = [
                timestamp,
                AB::Expr::from_bool(false), // read
                local.e.into(),
                address,
                local.inner.io.input[i].into(),
            ];

            builder.push_send(MEMORY_BUS, fields, local.is_opcode);
        }

        // WRITE
        for i in 0..WIDTH {
            let timestamp = local.clk + AB::F::from_canonical_usize(timestamp_offset);
            timestamp_offset += 1;

            let address = local.dst + AB::F::from_canonical_usize(i);

            let fields = [
                timestamp,
                AB::Expr::from_bool(true), // write
                local.e.into(),
                address,
                local.inner.io.output[i].into(),
            ];

            let count = local.is_opcode.into();
            builder.push_send(MEMORY_BUS, fields, count);
        }
    }
}
